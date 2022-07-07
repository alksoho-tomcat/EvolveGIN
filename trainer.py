import torch
import utils as u
import logger
import time
import pandas as pd
import numpy as np

class Trainer():
	def __init__(self,args, splitter, gcn, classifier, comp_loss, dataset, num_classes):
		# run_exp.pyで代入される

		self.args = args
		# splistter.pyで定義
		self.splitter = splitter
		# taskにより変化
		self.tasker = splitter.tasker
		# run_exp.pyで build_gcn(args, tasker)で定義
		self.gcn = gcn
		# run_exp.py build_classifierで定義
		self.classifier = classifier
		# comp_lossはrun_exp.pyでcross_entropyを選択
		self.comp_loss = comp_loss

		# データ系 run_exp.pyで代入されたもの
		self.num_nodes = dataset.num_nodes
		self.data = dataset
		self.num_classes = num_classes
		
		# ログ系 logger.pyで定義
		self.logger = logger.Logger(args, self.num_classes)

		self.init_optimizers(args)

		if self.tasker.is_static:
			adj_matrix = u.sparse_prepare_tensor(self.tasker.adj_matrix, torch_size = [self.num_nodes], ignore_batch_dim = False)
			self.hist_adj_list = [adj_matrix]
			self.hist_ndFeats_list = [self.tasker.nodes_feats.float()]
	# 最適化の初期化?
	def init_optimizers(self,args):
		params = self.gcn.parameters()
		self.gcn_opt = torch.optim.Adam(params, lr = args.learning_rate)
		params = self.classifier.parameters()
		self.classifier_opt = torch.optim.Adam(params, lr = args.learning_rate)
		self.gcn_opt.zero_grad()
		self.classifier_opt.zero_grad()

	# チェックポイントを保存
	def save_checkpoint(self, state, filename='checkpoint.pth.tar'):
		torch.save(state, filename)

	# 保存したチェックポイントをロード
	def load_checkpoint(self, filename, model):
		if os.path.isfile(filename):
			print("=> loading checkpoint '{}'".format(filename))
			checkpoint = torch.load(filename)
			epoch = checkpoint['epoch']
			self.gcn.load_state_dict(checkpoint['gcn_dict'])
			self.classifier.load_state_dict(checkpoint['classifier_dict'])
			self.gcn_opt.load_state_dict(checkpoint['gcn_optimizer'])
			self.classifier_opt.load_state_dict(checkpoint['classifier_optimizer'])
			self.logger.log_str("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']))
			return epoch
		else:
			self.logger.log_str("=> no checkpoint found at '{}'".format(filename))
			return 0

	# 学習
	def train(self):
		self.tr_step = 0		# 学習ステップ回数
		best_eval_valid = 0		# validationデータの最高スコア
		eval_valid = 0			# validationデータのスコア
		epochs_without_impr = 0	# early_stopを決定するカウンタ(フラグ)

		# エポックカウント
		for e in range(self.args.num_epochs):
			eval_train, nodes_embs = self.run_epoch(self.splitter.train, e, 'TRAIN', grad = True)
			# 検証データ(splitter.dev)での検証 (eval_after_epochsはyamlで定義、設定されたエポック以降でvalidationする)
			if len(self.splitter.dev)>0 and e>self.args.eval_after_epochs:
				eval_valid, _ = self.run_epoch(self.splitter.dev, e, 'VALID', grad = False)
				if eval_valid>best_eval_valid:	# ベストなvalidationスコアを取得
					best_eval_valid = eval_valid
					epochs_without_impr = 0
					print ('### w'+str(self.args.rank)+') ep '+str(e)+' - Best valid measure:'+str(eval_valid))
				else:	# アーリーストップをするかを判定
					epochs_without_impr+=1
					if epochs_without_impr>self.args.early_stop_patience:
						print ('### w'+str(self.args.rank)+') ep '+str(e)+' - Early stop.')
						break

			if len(self.splitter.test)>0 and eval_valid==best_eval_valid and e>self.args.eval_after_epochs:
				eval_test, _ = self.run_epoch(self.splitter.test, e, 'TEST', grad = False)

				if self.args.save_node_embeddings:
					self.save_node_embs_csv(nodes_embs, self.splitter.train_idx, log_file+'_train_nodeembs.csv.gz')
					self.save_node_embs_csv(nodes_embs, self.splitter.dev_idx, log_file+'_valid_nodeembs.csv.gz')
					self.save_node_embs_csv(nodes_embs, self.splitter.test_idx, log_file+'_test_nodeembs.csv.gz')


	def run_epoch(self, split, epoch, set_name, grad):
		t0 = time.time()
		# 999回回したときにログを取る
		log_interval=999
		if set_name=='TEST':
			log_interval=1
		self.logger.log_epoch_start(epoch, len(split), set_name, minibatch_log_interval=log_interval)
		
		# 勾配を計算するかどうか(True, False)
		torch.set_grad_enabled(grad)

		# データセットからサンプリング
		for s in split:
			if self.tasker.is_static:
				s = self.prepare_static_sample(s)
			else:
				s = self.prepare_sample(s)

			predictions, nodes_embs = self.predict(s.hist_adj_list,
												   s.hist_ndFeats_list,
												   s.label_sp['idx'],
												   s.node_mask_list)
			# validationのラベルから損失を計算 self.comp_lossはcross_entropy
			loss = self.comp_loss(predictions,s.label_sp['vals'])
			# print(loss)
			if set_name in ['TEST', 'VALID'] and self.args.task == 'link_pred':
				self.logger.log_minibatch(predictions, s.label_sp['vals'], loss.detach(), adj = s.label_sp['idx'])
			else:
				self.logger.log_minibatch(predictions, s.label_sp['vals'], loss.detach())
			if grad:
				self.optim_step(loss)

		torch.set_grad_enabled(True)
		eval_measure = self.logger.log_epoch_done()

		return eval_measure, nodes_embs

	# 予測 run_exp.pyで定義したclassifierで予測
	def predict(self,hist_adj_list,hist_ndFeats_list,node_indices,mask_list):
		# models.pyのforwardを呼び出し
		nodes_embs = self.gcn(hist_adj_list,		# 隣接行列
							  hist_ndFeats_list,	# ノード特徴量
							  mask_list)			# マスクリスト

		predict_batch_size = 100000
		gather_predictions=[]
		for i in range(1 +(node_indices.size(1)//predict_batch_size)):
			cls_input = self.gather_node_embs(nodes_embs, node_indices[:, i*predict_batch_size:(i+1)*predict_batch_size])
			predictions = self.classifier(cls_input)
			gather_predictions.append(predictions)
		gather_predictions=torch.cat(gather_predictions, dim=0)
		return gather_predictions, nodes_embs

	# idexと埋め込みを照合、リスト化
	def gather_node_embs(self,nodes_embs,node_indices):
		cls_input = []

		for node_set in node_indices:
			cls_input.append(nodes_embs[node_set])
		return torch.cat(cls_input,dim = 1)

	# 最適化をcross_entropyを使って計算,更新
	def optim_step(self,loss):
		self.tr_step += 1
		# backwardは呼び出すことで勝手にやってくれる
		loss.backward()

		if self.tr_step % self.args.steps_accum_gradients == 0:
			# 各変数の微分情報を使って更新
			self.gcn_opt.step()
			self.classifier_opt.step()
			
			# Pytorchでは勾配を計算されたときに以前に計算した勾配がある時加算する
			# RNNでは有用だが、それ以外では違うので初期化
			self.gcn_opt.zero_grad()
			self.classifier_opt.zero_grad()


	# run_epoch内で使用 splitからサンプリングするときに使用 sampleはデータセットからサンプリングされたもの
	def prepare_sample(self,sample):
		# dictionary内のオブジェクトを参照する関数 dict['key']の代わり
		sample = u.Namespace(sample)

		# 隣接行列を参照 indexと隣接行列を取り出す iがtimeの可能性あり? 
		for i,adj in enumerate(sample.hist_adj_list):
			# print('adj in prepare_sample is ', adj)
			# # 結果 {'idx': tensor([[[  0,   0],
			# #          [  0,   2],
			# #          [  0,   3],
			# #          ...,
			# #          [999, 974],
			# #          [999, 991],
			# #          [999, 999]]]), 'vals': tensor([[0.0088, 0.0086, 0.0086,  ..., 0.0087, 0.0092, 0.0093]])}
			# print('adj[idx] size is', adj['idx'].size())
			# # 結果 adj[idx] size is torch.Size([1, 106358, 2])
			# print('adj[vals] size is', adj['vals'].size())
			# # 結果 adj[vals] size is torch.Size([1, 106358])
			# 疎なテンソルを用意, 入力された隣接行列を疎なベクトル(tonsor型)に変換
			adj = u.sparse_prepare_tensor(adj,torch_size = [self.num_nodes])
			sample.hist_adj_list[i] = adj.to(self.args.device)

			# indexから特徴量行列を抽出
			nodes = self.tasker.prepare_node_feats(sample.hist_ndFeats_list[i])
			sample.hist_ndFeats_list[i] = nodes.to(self.args.device)

			# maskを抽出
			node_mask = sample.node_mask_list[i]
			sample.node_mask_list[i] = node_mask.to(self.args.device).t() #transposed to have same dimensions as scorer

		
		label_sp = self.ignore_batch_dim(sample.label_sp)

		# この辺りはdevice.t()を行っている GPU使用のために効率化するためにリストを更新
		if self.args.task in ["link_pred", "edge_cls"]:
			label_sp['idx'] = label_sp['idx'].to(self.args.device).t()   ####### ALDO TO CHECK why there was the .t() -----> because I concatenate embeddings when there are pairs of them, the embeddings are row vectors after the transpose
		else:
			label_sp['idx'] = label_sp['idx'].to(self.args.device)

		label_sp['vals'] = label_sp['vals'].type(torch.long).to(self.args.device)
		sample.label_sp = label_sp

		return sample

	# staticでのサンプル準備
	def prepare_static_sample(self,sample):
		sample = u.Namespace(sample)

		sample.hist_adj_list = self.hist_adj_list

		sample.hist_ndFeats_list = self.hist_ndFeats_list

		label_sp = {}
		label_sp['idx'] =  [sample.idx]
		label_sp['vals'] = sample.label
		sample.label_sp = label_sp

		return sample

	# バッチの次元を無視
	def ignore_batch_dim(self,adj):
		if self.args.task in ["link_pred", "edge_cls"]:
			adj['idx'] = adj['idx'][0]
		adj['vals'] = adj['vals'][0]
		return adj

	# ノード埋め込みをCSVに保存
	def save_node_embs_csv(self, nodes_embs, indexes, file_name):
		csv_node_embs = []
		for node_id in indexes:
			orig_ID = torch.DoubleTensor([self.tasker.data.contID_to_origID[node_id]])

			csv_node_embs.append(torch.cat((orig_ID,nodes_embs[node_id].double())).detach().numpy())

		pd.DataFrame(np.array(csv_node_embs)).to_csv(file_name, header=None, index=None, compression='gzip')
		#print ('Node embs saved in',file_name)
