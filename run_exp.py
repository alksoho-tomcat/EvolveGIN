import utils as u
import torch
import torch.distributed as dist
import numpy as np
import time
import random

#datasets
import bitcoin_dl as bc
import elliptic_temporal_dl as ell_temp
import uc_irv_mess_dl as ucim
import auto_syst_dl as aus
import sbm_dl as sbm
import reddit_dl as rdt


#taskers
import link_pred_tasker as lpt
import edge_cls_tasker as ect
import node_cls_tasker as nct

#models
import models as mls
import egcn_h
import egcn_o
import egin_h


import splitter as sp
import Cross_Entropy as ce

import trainer as tr

import logger


# パラメータをintかlogscaleに変更
def random_param_value(param, param_min, param_max, type='int'):
	if str(param) is None or str(param).lower()=='none':
		if type=='int':
			return random.randrange(param_min, param_max+1)
		elif type=='logscale':
			interval=np.logspace(np.log10(param_min), np.log10(param_max), num=100)
			return np.random.choice(interval,1)[0]
		else:
			return random.uniform(param_min, param_max)
	else:
		return param

# モデルのハイパーパラメータを設定 yamlで設定したもの
# argsはyamlをパーサーに掛けたもの
def build_random_hyper_params(args):
	# args.rankはif==__name__で定義 (MPI)に関係
	# まとめて実験するようのモデル選択か?
	# 無視
	if args.model == 'all':
		model_types = ['gcn', 'egcn_o', 'egcn_h','egin_h', 'gruA', 'gruB','egcn','lstmA', 'lstmB']
		args.model=model_types[args.rank]
	elif args.model == 'all_nogcn':
		model_types = ['egcn_o', 'egcn_h','egin_h', 'gruA', 'gruB','egcn','lstmA', 'lstmB']
		args.model=model_types[args.rank]
	elif args.model == 'all_noegcn3':
		model_types = ['gcn', 'egcn_h', 'egin_h', 'gruA', 'gruB','egcn','lstmA', 'lstmB']
		args.model=model_types[args.rank]
	elif args.model == 'all_nogruA':
		model_types = ['gcn', 'egcn_o', 'egcn_h','egin_h', 'gruB','egcn','lstmA', 'lstmB']
		args.model=model_types[args.rank]
		args.model=model_types[args.rank]
	elif args.model == 'saveembs':
		model_types = ['gcn', 'gcn', 'skipgcn', 'skipgcn']
		args.model=model_types[args.rank]

	# learning_rate, learnig_rate_min,_maxはyamlで定義
	# learning_rateをlogスケールでランダムパラメータを代入
	args.learning_rate =random_param_value(args.learning_rate, args.learning_rate_min, args.learning_rate_max, type='logscale')
	# args.adj_mat_time_window = random_param_value(args.adj_mat_time_window, args.adj_mat_time_window_min, args.adj_mat_time_window_max, type='int')

	# num_hist_stempsはyamlで定義
	if args.model == 'gcn':
		args.num_hist_steps = 0
	else:
		args.num_hist_steps = random_param_value(args.num_hist_steps, args.num_hist_steps_min, args.num_hist_steps_max, type='int')
	
	#  gcn_parametersはyaml内の gcn_parameters で定義
	args.gcn_parameters['feats_per_node'] =random_param_value(args.gcn_parameters['feats_per_node'], args.gcn_parameters['feats_per_node_min'], args.gcn_parameters['feats_per_node_max'], type='int')
	args.gcn_parameters['layer_1_feats'] =random_param_value(args.gcn_parameters['layer_1_feats'], args.gcn_parameters['layer_1_feats_min'], args.gcn_parameters['layer_1_feats_max'], type='int')
	if args.gcn_parameters['layer_2_feats_same_as_l1'] or args.gcn_parameters['layer_2_feats_same_as_l1'].lower()=='true':
		args.gcn_parameters['layer_2_feats'] = args.gcn_parameters['layer_1_feats']
	else:
		args.gcn_parameters['layer_2_feats'] =random_param_value(args.gcn_parameters['layer_2_feats'], args.gcn_parameters['layer_1_feats_min'], args.gcn_parameters['layer_1_feats_max'], type='int')
	args.gcn_parameters['lstm_l1_feats'] =random_param_value(args.gcn_parameters['lstm_l1_feats'], args.gcn_parameters['lstm_l1_feats_min'], args.gcn_parameters['lstm_l1_feats_max'], type='int')
	if args.gcn_parameters['lstm_l2_feats_same_as_l1'] or args.gcn_parameters['lstm_l2_feats_same_as_l1'].lower()=='true':
		args.gcn_parameters['lstm_l2_feats'] = args.gcn_parameters['lstm_l1_feats']
	else:
		args.gcn_parameters['lstm_l2_feats'] =random_param_value(args.gcn_parameters['lstm_l2_feats'], args.gcn_parameters['lstm_l1_feats_min'], args.gcn_parameters['lstm_l1_feats_max'], type='int')
	args.gcn_parameters['cls_feats']=random_param_value(args.gcn_parameters['cls_feats'], args.gcn_parameters['cls_feats_min'], args.gcn_parameters['cls_feats_max'], type='int')
	return args


# データセットを選択 yamlで選択
# argsはyamlをパーサーに掛けたもの
def build_dataset(args):
	if args.data == 'bitcoinotc' or args.data == 'bitcoinalpha':
		if args.data == 'bitcoinotc':
			args.bitcoin_args = args.bitcoinotc_args
		elif args.data == 'bitcoinalpha':
			args.bitcoin_args = args.bitcoinalpha_args
		return bc.bitcoin_dataset(args)
	elif args.data == 'aml_sim':
		return aml.Aml_Dataset(args)
	elif args.data == 'elliptic':
		return ell.Elliptic_Dataset(args)
	elif args.data == 'elliptic_temporal':
		return ell_temp.Elliptic_Temporal_Dataset(args)
	elif args.data == 'uc_irv_mess':
		return ucim.Uc_Irvine_Message_Dataset(args)
	elif args.data == 'dbg':
		return dbg.dbg_dataset(args)
	elif args.data == 'colored_graph':
		return cg.Colored_Graph(args)
	elif args.data == 'autonomous_syst':
		return aus.Autonomous_Systems_Dataset(args)
	elif args.data == 'reddit':
		return rdt.Reddit_Dataset(args)
	elif args.data.startswith('sbm'):
		if args.data == 'sbm20':
			args.sbm_args = args.sbm20_args
		elif args.data == 'sbm50':
			args.sbm_args = args.sbm50_args
		return sbm.sbm_dataset(args)
	else:
		raise NotImplementedError('only arxiv has been implemented')

# タスクを選択 yamlで選択
# argsはyamlにパーサーを掛けたもの、datasetは上記の関数に掛けたもの
def build_tasker(args,dataset):
	if args.task == 'link_pred':
		return lpt.Link_Pred_Tasker(args,dataset)
	elif args.task == 'edge_cls':
		return ect.Edge_Cls_Tasker(args,dataset)
	elif args.task == 'node_cls':
		return nct.Node_Cls_Tasker(args,dataset)
	elif args.task == 'static_node_cls':
		return nct.Static_Node_Cls_Tasker(args,dataset)

	else:
		raise NotImplementedError('still need to implement the other tasks')

# モデルを作成(定義はmodels.py egcn_h.py,egcn_o.py) yamlで選択
# argsはyamlをパーサーに掛けたもの、taskerは上記の関数で設定したもの
def build_gcn(args,tasker):
	# gcn_argsはyaml内 gcn_parameters で設定したもの
	gcn_args = u.Namespace(args.gcn_parameters)
	# feats_per_nodeはyaml内 gcn_parameters で設定したもの
	gcn_args.feats_per_node = tasker.feats_per_node
	
	# 静的なモデル
	if args.model == 'gcn':
		return mls.Sp_GCN(gcn_args,activation = torch.nn.RReLU()).to(args.device)
	elif args.model == 'skipgcn':
		return mls.Sp_Skip_GCN(gcn_args,activation = torch.nn.RReLU()).to(args.device)
	elif args.model == 'skipfeatsgcn':
		return mls.Sp_Skip_NodeFeats_GCN(gcn_args,activation = torch.nn.RReLU()).to(args.device)
	else:	# 動的なモデル
		assert args.num_hist_steps > 0, 'more than one step is necessary to train LSTM'
		if args.model == 'lstmA':
			return mls.Sp_GCN_LSTM_A(gcn_args,activation = torch.nn.RReLU()).to(args.device)
		elif args.model == 'gruA':
			return mls.Sp_GCN_GRU_A(gcn_args,activation = torch.nn.RReLU()).to(args.device)
		elif args.model == 'lstmB':
			return mls.Sp_GCN_LSTM_B(gcn_args,activation = torch.nn.RReLU()).to(args.device)
		elif args.model == 'gruB':
			return mls.Sp_GCN_GRU_B(gcn_args,activation = torch.nn.RReLU()).to(args.device)
		elif args.model == 'egcn':
			return egcn.EGCN(gcn_args, activation = torch.nn.RReLU()).to(args.device)
		elif args.model == 'egcn_h':	# GRU deviceはegcn_h.pyでcpuを選択
			return egcn_h.EGCN(gcn_args, activation = torch.nn.RReLU(), device = args.device)
		elif args.model == 'egin_h':	# EvolveGIN
			return egin_h.EGCN(gcn_args, activation = torch.nn.RReLU(), device = args.device)
		elif args.model == 'skipfeatsegcn_h':
			return egcn_h.EGCN(gcn_args, activation = torch.nn.RReLU(), device = args.device, skipfeats=True)
		elif args.model == 'egcn_o':	# LSTM deviceはegcn_i.pyでcpuを選択
			return egcn_o.EGCN(gcn_args, activation = torch.nn.RReLU(), device = args.device)
		else:
			raise NotImplementedError('need to finish modifying the models')

# classifier(models.py)の次元数決定
def build_classifier(args,tasker):
	# node_cls, static_node_clsの時
	if 'node_cls' == args.task or 'static_node_cls' == args.task:
		mult = 1
	else:	# それ以外の時
		mult = 2
	
	# 動的なモデルの時
	if 'gru' in args.model or 'lstm' in args.model:
		in_feats = args.gcn_parameters['lstm_l2_feats'] * mult
	elif args.model == 'skipfeatsgcn' or args.model == 'skipfeatsegcn_h':	# skipfeatsを使う時
		in_feats = (args.gcn_parameters['layer_2_feats'] + args.gcn_parameters['feats_per_node']) * mult
	else:
		in_feats = args.gcn_parameters['layer_2_feats'] * mult  # egcn_h,egin_hはここを通る in_feats = 200
		# print('build_classifier in els', in_feats)
	print('out_features',tasker.num_classes)

	# Classifierはtorch.nn.moduleクラス
	return mls.Classifier(args,in_features = in_feats, out_features = tasker.num_classes).to(args.device)


# 実行
if __name__ == '__main__':
	# yamlのパーサー
	parser = u.create_parser()
	# yamlで設定した項目を取得
	args = u.parse_args(parser)

	# PyTorch上でできる分散アプリケーション(MPI)のセットアップ
	global rank, wsize, use_cuda
	args.use_cuda = (torch.cuda.is_available() and args.use_cuda)
	args.device='cpu'
    # static modelを動かすときになぜかcpu,gpu使っているのでとりあえず切った
	if args.use_cuda:
		args.device='cuda'
	print ("use CUDA:", args.use_cuda, "- device:", args.device)
	try:
		dist.init_process_group(backend='mpi') #, world_size=4
		rank = dist.get_rank()
		wsize = dist.get_world_size()
		print('Hello from process {} (out of {})'.format(dist.get_rank(), dist.get_world_size()))
		if args.use_cuda:
			torch.cuda.set_device(rank )  # are we sure of the rank+1????
			print('using the device {}'.format(torch.cuda.current_device()))
	except:
		rank = 0
		wsize = 1
		print(('MPI backend not preset. Set process rank to {} (out of {})'.format(rank,
																				   wsize)))


	# 乱数シード固定
	if args.seed is None and args.seed!='None':
		seed = 123+rank#int(time.time())+rank
	else:
		seed=args.seed#+rank
	np.random.seed(seed)
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	args.seed=seed
	args.rank=rank
	args.wsize=wsize

	# Assign the requested random hyper parameters
	# リクエストされたランダムハイパーパラメータをアサイン
	args = build_random_hyper_params(args)

	#build the dataset
	# データセットを構築
	dataset = build_dataset(args)
	#build the tasker
	# タスク設定
	tasker = build_tasker(args,dataset)
	#build the splitter
	# スプリッターを設定
	splitter = sp.splitter(args,tasker)
	#build the models
	# モデル作成
	gcn = build_gcn(args, tasker)
	classifier = build_classifier(args,tasker)
	#build a loss
	# 損失関数を作成
	cross_entropy = ce.Cross_Entropy(args,dataset).to(args.device)

	#trainer
	# 学習環境を作成(trainer.py)
	trainer = tr.Trainer(args,
						 splitter = splitter,
						 gcn = gcn,
						 classifier = classifier,
						 comp_loss = cross_entropy,
						 dataset = dataset,
						 num_classes = tasker.num_classes)
	# 実行
	trainer.train()
