import utils as u
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import math

import dgl
# from dgl.nn import GINConv
from torch.nn.functional import relu
import torch.nn.functional as F

import dgl.function as fn
from dgl.utils import expand_as_pair

# LSTMバージョン

class EGCN(torch.nn.Module):
    def __init__(self, args, activation, device='cpu', skipfeats=False):
        super().__init__()
        GRCU_args = u.Namespace({})

        # 5層(インプットを含める)
        # feats = [args.feats_per_node,
        #          args.layer_1_feats,
        #          args.layer_2_feats,
        #          args.layer_2_feats,
        #          args.layer_2_feats]

        # 2層(単純に)
        feats = [args.feats_per_node,
                 args.layer_1_feats,
                 args.layer_2_feats]
        
        num_conv_layer = 2
        num_linear_last = num_conv_layer + 1

        self.device = device
        self.skipfeats = skipfeats
        self.GRCU_layers = []
        self._parameters = nn.ParameterList()
        self.activation = activation
        
        # ハードコーディング データセット変える時注意
        self.linear_0 = nn.Linear(162, args.layer_1_feats).to('cuda')
        self.linear_last = nn.Linear(args.layer_2_feats * num_linear_last, args.layer_2_feats).to('cuda')

        # ハイパーパラメータ化(0, 1e-5, 1e-4, 1e-3, 1e-2)
        self.weight_param = 0


        for i in range(1,len(feats)):
            GRCU_args = u.Namespace({'in_feats' : feats[i-1],
                                     'out_feats': feats[i],
                                     'activation': activation})

            grcu_i = GRCU(GRCU_args)
            #print (i,'grcu_i', grcu_i)
            self.GRCU_layers.append(grcu_i.to(self.device))
            self._parameters.extend(list(self.GRCU_layers[-1].parameters()))

    def parameters(self):
        return self._parameters

    def forward(self,A_list, Nodes_list,nodes_mask_list):
        node_feats= Nodes_list[-1]

        # 注意 linnerのサイズはマニュアルで設定、のちのち自動設定したい

        # グラフ埋め込みを格納するリスト
        graph_emb_l0_list = []
        graph_emb_list = []

        # L=0のグラフ埋め込み
        for node_embs in Nodes_list:
            feat = node_embs.to_dense()
            graph_emb = torch.sum(feat,0)
            out_graph_emb = self.activation(self.linear_0(graph_emb))
            graph_emb_l0_list.append(out_graph_emb)
        
        # l=0をlistにappend
        graph_emb_list.append(torch.stack(graph_emb_l0_list[:],0))

        
        # EvolveGIN_Oのレイヤーごとにノード、グラフ埋め込みを取得
        for unit in self.GRCU_layers:
            Nodes_list, graph_emb_seq = unit(A_list,Nodes_list)#,nodes_mask_list)
            # グラフ埋め込みをtensorで格納
            graph_emb_list.append(torch.stack(graph_emb_seq[:],0))
        


        # # num_layer * t * 100のグラフ埋め込みのtensorを作成
        # graph_emb_tensors = torch.stack(graph_emb_list[:],0)

        # # 書くグラフ埋め込みの総和を取り、t * 100のtensorを作成 
        # out_graph_embs_tensor = torch.sum(graph_emb_tensors,0)

        # # 正規化
        # out_graph_embs_tensor = out_graph_embs_tensor / torch.norm(out_graph_embs_tensor)



        # 各グラフ埋め込みをconcatし、t * (300 = size_layer * 100 )へ
        out_graph_embs_tensor = torch.cat((graph_emb_list[:]),1)

        # linear層通過
        out_graph_embs_tensor = self.activation(self.linear_last(out_graph_embs_tensor))
        

        # 最新の埋め込みを習得
        out_node_emb = Nodes_list[-1]
        out_graph_emb = out_graph_embs_tensor[-1]

        # 出力する埋め込みを作成        
        out = out_node_emb + self.weight_param * out_graph_emb

        if self.skipfeats:
            out = torch.cat((out,node_feats), dim=1)   # use node_feats.to_dense() if 2hot encoded input 
        return out


class GRCU(torch.nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        cell_args = u.Namespace({})
        cell_args.rows = args.in_feats
        cell_args.cols = args.out_feats

        # W1 LSTM
        self.evolve_weight1 = mat_GRU_cell(cell_args)

        # 2層目のinputの隠れ層似合わせるためにrowsを変更
        cell_args.rows = args.out_feats

        # W2 LSTM
        self.evolve_weight2 = mat_GRU_cell(cell_args)
        # # W3 LSTM
        self.evolve_weight3 = mat_GRU_cell(cell_args)

        self.activation = self.args.activation

        # # linear層
        # self.linear = nn.Linear(self.args.out_feats, self.args.out_feats)
        

        # 1層目
        self.GIN_init_W1 = Parameter(torch.Tensor(self.args.in_feats,self.args.out_feats))
        self.W1_init_bias = Parameter(torch.Tensor(self.args.out_feats))
        self.reset_param(self.GIN_init_W1)
        self.reset_bias(self.W1_init_bias)

        # 2層目
        self.GIN_init_W2 = Parameter(torch.Tensor(self.args.out_feats,self.args.out_feats))
        self.W2_init_bias = Parameter(torch.Tensor(self.args.out_feats))
        self.reset_param(self.GIN_init_W2)
        self.reset_bias(self.W2_init_bias)

        # # 3層目
        self.GIN_init_W3 = Parameter(torch.Tensor(self.args.out_feats,self.args.out_feats))
        self.W3_init_bias = Parameter(torch.Tensor(self.args.out_feats))
        self.reset_param(self.GIN_init_W3)
        self.reset_bias(self.W3_init_bias)

    def reset_param(self,t):
        #Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv,stdv)
    
    def reset_bias(self,t):
        stdv = 1. / math.sqrt(t.size(0))
        t.data.uniform_(-stdv,stdv)
    

    def forward(self,A_list,node_embs_list):#,mask_list):
        GIN_W1 = self.GIN_init_W1
        W1_bias = self.W1_init_bias

        GIN_W2 = self.GIN_init_W2
        W2_bias = self.W2_init_bias

        GIN_W3 = self.GIN_init_W3
        W3_bias = self.W3_init_bias
        # print(mask_list)

        # ノード埋め込み格納用リスト
        out_seq = []

        # グラフ埋め込み格納用リスト
        out_g_emb_seq = []

        for t,Ahat in enumerate(A_list):
            # print('t is ',t)
            # print('hidden_state size is',hidden_state.size())
            node_embs = node_embs_list[t]
       
            # グラフのノードリストをAhatから作成
            graph_node_list = Ahat._indices()
            u, v = graph_node_list[0], graph_node_list[1]
            
            # dgl.graphでグラフ作成
            g = dgl.graph((u,v),num_nodes=Ahat.size(0))
            
            is_sparse_coo = str(node_embs.layout)

            if is_sparse_coo == 'torch.sparse_coo':
                feat = node_embs.to_dense()
            else:
                feat = node_embs

            # aggregate層設定
            conv = GINConv('sum').to('cuda') # learn_eps = True)
            # aggregation
            node_embs = conv(g, feat)

            # 1層目
            GIN_W1 = self.evolve_weight1(GIN_W1)# ,node_embs,mask_list[t])            
            #first_node_embs = self.activation(F.linear(node_embs,GIN_W1.t(),W1_bias))           
            first_node_embs = self.activation(node_embs.matmul(GIN_W1)) # + W1_bias

            # 2層目
            GIN_W2 = self.evolve_weight2(GIN_W2)# ,first_node_embs,mask_list[t])
            # second_node_embs = self.activation(F.linear(first_node_embs,GIN_W2.t(),W2_bias))
            second_node_embs = self.activation(first_node_embs.matmul(GIN_W2)) # + W2_bias
            

            # 3層目
            GIN_W3 = self.evolve_weight3(GIN_W3)# ,second_node_embs,mask_list[t])
            # # last_node_embs = self.activation(F.linear(second_node_embs,GIN_W3.t(),W3_bias))     
            last_node_embs = self.activation(second_node_embs.matmul(GIN_W3)) # + W3_bias

            
            # グラフ埋め込み作成
            out_graph_emb = torch.sum(last_node_embs,0)

            # # linearを通過
            # out_graph_emb = self.activation(self.linear(graph_emb))
            
            # print('out_graph_emb size is',out_graph_emb.size())
            # print('graph_emb is',graph_emb)

            out_g_emb_seq.append(out_graph_emb)

            out_seq.append(last_node_embs)
    
        return out_seq, out_g_emb_seq



class mat_GRU_cell(torch.nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.update = mat_GRU_gate(args.rows,
                                   args.cols,
                                   torch.nn.Sigmoid())

        self.reset = mat_GRU_gate(args.rows,
                                   args.cols,
                                   torch.nn.Sigmoid())

        self.htilda = mat_GRU_gate(args.rows,
                                   args.cols,
                                   torch.nn.Tanh())
        
        self.choose_topk = TopK(feats = args.rows,
                                k = args.cols)

    def forward(self,prev_Q):#,prev_Z,mask):
        # z_topk = self.choose_topk(prev_Z,mask)
        z_topk = prev_Q

        update = self.update(z_topk,prev_Q)
        reset = self.reset(z_topk,prev_Q)

        h_cap = reset * prev_Q
        h_cap = self.htilda(z_topk, h_cap)

        new_Q = (1 - update) * prev_Q + update * h_cap

        return new_Q


class GINConv(torch.nn.Module):
    def __init__(self,
                 apply_func=None,
                 aggregator_type='sum',
                 init_eps=0,
                 learn_eps=False,
                 activation=None):
        super(GINConv, self).__init__()
        self.apply_func = apply_func
        self._aggregator_type = aggregator_type
        self.activation = activation
        if aggregator_type not in ('sum', 'max', 'mean'):
            raise KeyError(
                'Aggregator type {} not recognized.'.format(aggregator_type))
        # to specify whether eps is trainable or not.
        if learn_eps:
            self.eps = torch.nn.Parameter(torch.FloatTensor([init_eps]))
        else:
            self.register_buffer('eps', torch.FloatTensor([init_eps]))

    def forward(self, graph, feat, edge_weight=None):
        _reducer = getattr(fn, self._aggregator_type)
        with graph.local_scope():
            aggregate_fn = fn.copy_src('h', 'm')
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.number_of_edges()
                graph.edata['_edge_weight'] = edge_weight
                aggregate_fn = fn.u_mul_e('h', '_edge_weight', 'm')

            feat_src, feat_dst = expand_as_pair(feat, graph)
            graph.srcdata['h'] = feat_src
            graph.update_all(aggregate_fn, _reducer('m', 'neigh'))
            # print('in egcn_h.py')
            rst = (1 + self.eps) * feat_dst + graph.dstdata['neigh']

            

            # if self.apply_func is not None:
            #     rst = self.apply_func(rst)
            # # activation
            # if self.activation is not None:
            #     rst = self.activation(rst)
            return rst

class mat_GRU_gate(torch.nn.Module):
    def __init__(self,rows,cols,activation):
        super().__init__()
        self.activation = activation
        #the k here should be in_feats which is actually the rows
        self.W = Parameter(torch.Tensor(rows,rows))
        self.reset_param(self.W)

        self.U = Parameter(torch.Tensor(rows,rows))
        self.reset_param(self.U)

        self.bias = Parameter(torch.zeros(rows,cols))

    def reset_param(self,t):
        #Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv,stdv)

    def forward(self,x,hidden):
        out = self.activation(self.W.matmul(x) + \
                              self.U.matmul(hidden) + \
                              self.bias)

        return out

class TopK(torch.nn.Module):
    def __init__(self,feats,k):
        super().__init__()
        self.scorer = Parameter(torch.Tensor(feats,1))
        self.reset_param(self.scorer)
        
        self.k = k

    def reset_param(self,t):
        #Initialize based on the number of rows
        stdv = 1. / math.sqrt(t.size(0))
        t.data.uniform_(-stdv,stdv)

    def forward(self,node_embs,mask):
        scores = node_embs.matmul(self.scorer) / self.scorer.norm()
        scores = scores + mask

        vals, topk_indices = scores.view(-1).topk(self.k)
        topk_indices = topk_indices[vals > -float("Inf")]

        if topk_indices.size(0) < self.k:
            topk_indices = u.pad_with_last_val(topk_indices,self.k)
            
        tanh = torch.nn.Tanh()

        if isinstance(node_embs, torch.sparse.FloatTensor) or \
           isinstance(node_embs, torch.cuda.sparse.FloatTensor):
            node_embs = node_embs.to_dense()

        out = node_embs[topk_indices] * tanh(scores[topk_indices].view(-1,1))

        #we need to transpose the output
        return out.t()
