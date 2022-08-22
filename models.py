import torch
import utils as u
from argparse import Namespace
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import torch.nn as nn
import math

# dgl
import dgl
from dgl.nn import GINConv
from torch.nn.functional import relu

# GCNクラス
class Sp_GCN(torch.nn.Module):
    # インスタンスを作成
    def __init__(self,args,activation):
        super().__init__()
        # 活性化関数
        self.activation = activation
        # レイヤー数
        self.num_layers = args.num_layers

        # レイヤー
        self.lin1 = nn.Linear(162, args.layer_1_feats)
        self.conv1 = GINConv(self.lin1,'sum')

        self.lin2 = nn.Linear(args.layer_1_feats, args.layer_2_feats) 
        self.conv2 = GINConv(self.lin2, 'sum') 


        # # 重み
        # self.w_list = nn.ParameterList()

        # # レイヤー数分パラメータをリストに追加
        # for i in range(self.num_layers):
        #     if i==0:
        #         # feats_per_node, layer_1_featsはyaml gcn_paramertersで定義
        #         # exampleだとfeats_per_node = 100, layer_1_feats = 100
        #         w_i = Parameter(torch.Tensor(args.feats_per_node, args.layer_1_feats))
        #         u.reset_param(w_i)
        #     else:
        #         # layer_1_feats, layer_2_featsはyaml gcn_paramertersで定義
        #         # exampleだとlayer_1_feats = 100, layer_2_feats = 100
        #         w_i = Parameter(torch.Tensor(args.layer_1_feats, args.layer_2_feats))
        #         u.reset_param(w_i)
        #     self.w_list.append(w_i)

    # 順伝播ステップ GIN
    def forward(self,A_list, Nodes_list, nodes_mask_list):
        
        node_feats = Nodes_list[-1]
        # print('node_feats is',node_feats)
        graph_node_list = A_list[-1]._indices()
        # print('graph_node_list is',graph_node_list)

        u, v = graph_node_list[0], graph_node_list[1]
        u = u.to('cpu')
        v = v.to('cpu')

        # dgl.graphでグラフ作成
        g = dgl.graph((u,v))
        g = g.to('cpu')
        # print(g.device)
        # print(' graph ok')

        is_sparse_coo = str(node_feats.layout)

        if is_sparse_coo == 'torch.sparse_coo':
            # print(' coo')
            feat = node_feats.to_dense()
        else:
            # print(' not coo')
            feat = node_feats
        # print(' dense ok')
        
        feat = feat.to('cpu')
        # print(' feat ok')
        

        last_l = self.conv1(g,feat).to('cpu')
        last_l = self.conv2(g,last_l).to('cpu')
        
        return last_l
    
    ## 元バージョン
    # class Sp_GCN(torch.nn.Module):
    # # インスタンスを作成
    # def __init__(self,args,activation):
    #     super().__init__()
    #     # 活性化関数
    #     self.activation = activation
    #     # レイヤー数
    #     self.num_layers = args.num_layers
    #     # 重み
    #     self.w_list = nn.ParameterList()

    #     # レイヤー数分パラメータをリストに追加
    #     for i in range(self.num_layers):
    #         if i==0:
    #             # feats_per_node, layer_1_featsはyaml gcn_paramertersで定義
    #             # exampleだとfeats_per_node = 100, layer_1_feats = 100
    #             w_i = Parameter(torch.Tensor(args.feats_per_node, args.layer_1_feats))
    #             u.reset_param(w_i)
    #         else:
    #             # layer_1_feats, layer_2_featsはyaml gcn_paramertersで定義
    #             # exampleだとlayer_1_feats = 100, layer_2_feats = 100
    #             w_i = Parameter(torch.Tensor(args.layer_1_feats, args.layer_2_feats))
    #             u.reset_param(w_i)
    #         self.w_list.append(w_i)
    ## 順伝播ステップ
    # def forward(self,A_list, Nodes_list, nodes_mask_list):
        
    #     node_feats = Nodes_list[-1]
    #     #A_list: T, each element sparse tensor
    #     #take only last adj matrix in time
    #     Ahat = A_list[-1]
    #     #Ahat: NxN ~ 30k
    #     #sparse multiplication

    #     # Ahat NxN
    #     # self.node_embs = Nxk
    #     #
    #     # note(bwheatman, tfk): change order of matrix multiply
    #     last_l = self.activation(Ahat.matmul(node_feats.matmul(self.w_list[0])))
    #     for i in range(1, self.num_layers):
    #         last_l = self.activation(Ahat.matmul(last_l.matmul(self.w_list[i])))
    #     return last_l


class Sp_Skip_GCN(Sp_GCN):
    def __init__(self,args,activation):
        super().__init__(args,activation)
        self.W_feat = Parameter(torch.Tensor(args.feats_per_node, args.layer_1_feats))

    def forward(self,A_list, Nodes_list = None):
        node_feats = Nodes_list[-1]
        #A_list: T, each element sparse tensor
        #take only last adj matrix in time
        Ahat = A_list[-1]
        #Ahat: NxN ~ 30k
        #sparse multiplication

        # Ahat NxN
        # self.node_feats = Nxk
        #
        # note(bwheatman, tfk): change order of matrix multiply
        l1 = self.activation(Ahat.matmul(node_feats.matmul(self.W1)))
        l2 = self.activation(Ahat.matmul(l1.matmul(self.W2)) + (node_feats.matmul(self.W3)))

        return l2

class Sp_Skip_NodeFeats_GCN(Sp_GCN):
    def __init__(self,args,activation):
        super().__init__(args,activation)

    def forward(self,A_list, Nodes_list = None):
        node_feats = Nodes_list[-1]
        Ahat = A_list[-1]
        last_l = self.activation(Ahat.matmul(node_feats.matmul(self.w_list[0])))
        for i in range(1, self.num_layers):
            last_l = self.activation(Ahat.matmul(last_l.matmul(self.w_list[i])))
        skip_last_l = torch.cat((last_l,node_feats), dim=1)   # use node_feats.to_dense() if 2hot encoded input
        return skip_last_l

class Sp_GCN_LSTM_A(Sp_GCN):
    def __init__(self,args,activation):
        super().__init__(args,activation)
        self.rnn = nn.LSTM(
                input_size=args.layer_2_feats,
                hidden_size=args.lstm_l2_feats,
                num_layers=args.lstm_l2_layers
                )

    def forward(self,A_list, Nodes_list = None, nodes_mask_list = None):
        last_l_seq=[]
        for t,Ahat in enumerate(A_list):
            node_feats = Nodes_list[t]
            #A_list: T, each element sparse tensor
            #note(bwheatman, tfk): change order of matrix multiply
            last_l = self.activation(Ahat.matmul(node_feats.matmul(self.w_list[0])))
            for i in range(1, self.num_layers):
                last_l = self.activation(Ahat.matmul(last_l.matmul(self.w_list[i])))
            last_l_seq.append(last_l)

        last_l_seq = torch.stack(last_l_seq)

        out, _ = self.rnn(last_l_seq, None)
        return out[-1]


class Sp_GCN_GRU_A(Sp_GCN_LSTM_A):
    def __init__(self,args,activation):
        super().__init__(args,activation)
        self.rnn = nn.GRU(
                input_size=args.layer_2_feats,
                hidden_size=args.lstm_l2_feats,
                num_layers=args.lstm_l2_layers
                )

class Sp_GCN_LSTM_B(Sp_GCN):
    def __init__(self,args,activation):
        super().__init__(args,activation)
        # データの形状などの入力形式を検証する文 assert if, notの時の出力
        assert args.num_layers == 2, 'GCN-LSTM and GCN-GRU requires 2 conv layers.'
        self.rnn_l1 = nn.LSTM(
                input_size=args.layer_1_feats,
                hidden_size=args.lstm_l1_feats,
                num_layers=args.lstm_l1_layers
                )

        self.rnn_l2 = nn.LSTM(
                input_size=args.layer_2_feats,
                hidden_size=args.lstm_l2_feats,
                num_layers=args.lstm_l2_layers
                )
        self.W2 = Parameter(torch.Tensor(args.lstm_l1_feats, args.layer_2_feats))
        u.reset_param(self.W2)

    def forward(self,A_list, Nodes_list = None, nodes_mask_list = None):
        l1_seq=[]
        l2_seq=[]
        for t,Ahat in enumerate(A_list):
            node_feats = Nodes_list[t]
            l1 = self.activation(Ahat.matmul(node_feats.matmul(self.w_list[0])))
            l1_seq.append(l1)

        l1_seq = torch.stack(l1_seq)

        out_l1, _ = self.rnn_l1(l1_seq, None)

        for i in range(len(A_list)):
            Ahat = A_list[i]
            out_t_l1 = out_l1[i]
            #A_list: T, each element sparse tensor
            l2 = self.activation(Ahat.matmul(out_t_l1).matmul(self.w_list[1]))
            l2_seq.append(l2)

        l2_seq = torch.stack(l2_seq)

        out, _ = self.rnn_l2(l2_seq, None)
        return out[-1]


class Sp_GCN_GRU_B(Sp_GCN_LSTM_B):
    def __init__(self,args,activation):
        super().__init__(args,activation)
        self.rnn_l1 = nn.GRU(
                input_size=args.layer_1_feats,
                hidden_size=args.lstm_l1_feats,
                num_layers=args.lstm_l1_layers
               )

        self.rnn_l2 = nn.GRU(
                input_size=args.layer_2_feats,
                hidden_size=args.lstm_l2_feats,
                num_layers=args.lstm_l2_layers
                )

class Classifier(torch.nn.Module):
    def __init__(self,args,out_features=2, in_features = None):
        super(Classifier,self).__init__()
        activation = torch.nn.ReLU()

        if in_features is not None:
            num_feats = in_features
        elif args.experiment_type in ['sp_lstm_A_trainer', 'sp_lstm_B_trainer',
                                    'sp_weighted_lstm_A', 'sp_weighted_lstm_B'] :
            num_feats = args.gcn_parameters['lstm_l2_feats'] * 2
        else:
            num_feats = args.gcn_parameters['layer_2_feats'] * 2
        print ('CLS num_feats',num_feats)

        self.mlp = torch.nn.Sequential(torch.nn.Linear(in_features = num_feats,
                                                       out_features =args.gcn_parameters['cls_feats']),
                                       activation,
                                       torch.nn.Linear(in_features = args.gcn_parameters['cls_feats'],
                                                       out_features = out_features))

    def forward(self,x):
        return self.mlp(x)
