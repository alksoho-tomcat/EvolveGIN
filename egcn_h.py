import utils as u
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import math

import dgl
from dgl.nn import GINConv
from torch.nn.functional import relu
# GRUバージョン

class EGCN(torch.nn.Module):
    def __init__(self, args, activation, device='cpu', skipfeats=False):
        super().__init__()
        GRCU_args = u.Namespace({})

        feats = [args.feats_per_node,   # in yaml, 100 min: 50, max: 256
                 args.layer_1_feats,    # in yaml, 100 min: 10, max: 200
                 args.layer_2_feats]    # in yaml, 100
        self.device = device
        self.skipfeats = skipfeats
        self.GRCU_layers = []
        self._parameters = nn.ParameterList()
        for i in range(1,len(feats)):   # exampleだとi = 1, 2の2層
            GRCU_args = u.Namespace({'in_feats' : feats[i-1],
                                     'out_feats': feats[i],
                                     'activation': activation})

            grcu_i = GRCU_GIN(GRCU_args)
            # print (i,'grcu_i', grcu_i)
            # # 出力例
            # # 1 grcu_i GRCU(
            # #   (evolve_weights): mat_GRU_cell(
            # #     (update): mat_GRU_gate(
            # #       (activation): Sigmoid()
            # #     )
            # #     (reset): mat_GRU_gate(
            # #       (activation): Sigmoid()
            # #     )
            # #     (htilda): mat_GRU_gate(
            # #       (activation): Tanh()
            # #     )
            # #     (choose_topk): TopK()
            # #   )
            # #   (activation): RReLU(lower=0.125, upper=0.3333333333333333)
            # # )
            # # 2 つづく

            self.GRCU_layers.append(grcu_i.to(self.device))
            self._parameters.extend(list(self.GRCU_layers[-1].parameters()))

    def parameters(self):
        return self._parameters

    def forward(self,A_list, Nodes_list,nodes_mask_list):
        node_feats= Nodes_list[-1]

        for unit in self.GRCU_layers:
            Nodes_list = unit(A_list,Nodes_list,nodes_mask_list)

        out = Nodes_list[-1]
        if self.skipfeats:
            out = torch.cat((out,node_feats), dim=1)   # use node_feats.to_dense() if 2hot encoded input 
        return out

# GIN用GRCU        
class GRCU_GIN(torch.nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        cell_args = u.Namespace({})
        cell_args.rows = args.in_feats
        cell_args.cols = args.out_feats

        self.evolve_weights = mat_GRU_cell(cell_args)

        self.activation = self.args.activation
        self.GCN_init_weights = Parameter(torch.Tensor(self.args.in_feats,self.args.out_feats))
        self.reset_param(self.GCN_init_weights)

    def reset_param(self,t):
        #Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv,stdv)
    
    # GIN
    def forward(self,A_list,node_embs_list,mask_list):
        GCN_weights = self.GCN_init_weights
        out_seq = []
        for t,Ahat in enumerate(A_list):
            # print('t is ',t)    # default 0~5 yaml num_hist_stepsの値

            # # nodeの数はsbmでは1000個 
            print('Ahat is ',Ahat)  
            # tensor(indices=tensor([[  0,   0,   0,  ..., 999, 999, 999],
            #                        [  0,   2,   3,  ..., 974, 991, 999]]),
            #        values=tensor([0.0088, 0.0086, 0.0086,  ..., 0.0087, 0.0092, 0.0093]),
            #        device='cuda:0', size=(1000, 1000), nnz=106358, layout=torch.sparse_coo)
            
            # print('Ahat[idices] size is', Ahat._indices().size())
            # # Ahat[idices] size is torch.Size([2, 99622])
            # print('Ahat[values] size is', Ahat._values().size())
            # # Ahat[values] size is torch.Size([99622])

            node_embs = node_embs_list[t]

            
            # print('node_embs is',node_embs)
            # # tensor(indices=tensor([[  0,   1,   2,  ..., 997, 998, 999],
            # #                        [113, 104, 118,  ..., 109, 126, 107]]),
            # #        values=tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            # #                      ...                      1., 1., 1., 1., 1., 1.]),
            # # device='cuda:0', size=(1000, 162), nnz=1000, layout=torch.sparse_coo)
            # print('node_embs size is ',node_embs.size())
            # # node_embs size is  torch.Size([1000, 162])            
            # print('node_embs[indices] size is', node_embs._indices().size())
            # # node_embs[indices] size is torch.Size([2, 1000])
            # print('node_embs[values] size is', node_embs._values().size())
            # # node_embs[values] size is torch.Size([1000])
            
            # グラフのノードリストをAhatから作成
            graph_node_list = Ahat._indices()

            # print(graph_node_list[0])
            u, v = graph_node_list[0], graph_node_list[1]
            print(u)
            print(v)
            # dgl.graphでグラフ作成
            g = dgl.graph((u,v))
            print('graph ok')
            
            # feat
            feat = node_embs
            print('feat ok')

            lin = (100,100)
            print('lin ok')
            # ここまでok

            conv = GINConv(lin, 'max')
            print('conv ok')
            
            node_embs = conv(g, feat) 
            # ここでだめになる
            print('node_embs ok')



            #first evolve the weights from the initial and use the new weights with the node_embs
            # mask_list[t]はtop_kで使うので考えなくてよし
            GCN_weights = self.evolve_weights(GCN_weights,node_embs,mask_list[t])
            # GCNの式のまま /sigma(Ahat, H, W)
            node_embs = self.activation(Ahat.matmul(node_embs.matmul(GCN_weights)))

            # print('node_embs is',node_embs)
            # # tensor([[ 2.3553e-03,  4.7566e-03,  3.8439e-03,  ...,  2.7891e-03,
            # #          -7.1719e-04, -4.6245e-05],
            # #         [ 1.3924e-03,  4.9633e-03,  3.9742e-03,  ...,  2.9675e-03,
            # #          -4.8462e-04, -1.7144e-05],
            # #         [ 2.7072e-03,  5.9885e-03,  4.6321e-03,  ...,  2.5399e-03,
            # #          -5.2102e-04, -1.3480e-05],
            # #         ...,
            # #         [ 2.1789e-03,  4.4128e-03,  4.0893e-03,  ...,  2.9770e-03,
            # #          -5.8977e-04,  1.7529e-03],
            # #         [ 2.9603e-03,  5.6537e-03,  5.3050e-03,  ...,  2.7975e-03,
            # #          -6.1702e-04, -2.4860e-04],
            # #         [ 2.1423e-03,  4.9230e-03,  5.4246e-03,  ...,  2.7250e-03,
            # #          -3.6473e-04, -6.3530e-05]], device='cuda:0',
            # #        grad_fn=<RreluWithNoiseBackward0>)
            # print('node_embs size is ',node_embs.size())
            # # torch.Size([1000, 100])

            out_seq.append(node_embs)

        return out_seq


# GCN用GRCU 
class GRCU(torch.nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        cell_args = u.Namespace({})
        cell_args.rows = args.in_feats
        cell_args.cols = args.out_feats

        self.evolve_weights = mat_GRU_cell(cell_args)

        self.activation = self.args.activation
        self.GCN_init_weights = Parameter(torch.Tensor(self.args.in_feats,self.args.out_feats))
        self.reset_param(self.GCN_init_weights)

    def reset_param(self,t):
        #Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv,stdv)
    
    # GCNか? ほぼ確定
    def forward(self,A_list,node_embs_list,mask_list):
        GCN_weights = self.GCN_init_weights
        out_seq = []
        for t,Ahat in enumerate(A_list):
            # print('t is ',t)    # default 0~5 yaml num_hist_stepsの値

            # # nodeの数はsbmでは1000個 
            print('Ahat is ',Ahat)  
            # tensor(indices=tensor([[  0,   0,   0,  ..., 999, 999, 999],
            #                        [  0,   2,   3,  ..., 974, 991, 999]]),
            #        values=tensor([0.0088, 0.0086, 0.0086,  ..., 0.0087, 0.0092, 0.0093]),
            #        device='cuda:0', size=(1000, 1000), nnz=106358, layout=torch.sparse_coo)
            
            # print('Ahat[idices] size is', Ahat._indices().size())
            # # Ahat[idices] size is torch.Size([2, 99622])
            # print('Ahat[values] size is', Ahat._values().size())
            # # Ahat[values] size is torch.Size([99622])

            node_embs = node_embs_list[t]

            
            # print('node_embs is',node_embs)
            # # tensor(indices=tensor([[  0,   1,   2,  ..., 997, 998, 999],
            # #                        [113, 104, 118,  ..., 109, 126, 107]]),
            # #        values=tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            # #                      ...                      1., 1., 1., 1., 1., 1.]),
            # # device='cuda:0', size=(1000, 162), nnz=1000, layout=torch.sparse_coo)
            # print('node_embs size is ',node_embs.size())
            # # node_embs size is  torch.Size([1000, 162])            
            # print('node_embs[indices] size is', node_embs._indices().size())
            # # node_embs[indices] size is torch.Size([2, 1000])
            # print('node_embs[values] size is', node_embs._values().size())
            # # node_embs[values] size is torch.Size([1000])


            #first evolve the weights from the initial and use the new weights with the node_embs
            # mask_list[t]はtop_kで使うので考えなくてよし
            GCN_weights = self.evolve_weights(GCN_weights,node_embs,mask_list[t])
            # GCNの式のまま /sigma(Ahat, H, W)
            node_embs = self.activation(Ahat.matmul(node_embs.matmul(GCN_weights)))

            # print('node_embs is',node_embs)
            # # tensor([[ 2.3553e-03,  4.7566e-03,  3.8439e-03,  ...,  2.7891e-03,
            # #          -7.1719e-04, -4.6245e-05],
            # #         [ 1.3924e-03,  4.9633e-03,  3.9742e-03,  ...,  2.9675e-03,
            # #          -4.8462e-04, -1.7144e-05],
            # #         [ 2.7072e-03,  5.9885e-03,  4.6321e-03,  ...,  2.5399e-03,
            # #          -5.2102e-04, -1.3480e-05],
            # #         ...,
            # #         [ 2.1789e-03,  4.4128e-03,  4.0893e-03,  ...,  2.9770e-03,
            # #          -5.8977e-04,  1.7529e-03],
            # #         [ 2.9603e-03,  5.6537e-03,  5.3050e-03,  ...,  2.7975e-03,
            # #          -6.1702e-04, -2.4860e-04],
            # #         [ 2.1423e-03,  4.9230e-03,  5.4246e-03,  ...,  2.7250e-03,
            # #          -3.6473e-04, -6.3530e-05]], device='cuda:0',
            # #        grad_fn=<RreluWithNoiseBackward0>)
            # print('node_embs size is ',node_embs.size())
            # # torch.Size([1000, 100])

            out_seq.append(node_embs)

        return out_seq

# GRUの定義
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
    # GRUの順伝播(式そのまま)
    def forward(self,prev_Q,prev_Z,mask):
        z_topk = self.choose_topk(prev_Z,mask)

        update = self.update(z_topk,prev_Q)
        reset = self.reset(z_topk,prev_Q)

        h_cap = reset * prev_Q
        h_cap = self.htilda(z_topk, h_cap)

        new_Q = (1 - update) * prev_Q + update * h_cap

        return new_Q

        

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
