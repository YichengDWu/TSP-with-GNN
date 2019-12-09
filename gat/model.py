from dgl import function as fn
import torch as th
from torch import nn
from dgl.nn.pytorch.softmax import edge_softmax

class GATLayer(nn.Module):
    def __init__(self, 
                 in_feats,
                 out_feats,
                 residual=True):
        super(GATLayer, self).__init__()
        num_heads = int(in_feats/out_feats)
        self._num_heads = num_heads
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.Q = nn.Linear(in_feats, out_feats*num_heads, bias=False)
        self.K = nn.Linear(in_feats, out_feats*num_heads, bias=False)
        self.V = nn.Linear(in_feats, out_feats*num_heads, bias=False)
        
        self.scale =  nn.Linear(out_feats*num_heads, 
                                in_feats*num_heads,
                                bias=False)
        
        nn.Parameter(
            th.FloatTensor(size = (out_feats, 1, in_feats))
        )
        if residual:
            self.res_fc = nn.Identity()
        
        self.reset_parameters()

    def reset_parameters(self):
        for param in self.parameters():
            stdv = param.size(-1)**(-0.5)
            param.data.uniform_(-stdv, stdv)

    def forward(self, g, feature):
        g = g.local_var()
        g.ndata['v'] = self.V(feature).view(-1, self._num_heads, self._out_feats)
        g.ndata['q'] = self.Q(feature).view(-1, self._num_heads, self._out_feats)
        g.ndata['k'] = self.K(feature).view(-1, self._num_heads, self._out_feats)

        g.apply_edges(fn.u_mul_v('q', 'k', 'u'))
        #e*h*1
        u = g.edata['u'].sum(-1,keepdim = True)*(self._out_feats)**(-0.5)
        a = edge_softmax(g, u)
        g.edata['a'] = a
        g.update_all(fn.u_mul_e('v','a','m'), fn.sum('m', 'ft'))
        #n*(h*in_feats)
        rst = self.scale(g.ndata['ft'].view(-1, self._out_feats * self._num_heads))
        if self.res_fc is not None:
            rst = rst.view(-1,self._num_heads, self._in_feats).sum(dim = 1) + feature
        return rst

class FFLayer(nn.Module):
    def __init__(self, 
                 in_feats,
                 out_feats,
                 residual=True):
        super(FFLayer, self).__init__()
        self.residual = residual
        self.FF = nn.Sequential(
            nn.Linear(in_feats,out_feats),
            nn.ReLU(),
            nn.Linear(out_feats,in_feats)
        )
    
    def forward(self, x):
        out = self.FF(x)
        if self.residual:
            out = out + x
        return out

class Normalization(nn.Module):
    def __init__(self, embed_dim):
        super(Normalization, self).__init__()

        self.normalizer = nn.BatchNorm1d(embed_dim, affine=True)

        # Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
        # self.init_parameters()

    def init_parameters(self):

        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self,input):
        return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())


class AttentionLayer(nn.Module):
    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden=512):
        super(AttentionLayer, self).__init__()
        self.mht = GATLayer(embed_dim,int(embed_dim/n_heads))
        self.norm1 = Normalization(embed_dim)
        self.ff = FFLayer(embed_dim,feed_forward_hidden)
        self.norm2 = Normalization(embed_dim)
    
    def forward(self, g, feat):
        feat = self.mht(g,feat)
        feat = self.norm1(feat)
        feat = self.ff(feat)
        feat = self.norm2(feat)
        return feat

class GraphAttentionEncoder(nn.Module):
    def __init__(
            self,
            n_heads,
            embed_dim,
            n_layers,
            node_dim=2,
            feed_forward_hidden=512
    ):
        super(GraphAttentionEncoder, self).__init__()

        # To map input to embedding space
        self.init_embed = nn.Linear(node_dim, embed_dim) 

        self.layers = nn.ModuleList([
            AttentionLayer(n_heads, embed_dim, feed_forward_hidden)
            for _ in range(n_layers)
            ])

    def forward(self, g, feat):
        feat = self.init_embed(feat)
        for layer in self.layers:
            feat = layer(g,feat)
        return feat
