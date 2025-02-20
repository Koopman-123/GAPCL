from torch import nn
import torch
from layers import *
import numpy as np
from torch_geometric.nn import ChebConv,GATConv

def gaussian_init_(n_units, std=1):    
    sampler = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([std/n_units]))
    Omega = sampler.sample((n_units, n_units))[..., 0]  
    return Omega

class encoderNet(nn.Module):
    def __init__(self, m, n, b, ALPHA = 1):
        super(encoderNet, self).__init__()
        self.N = m * n
        self.tanh = nn.LeakyReLU()
        self.gcn_out = 8
        self.dim = 128
        self.heads = 4
        self.conv = GATConv(1, self.gcn_out, heads=self.heads)
        self.conv = ChebConv(1,self.gcn_out, K=2)
        self.fc1 = nn.Linear(self.N*self.gcn_out, self.dim*ALPHA)
        self.fc2 = nn.Linear(self.dim*ALPHA, self.dim*ALPHA)
        self.fc3 = nn.Linear(self.dim*ALPHA, b)
        self.bn = nn.BatchNorm1d(self.N,momentum=0.5)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)          

    def forward(self, x, edge_index, edge_attr):
        batchsize = x.size()[0]
        x = x.view(batchsize,self.N,-1)
        x_gcn = x.contiguous()

        x_gcn = x_gcn.view(batchsize*self.N,-1)
        edge_attr = edge_attr.to(torch.float32)
        x_gcn = F.leaky_relu(self.conv(x_gcn,edge_index,edge_attr))
        x_gcn = x_gcn.view(batchsize,1,-1)

        x = self.tanh(self.fc1(x_gcn))
        x = self.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

class decoderNet(nn.Module):
    def __init__(self, m, n, b, ALPHA = 1):
        super(decoderNet, self).__init__()

        self.m = m
        self.n = n
        self.b = b

        self.tanh = nn.LeakyReLU()
        self.gcn_out =8
        self.dim = 128

        self.fc1 = nn.Linear(b, self.dim * ALPHA)
        self.fc2 = nn.Linear(self.dim * ALPHA, self.dim * ALPHA)
        self.fc3 = nn.Linear(self.dim * ALPHA, m * n*self.gcn_out)
        self.fc4 = nn.Linear(m * n * self.gcn_out,m*n)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)          

    def forward(self, x):
        x = x.view(-1, 1, self.b)
        x = self.tanh(self.fc1(x)) 
        x = self.tanh(self.fc2(x)) 
        x = self.tanh(self.fc3(x))
        x = self.tanh(self.fc4(x))
        x = x.view(-1, 1, self.m, self.n)
        return x

class dynamics(nn.Module):
    def __init__(self, b, init_scale):
        super(dynamics, self).__init__()
        self.dynamics = nn.Linear(b, b, bias=False)
        self.dynamics.weight.data = gaussian_init_(b, std=1)           
        U, _, V = torch.svd(self.dynamics.weight.data)
        self.dynamics.weight.data = torch.mm(U, V.t()) * init_scale

    def forward(self, x):
        x = self.dynamics(x)
        return x

class dynamics_back(nn.Module):
    def __init__(self, b, omega):
        super(dynamics_back, self).__init__()
        self.dynamics = nn.Linear(b, b, bias=False)
        self.dynamics.weight.data = torch.pinverse(omega.dynamics.weight.data.t())     

    def forward(self, x):
        x = self.dynamics(x)
        return x

class GCNencoder(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNencoder, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
    def forward(self, x, adj):

        batch_size =x.shape[0]
        y = np.zeros((batch_size,121, 1))
        for i in range(batch_size):
            tmp = x[i].view(-1,1)
            hidden1 = self.gc1(tmp, adj)
            hidden2 = self.gc2(hidden1, adj)
            y[i] = hidden2.cpu().detach().numpy()
        y = torch.tensor(y)
        y=y.to(torch.device('cuda'))
        return y.unsqueeze(1)


class GCNdecoder(nn.Module):
    """Decoder for using inner product for prediction."""
    def __init__(self, dropout, act=torch.sigmoid):
        super(GCNdecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj

class GraphKoopmanGCN(nn.Module):
    def __init__(self, m, n, b, steps, steps_back, alpha = 1, init_scale=1):
        super(GraphKoopmanGCN, self).__init__()
        self.steps = steps
        self.steps_back = steps_back
        self.encoder = encoderNet(m, n, b, ALPHA = alpha)
        self.dynamics = dynamics(b, init_scale)
        self.backdynamics = dynamics_back(b, self.dynamics)
        self.decoder = decoderNet(m, n, b, ALPHA = alpha)
        self.GCNencoder = GCNencoder(1,128,1,0)

    def forward(self, x,edge_index,edge_attr,mode='forward'):
        out = []
        out_back = []

        z = self.encoder(x.contiguous(),edge_index,edge_attr)
        q = z.contiguous()
        if mode == 'forward':
            for _ in range(self.steps):
                q = self.dynamics(q)
                out.append(self.decoder(q))

            out.append(self.decoder(z.contiguous()))
            return out, out_back    

        if mode == 'backward':
            for _ in range(self.steps_back):
                q = self.backdynamics(q)
                out_back.append(self.decoder(q))
                
            out_back.append(self.decoder(z.contiguous()))
            return out, out_back
