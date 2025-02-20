import torch
# import torchvision
from torch import nn
from torch import autograd
from torch import optim
# from torchvision import transforms, datasets
from torch.autograd import grad
from timeit import default_timer as timer
import datetime
import matplotlib.pyplot as plt
import os
import math
import cmath
import numpy as np
import logging
import scipy.sparse as sp
import torch.nn.init as init

def get_box_plt(label,pre,m):
    r_box=[]
    mae_box=[]
    rmse_box = []
    IA_box = []
    for i in range(m):
        r_box.append(np.corrcoef(label[:,i],pre[:,i])[0, 1])
        mae_box.append(np.sum(np.abs(label[:,i]-pre[:,i]))/2736)
        rmse_box.append(np.sqrt(np.sum(np.power(label[:,i]-pre[:,i],2))/2736))
        up =np.sum(np.power(label[:,i]-pre[:,i],2))
        mean = np.mean(label[:,i])
        bottom = np.sum(np.power(np.abs(label[:,i]-mean)+np.abs(pre[:,i]-mean),2))
        IA_box.append(1-up/bottom)
    np.savetxt('/home/zju/zhy/koopmanAE-master/metrics/gcn_box.csv',np.stack([np.array(mae_box),np.array(rmse_box),np.array(r_box),np.array(IA_box)]))
    return r_box

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    # return sparse_to_tuple(adj_normalized)
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false

def process_adj(adj):
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    adj = adj_train

    # Some preprocessing
    adj_norm = preprocess_graph(adj)
    adj_label = adj_train + sp.eye(adj_train.shape[0])
    # adj_label = sparse_to_tuple(adj_label)
    adj_label = torch.FloatTensor(adj_label.toarray())

    pos_weight = torch.Tensor(np.asarray(adj.shape[0] * adj.shape[0] - adj.sum() / adj.sum()))
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    return adj_norm


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    # fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    # sh.setFormatter(formatter)
    logger.addHandler(sh)

    logger.info("Start print log")
    return logger

def log_params(logger,args):
    logger.info('----------------------- args -------------------------')
    arg_dict = args.__dict__
    for key in arg_dict.keys():
        logger.info(f'{key}:{arg_dict[key]}')
    logger.info('------------------------------------------------------')

def create_modelfile(folder):
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    pwd = nowTime
    isExists = os.path.exists(os.path.join(folder,pwd))
    if not isExists:
        os.makedirs(os.path.join(folder,pwd))
    return os.path.join(folder,pwd)

def cal_mode_loss(freq1,freq2):
    err = 1/freq2-1/freq1[:int(freq1.shape[-1]/2)]
    # err = torch.where(torch.isinf(err), torch.full_like(err, 99), err)
    err = torch.pow(err,2)
    return torch.sum(err)


def get_freq(w):
    freq = np.zeros((len(w), 2))
    for i,val in enumerate(w):
        val =cmath.log(val)
        Freal = val.imag/ (2 * math.pi)
        if Freal==0:
            freq[i]=[9999,9999]
        else:
            freq[i]=[1/Freal,val.real]
    return freq

# https://m.edu.iask.sina.com.cn/bdjx/6dTzenEdzHt.html
def log_complex(x,y):
    r = torch.sqrt(x*x+y*y)
    theta = torch.atan(y/x)
    return torch.log(r),theta

def get_freq_torch(eig):
    return (torch.log(eig).imag)/(2 * math.pi)
    # return 1/((torch.log(eig).imag) / (2 * math.pi))
    # freq =  torch.zeros((len(eig), 1), requires_grad=True)
    #     for i,val in enumerate(eig):
    #     val = torch.log(val)
    #     # val =log_complex(val[0],val[1])
    #     Freal = val.imag/ (2 * math.pi)
    #     # if Freal==0:
    #     #     freq[i]=[9999,9999]
    #     # else:
    #     #     freq[i]=[1/Freal,val[0]]
    #     if Freal==0:
    #         freq[i]=Freal
    #     else:
    #         freq[i]=1/Freal
    # return freq

def plt_Xdata(Xinput,loadpath):
    Xdata = Xinput.cpu().detach().numpy()
    Xdata = np.mean(Xdata, axis=2)
    Xdata = np.squeeze(Xdata,1)
    Xdata = np.squeeze(Xdata, 1)
    plt.plot(range(len(Xdata)),Xdata)
    plt.savefig(loadpath +'/000Xinput' +'.png')
    plt.close()

def set_seed(seed=0):
    """Set one seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_device():
    """Get a gpu if available."""
    if torch.cuda.device_count()>0:
        device = torch.device('cuda')
        print("Connected to a GPU")
    else:
        print("Using the CPU")
        device = torch.device('cpu')
    return device


def add_channels(X):
    if len(X.shape) == 2:
        return X.reshape(X.shape[0], 1, X.shape[1],1)

    elif len(X.shape) == 3:
        return X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])

    else:
        return "dimenional error"


def weights_init(m):
    if isinstance(m, nn.Linear):
        if m.weight is not None:
            init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0.0)




