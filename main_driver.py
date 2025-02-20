import argparse
import sys
import numpy
import pandas as pd
from torch.utils.data import DataLoader
from read_dataset import data_from_name
from model import *
from model_gae import *
from model_gcn import *
import train_val
from tools import *
import scipy as sp
import os

# ==============================================================================
# Training settings
# ==============================================================================
parser = argparse.ArgumentParser(description='PyTorch Example')
#
parser.add_argument('--model', type=str, default='koopmanAE', metavar='N', help='model')
#
parser.add_argument('--alpha', type=int, default='1', help='model width')
#
parser.add_argument('--dataset', type=str, default='PM2.5', metavar='N', help='dataset')

parser.add_argument('--theta', type=float, default=2.4, metavar='N', help='angular displacement')
#
parser.add_argument('--noise', type=float, default=0.0, metavar='N', help='noise level')
#
parser.add_argument('--lr', type=float, default=1e-3, metavar='N', help='learning rate (default: 0.01)')
#
parser.add_argument('--wd', type=float, default=0.0, metavar='N', help='weight_decay (default: 1e-5)')
#
parser.add_argument('--epochs', type=int, default=31, metavar='N', help='number of epochs to train (default: 10)')
#
parser.add_argument('--batch', type=int, default=64, metavar='N', help='batch size (default: 10000)')
#
parser.add_argument('--batch_test', type=int, default=200, metavar='N',
                    help='batch size  for test set (default: 10000)')
#
parser.add_argument('--plotting', type=bool, default=True, metavar='N', help='number of epochs to train (default: 10)')
#
parser.add_argument('--folder', type=str, default='test', help='specify directory to print results to')
#
parser.add_argument('--lamb', type=float, default='1', help='balance between reconstruction and prediction loss')
#
parser.add_argument('--nu', type=float, default='1e-1', help='tune backward loss')
#
parser.add_argument('--eta', type=float, default='1e-1', help='tune consistent loss')
#
parser.add_argument('--steps', type=int, default='24', help='steps for learning forward dynamics')
#
parser.add_argument('--steps_back', type=int, default='24', help='steps for learning backwards dynamics')
#
parser.add_argument('--bottleneck', type=int, default='32', help='size of bottleneck layer')
#
parser.add_argument('--lr_update', type=int, nargs='+', default=[10,20, 30,40],
                    help='decrease learning rate at these epochs')
#
parser.add_argument('--lr_decay', type=float, default='0.4', help='PCL penalty lambda hyperparameter')
#
parser.add_argument('--gamma', type=float, default='0.95', help='learning rate decay')

parser.add_argument('--backward', type=int, default=1, help='train with backward dynamics')
# parser.add_argument('--backward', type=int, default=0, help='train with backward dynamics')
#
parser.add_argument('--init_scale', type=float, default=0.99, help='init scaling')
#
parser.add_argument('--gradclip', type=float, default=0.05, help='gradient clipping')
#
parser.add_argument('--pred_steps', type=int, default='12', help='prediction steps')
#
parser.add_argument('--seed', type=int, default='999', help='seed value')
#
parser.add_argument('--gpu', type=str, default='-1', help='gpu')

parser.add_argument('--early_stop', type=int, default='20', help='early stop')

parser.add_argument('--use_gae', type=int, default=2, help='0:not use graph, 1:use graph encoder, 2:use Chebnet')

parser.add_argument('--mode_constraint', type=int, default='0', help='0:do not need mode constraint,1:need mode constraint')

parser.add_argument('--pretrain', type=bool, default=False)
args = parser.parse_args()

# torch.cuda.set_per_process_memory_fraction(0.5, 0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
set_seed(args.seed)
device = get_device()
# ******************************************************************************
# Create folder to save results
# ******************************************************************************
use_gae = args.use_gae
proj_dir = os.path.dirname(os.path.abspath(__file__))
result_path = os.path.join(proj_dir, args.folder)
if not os.path.isdir(args.folder):
    os.mkdir(args.folder)

modelpath = result_path
logger = get_logger(os.path.join(modelpath, 'exp.log'))
print(f'model path:{modelpath}')

# ******************************************************************************
# load data
# ******************************************************************************

Xtrain, Xtest, Xval, Xtrain_clean, Xtest_clean, m, n = data_from_name(proj_dir, args.dataset, noise=args.noise,
                                                                      theta=args.theta)
# ******************************************************************************
if use_gae == 0:
    adj = 0
    edge_index = 0
    edge_attr = 0
elif use_gae == 1:
    adj_file = os.path.join(proj_dir, 'data/adj_binary.npy')
    adj = np.load(adj_file)
    adj = torch.tensor(adj)
    adj = preprocess_graph(adj)
    adj = adj.to(device)
    edge_index = 0
    edge_attr = 0
elif use_gae == 2:
    index_file = os.path.join(proj_dir, 'data/edge_index.npy')
    attr_file = os.path.join(proj_dir, 'data/edge_attr.npy')
    edge_index = np.load(index_file)
    edge_attr = np.load(attr_file)
    edge_index = torch.tensor(edge_index)
    edge_attr = torch.tensor(edge_attr)
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)
    adj=0
edge_index = edge_index.to(torch.int64)

# ******************************************************************************
# Reshape data for pytorch into 4D tensor Samples x Channels x Width x Hight
# ******************************************************************************
Xtrain = add_channels(Xtrain)
Xtest = add_channels(Xtest)
Xval = add_channels(Xval)

# transfer to tensor
Xtrain = torch.from_numpy(Xtrain).float().contiguous()
Xtest = torch.from_numpy(Xtest).float().contiguous()
Xval = torch.from_numpy(Xval).float().contiguous()

# ******************************************************************************
# Create Dataloader objects
# ******************************************************************************
trainDat = []
valDat = []
testDat = []
start = 0
for i in np.arange(args.steps, -1, -1):
    if i == 0:
        trainDat.append(Xtrain[start:].float())
        valDat.append(Xval[start:].float())
        testDat.append(Xtest[start:].float())
    else:
        trainDat.append(Xtrain[start:-i].float())
        valDat.append(Xval[start:-i].float())
        testDat.append(Xtest[start:-i].float())
    start += 1

train_data = torch.utils.data.TensorDataset(*trainDat)
val_data = torch.utils.data.TensorDataset(*valDat)
test_data = torch.utils.data.TensorDataset(*testDat)
del (trainDat)
del (valDat)
del (testDat)

train_loader = DataLoader(dataset=train_data,
                          batch_size=args.batch,
                          shuffle=True)
val_loader = DataLoader(dataset=val_data,
                        batch_size=args.batch,
                        shuffle=False)
test_loader = DataLoader(dataset=test_data,
                        batch_size=args.batch,
                        shuffle=False)
# ==============================================================================
# Model
# ==============================================================================
print(Xtrain.shape)

if use_gae == 0:  #model
    model = koopmanAE(m, n, args.bottleneck, args.steps, args.steps_back, args.alpha, args.init_scale)
elif use_gae == 1:   # model_gae
    model = GraphKoopmanAE(m, n, args.bottleneck, args.steps, args.steps_back, args.alpha, args.init_scale)
elif use_gae == 2:   # model_gcn
    model = GraphKoopmanGCN(m, n, args.bottleneck, args.steps, args.steps_back, args.alpha, args.init_scale)

# ==============================================================================
# Model summary
# ==============================================================================
logger.info('---------------------- Setup ---------------------')
logger.info('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
logger.info('Total params: %.2fk' % (sum(p.numel() for p in model.parameters()) / 1000.0))
logger.info('--------------------------------------------------')
logger.info(model)

# ==============================================================================
# Start training
# ==============================================================================
model = train_val.train(model, modelpath, train_loader, val_loader, test_loader, logger,
                        lr=args.lr, gamma = args.gamma, weight_decay=args.wd, lamb=args.lamb, num_epochs=args.epochs,
                        learning_rate_change=args.lr_decay, epoch_update=args.lr_update, early_stop=args.early_stop,
                        use_gae=use_gae, adj=adj, edge_index=edge_index, edge_attr=edge_attr,
                        nu=args.nu, eta=args.eta, backward=args.backward, steps=args.steps, steps_back=args.steps_back,
                        gradclip=args.gradclip, mode_cons=args.mode_constraint)

# ******************************************************************************
# Prediction
# ******************************************************************************

Xinput, Xtarget = torch.cat((Xval[-1:],Xtest[:-1]),dim =0),Xtest
model = model.to(device)
snapshots_pred = [[] for i in range(0,args.pred_steps)]
snapshots_truth = [[] for i in range(0,args.pred_steps)]
length = 5183-args.pred_steps+1
error = [0 for i in range(0,args.pred_steps)]
rmse = [0 for i in range(0,args.pred_steps)]
IA  = [0 for i in range(0,args.pred_steps)]

for i in range(0,length):
    error_temp = []
    rmse_error = []
    init = Xinput[i].float().to(device)
    if i == 0:
        init0 = init
    if use_gae==1:
        g = model.GCNencoder(init, adj)
        g = g.to(torch.float32)
        z = model.encoder(g)
    elif use_gae==0:
        z = model.encoder(init)   #embedd data in latent space
    elif use_gae == 2:
        z = model.encoder(init,edge_index,edge_attr)

    for j in range(args.pred_steps):
        if isinstance(z, tuple):
            z = model.dynamics(*z)
        else:
            z = model.dynamics(z)
        if isinstance(z, tuple):
            if use_gae==1:
                x_pred = model.decoder(z[0],adj)
            else:
                x_pred = model.decoder(z[0])
        else:
            if use_gae==1:
                x_pred = model.decoder(z,adj)
            else:
                x_pred = model.decoder(z) #
        target_temp = Xtarget[i+j].data.cpu().numpy().reshape(m,n)  #[64,1]

        error[j] = error[j]+ np.sum(np.abs(x_pred.data.cpu().numpy().reshape(m, n) - target_temp))
        rmse[j] = rmse[j] + np.sum(np.power(x_pred.data.cpu().numpy().reshape(m, n) - target_temp,2))
        IA[j] = IA[j] + np.sum(np.power(np.abs(x_pred.data.cpu().numpy().reshape(m, n)-np.mean(target_temp)) + np.abs(target_temp - np.mean(target_temp)),2))

        snapshots_pred[j].extend(x_pred.data.cpu().numpy().reshape(m))
        snapshots_truth[j].extend(target_temp.reshape(m))

for i in range(args.pred_steps):
    pearson = np.corrcoef(snapshots_pred[i],snapshots_truth[i])[0,1]
    logger.info(f'{i}hour: {pearson}')

error = np.asarray(error)
rmse1 = np.asarray(rmse)
mae = error/(m*length)
rmse = np.sqrt(rmse1/(m*length))
IA1 = 1-rmse1/IA
print(f'IA1:{IA1}')
print(f'mae:{mae}')
print(f'rmse:{rmse}')

aaa = pd.DataFrame(data=snapshots_pred).T
bbb = pd.DataFrame(data=snapshots_truth).T
numpy.savetxt("Predict-12.csv", aaa, delimiter=",")
numpy.savetxt("truth-24.csv", bbb, delimiter=",")

# ==============================================================================

# =======================================
model.eval()
A =  model.dynamics.dynamics.weight.cpu().data.numpy()
w, v = np.linalg.eig(A)
