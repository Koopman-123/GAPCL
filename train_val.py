from torch import nn
from tools import *
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from torch.optim import lr_scheduler
mpl.use('Agg')

def train_batch(train_loader, model, optimizer, device, criterion, steps, steps_back, backward, lamb, nu, eta,
                gradclip,use_gae,adj,edge_index,edge_attr,mode_cons):
    train_loss = 0
    count=0
    for batch_idx, data_list in enumerate(train_loader):
        count+=1
        model.train()

        if use_gae==1:
            out, out_back = model(data_list[0].to(device),adj, mode='forward')
        elif use_gae==0:
            out, out_back = model(data_list[0].to(device), mode='forward')
        elif use_gae==2:
            out, out_back = model(data_list[0].to(device),edge_index.to(torch.int64),edge_attr, mode='forward')
        for k in range(steps):
            if k == 0:
                loss_fwd = criterion(out[k], data_list[k + 1].to(device))
            else:
                loss_fwd += criterion(out[k], data_list[k + 1].to(device))

        loss_identity = criterion(out[-1], data_list[0].to(device)) * steps

        loss_bwd = 0.0
        loss_consist = 0.0

        if backward == 1:
            if use_gae == 1:
                out, out_back = model(data_list[-1].to(device),adj, mode='backward')
            elif use_gae == 0:
                out, out_back = model(data_list[-1].to(device), mode='backward')
            elif use_gae == 2:
                out, out_back = model(data_list[-1].to(device),edge_index.to(torch.int64),edge_attr, mode='backward')

            for k in range(steps_back):
                if k == 0:
                    loss_bwd = criterion(out_back[k], data_list[::-1][k + 1].to(device))
                else:
                    loss_bwd += criterion(out_back[k], data_list[::-1][k + 1].to(device))

            A = model.dynamics.dynamics.weight
            B = model.backdynamics.dynamics.weight

            K = A.shape[-1]

            for k in range(1, K + 1):
                As1 = A[:, :k]    #As1=16
                Bs1 = B[:k, :]    #Bs1=16
                As2 = A[:k, :]    #As2=16
                Bs2 = B[:, :k]    #Bs2=16

                Ik = torch.eye(k).float().to(device)

                if k == 1:
                    loss_consist = (torch.sum((torch.mm(Bs1, As1) - Ik) ** 2) + \
                                    torch.sum((torch.mm(As2, Bs2) - Ik) ** 2)) / (2.0 * k)
                else:
                    loss_consist += (torch.sum((torch.mm(Bs1, As1) - Ik) ** 2) + \
                                     torch.sum((torch.mm(As2, Bs2) - Ik) ** 2)) / (2.0 * k)
            if mode_cons:
                I = torch.eye(K,K).to(device)
                A_I= model(I, mode='Linear')
                eigenvalue=torch.linalg.eigvals(A)
                freq = get_freq_torch(eigenvalue)
                freq =torch.sort(freq,descending =True).values
                freq_real = torch.from_numpy(np.array([1 / 2.5, 1 / 5.5,1 / 7.5, 1 / 12, 1 / 24]))
                loss_mode = cal_mode_loss(freq,freq_real).to(device)

                mo = torch.from_numpy(np.ones(K))
                loss_mode2= torch.sum(mo-torch.abs(eigenvalue))
                loss = loss_fwd + lamb * loss_identity + nu * loss_bwd + eta * loss_consist + 100 * loss_mode+50*loss_mode2
            else:
                loss = loss_fwd + lamb * loss_identity + nu * loss_bwd + eta * loss_consist
            # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradclip)
        optimizer.step()
        train_loss += loss.item()
    train_loss /= batch_idx+1
    return train_loss

def val_batch(val_loader, model, device, criterion, steps, use_gae, adj, edge_index,edge_attr):
    val_loss = 0
    model.eval()
    for batch_idx, data_list in enumerate(val_loader):
        # model.train()
        if use_gae==1:
            out, out_back = model(data_list[0].to(device),adj, mode='forward')
        elif use_gae==0:
            out, out_back = model(data_list[0].to(device), mode='forward')
        elif use_gae==2:
            out, out_back = model(data_list[0].to(device),edge_index.to(torch.int64),edge_attr, mode='forward')

        for k in range(steps):
            if k == 0:
                loss_fwd = criterion(out[k], data_list[k + 1].to(device))
            else:
                loss_fwd += criterion(out[k], data_list[k + 1].to(device))
        val_loss += loss_fwd.item()
    val_loss /= batch_idx+1
    return val_loss

def test_batch(test_loader, model, device, criterion, steps, use_gae, adj, edge_index,edge_attr):
    val_loss = np.zeros(steps)
    total_loss = 0
    model.eval()
    for batch_idx, data_list in enumerate(test_loader):
        model.eval()
        if use_gae==1:
            out, out_back = model(data_list[0].to(device),adj, mode='forward')
        elif use_gae==0:
            out, out_back = model(data_list[0].to(device), mode='forward')
        elif use_gae==2:
            out, out_back = model(data_list[0].to(device),edge_index.to(torch.int64),edge_attr, mode='forward')
        for k in range(steps):
            if k==0:
                loss_fwd = nn.MSELoss().to(device)(out[k], data_list[k + 1].to(device))
            else:
                loss_fwd += nn.MSELoss().to(device)(out[k], data_list[k + 1].to(device))
            val_loss[k] += nn.MSELoss().to(device)(out[k], data_list[k + 1].to(device)).item()
        total_loss+=loss_fwd.item()
    val_loss /= batch_idx + 1
    total_loss /= batch_idx + 1
    return np.sqrt(val_loss),total_loss

def train(model, modelpath, train_loader, val_loader,test_loader,logger, lr,gamma, weight_decay,
          lamb, num_epochs, learning_rate_change, epoch_update,early_stop,use_gae,adj,edge_index,edge_attr,
          nu=0.0, eta=0.0, backward=0, steps=1, steps_back=1, gradclip=1, mode_cons=0):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    print(optimizer)
    device = get_device()
    criterion = nn.MSELoss().to(device)

    best_epoch = 0
    val_loss_min = 1000000
    train_loss_epoch=[]
    val_loss_epoch = []

    for epoch in range(num_epochs):
        train_loss = train_batch(train_loader, model, optimizer, device, criterion, steps, steps_back, backward, lamb, nu, eta,
                gradclip,use_gae,adj,edge_index,edge_attr,mode_cons)
        val_loss = val_batch(val_loader, model,  device, criterion, steps,use_gae,adj,edge_index,edge_attr)
        rmse,test_loss = test_batch(test_loader, model, device, criterion, steps, use_gae, adj, edge_index, edge_attr)
        train_loss_epoch.append(train_loss)
        val_loss_epoch.append(val_loss)

        scheduler.step()

        if val_loss<val_loss_min:
            val_loss_min=val_loss
            best_epoch = epoch
            print(f'best epoch: {best_epoch}, val loss: {val_loss}, Minimum val loss!!!!')
            torch.save(model.state_dict(), modelpath + '/model'+'.pkl')

        if (epoch) % 1 == 0:
            batch_x = range(0, len(train_loss_epoch))
            plt.plot(batch_x, train_loss_epoch)
            plt.close()
            batch_x_val = range(0, len(val_loss_epoch))
            plt.plot(batch_x_val, val_loss_epoch)
            plt.close()

            print('-------------------- Epoch %s ------------------' % (epoch + 1))
            print(f"train loss: {train_loss}")
            print(f"val loss: {val_loss}")
            print(f"rmse on test: {rmse}")

            if hasattr(model.dynamics, 'dynamics'):
                w, _ = np.linalg.eig(model.dynamics.dynamics.weight.data.cpu().numpy())

    return model
