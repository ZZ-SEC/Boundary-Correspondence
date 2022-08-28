import os
import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from matplotlib import pyplot as plt
from Param import DefaultConfig
from FPDataset import FPDataset
from Net import DNet,Smooth
from itertools import combinations
import time
# 超参数设置
opt = DefaultConfig()
# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def file_preparation(opt):
    num_samples=len(pd.read_csv(opt.original_csv_corners).values)
    if not os.path.exists(opt.base_dir):
        os.mkdir(opt.base_dir)
    train_dir = os.path.join(opt.base_dir, 'traindata')
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    test_dir = os.path.join(opt.base_dir, 'testdata')
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    val_dir = os.path.join(opt.base_dir, 'valdata')
    if not os.path.exists(val_dir):
        os.mkdir(val_dir)
    if not os.path.exists('./val'):
        os.mkdir('./val')
    ori_data_angles = pd.read_csv(opt.original_csv_angles).values
    ori_data_points = pd.read_csv(opt.original_csv_points).values
    ori_data_corners = pd.read_csv(opt.original_csv_corners).values.astype(np.int32)
    left=num_samples
    nums_val=[]
    sum_val=0
    for turn in range(0,10):
        nums_val.append(left//(10-turn))
        num_val = nums_val[turn]
        sum_val+=num_val
        left-=num_val
        num_test = int(0.2 * num_samples)
        num_train = num_samples - num_val - num_test
        data_angles = np.concatenate(
            [ori_data_angles[sum_val:, : ],ori_data_angles[0:sum_val, : ]],0)
        data_points = np.concatenate(
            [ori_data_points[2*sum_val:, : ],ori_data_points[0:2*sum_val, : ]],0)
        data_corners = np.concatenate(
            [ori_data_corners[sum_val:, : ],ori_data_corners[0:sum_val, : ]],0)
        # Train Data
        data_train_angles_o = data_angles[0:num_train, :]
        data_train_corners_o = data_corners[0:num_train, :]
        data_train_angles_inv = np.empty(data_train_angles_o.shape)
        data_train_corners_inv = np.empty(data_train_corners_o.shape, dtype=np.int32)
        for k in range(0, 1024):
            data_train_angles_inv[:, k] = data_train_angles_o[:, 1023 - k]
        for k in range(0, 4):
            data_train_corners_inv[:, k] = 1025 - data_train_corners_o[:, 3 - k]
        data_train_corners_inv = data_train_corners_inv * (data_train_corners_inv < 1025)
        data_train_angles = np.empty([data_train_angles_o.shape[0] * 2, data_train_angles_o.shape[1]])
        data_train_corners = np.empty([data_train_corners_o.shape[0] * 2, data_train_corners_o.shape[1]],
                                      dtype=np.int32)
        data_train_angles[0::2, :] = data_train_angles_o
        data_train_angles[1::2, :] = data_train_angles_inv
        data_train_corners[0::2, :] = data_train_corners_o
        data_train_corners[1::2, :] = data_train_corners_inv
        path_csv_train_angles = os.path.join(train_dir, 'train_angles' + str(turn) + '.csv')
        path_csv_train_corners = os.path.join(train_dir, 'train_corners' + str(turn) + '.csv')
        data = pd.DataFrame(data_train_angles)
        data.to_csv(path_csv_train_angles, index=False)
        data = pd.DataFrame(data_train_corners)
        data.to_csv(path_csv_train_corners, index=False)
        # Test Data
        data_test_angles = data_angles[(num_train):((num_train+num_test)), :]
        data_test_corners = data_corners[(num_train):(num_train+num_test), :]
        path_csv_test_angles = os.path.join(test_dir, 'test_angles'+str(turn)+'.csv')
        path_csv_test_corners = os.path.join(test_dir, 'test_corners'+str(turn)+'.csv')
        data = pd.DataFrame(data_test_angles)
        data.to_csv(path_csv_test_angles, index=False)
        data = pd.DataFrame(data_test_corners)
        data.to_csv(path_csv_test_corners, index=False)
        # Validation Data
        data_val_angles = data_angles[(num_train+num_test):(num_samples), :]
        data_val_corners = data_corners[(num_train + num_test): num_samples, :]
        path_csv_val_angles = os.path.join(val_dir, 'val_angles'+str(turn)+'.csv')
        path_csv_val_corners = os.path.join(val_dir, 'val_corners'+str(turn)+'.csv')
        data = pd.DataFrame(data_val_angles)
        data.to_csv(path_csv_val_angles, index=False)
        data = pd.DataFrame(data_val_corners)
        data.to_csv(path_csv_val_corners, index=False)
        # val_points
        path_csv_val_points = os.path.join('./val', 'points_val'+str(turn) + '.csv')
        data_val_points = data_points[2*(num_train + num_test):2*(num_samples), :]
        data = pd.DataFrame(data_val_points)
        data.to_csv(path_csv_val_points, index=False)

#数据文件位置、文件名
train_dir = os.path.join(opt.base_dir, 'traindata')
test_dir = os.path.join(opt.base_dir, 'testdata')
val_dir = os.path.join(opt.base_dir, 'valdata')

class Criterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss=nn.MSELoss()
    def line(self,n):
        x=np.array(list(range(0,n)))*(2*np.pi/n)
        return np.cos(x)*0.5+0.5
    def trans(self,index):
        ind = torch.zeros([1, 1024])
        index.sort()
        line = np.zeros(1024)
        max=index.copy()
        for i in range(0,3):
            line[max[i]:max[i+1]]=self.line(max[i+1]-max[i])
        dis=(max[0]-max[3])%1024
        temp=self.line(dis)
        line[max[3]:1024]=temp[0:1024-max[3]]
        line[0:max[0]] = temp[1024 - max[3]:]
        ind[0, :] = torch.from_numpy(line)
        return ind
    def forward(self,outputs,indexes):
        batch,N=outputs.size()[0],outputs.size()[2]
        ind=torch.zeros([batch,1,1024])
        indexes=indexes.detach().cpu().numpy()
        for i in range(0,batch):
            index=indexes[i,:]
            ind[i,:, :] = self.trans(index)

        ind=ind.to(device)
        loss=torch.sum(torch.pow(outputs[:,0,:]-ind[:,0,:],2))
        return loss/(batch*N)

# 训练
def train_work(net,trainloader,testloader,criterion,optimizer,mode=None,turn=0):
    loss_min=10000
    net.train()
    save_path = os.path.join(opt.save_dir, str(turn) + '.pth')
    loss_over_time = []
    test_loss = []
    smooth = Smooth().to(device)
    smooth.load_state_dict(torch.load('./model/smooth.pth'))
    smooth.eval()
    if mode=='Update'and os.path.exists(save_path):
        net.load_state_dict(torch.load(save_path))
        with torch.no_grad():
            loss=0.0
            print('第-1个epoch的loss=',)
            num=0
            for data in testloader:
                num+=1
                points,indexes = data
                points,indexes = points.to(device),indexes.to(device)
                outputs=net(points)
                loss+=criterion(outputs,indexes).item()
            print(loss/num)
            axis_x = np.array(list(range(0, 1024))) / 1024
            axis_y=outputs[0,0,:].detach().cpu().numpy()
            axis_y_smooth=smooth(outputs[0:1,:,:])[0,0,:].detach().cpu().numpy()
            plt.plot(axis_x,axis_y)
            plt.plot(axis_x, axis_y_smooth)
            plt.ion()
            plt.pause(1)
            plt.clf()
            test_loss.append(loss/num)
    for epoch in range(opt.EPOCH):
        running_loss=0.0
        i=0
        for data in trainloader:
            i+=1
            points,indexes=data
            points,indexes=points.to(device),indexes.to(device)
            optimizer.zero_grad()
            outputs=net(points)
            loss=criterion(outputs,indexes)
            loss.backward()
            optimizer.step()
            # 每训练10个batch打印一次平均loss
            running_loss += loss.item()
            if i % 10 == 0:
                avg_loss = running_loss / 10
                loss_over_time.append(avg_loss)
                print('[%d] loss: %.05f' % (epoch + 1, avg_loss))
                running_loss = 0.0
        # 每跑完一次epoch测试一下准确率
        with torch.no_grad():
            loss=0.0
            print('第',(epoch + 1),'个epoch的loss=',)
            num=0
            for data in testloader:
                num+=1
                points,indexes = data
                points,indexes = points.to(device),indexes.to(device)
                outputs=net(points)
                loss+=criterion(outputs,indexes).item()
            print(loss/num)
            axis_x = np.array(list(range(0, 1024))) / 1024
            axis_y=outputs[0,0,:].detach().cpu().numpy()
            axis_y_smooth = smooth(outputs[0:1, :, :])[0, 0, :].detach().cpu().numpy()
            plt.plot(axis_x,axis_y)
            plt.plot(axis_x, axis_y_smooth)
            plt.ion()
            plt.pause(1)
            plt.clf()
            test_loss.append(loss/num)
            if(loss/num==min(test_loss)):
                torch.save(net.state_dict(), save_path)
    print('Finished Training')
    return loss_over_time,test_loss

def show_loss(training_loss,xlable='',text=''):
    '''
    可视化损失变化
    '''
    plt.plot(training_loss)
    plt.xlabel(xlable)
    plt.ylabel('loss')
    ymax=np.max(np.array(training_loss[len(training_loss)//10:len(training_loss)//5]))
    ymin=np.min(np.array(training_loss))
    ymax+=(ymax-ymin)*0.3
    ymin -= (ymax - ymin) * 0.3/1.3
    plt.ylim(ymin, ymax) # consistent scale
    if text:
        plt.savefig(text)
    plt.ion()
    plt.pause(3)
    plt.clf()
def max4(x,angle):
    shape=x.shape
    N=shape[-1]
    x=x.reshape(-1,N)
    length=x.shape[0]
    result=np.zeros([length,4],dtype=np.int32)
    for i in range(0,length):
        xx=x[i,:]
        max_index=[]
        j=0
        while(j<N):
            temp=1
            for k in range(-10,10):
                idx=(j+k)%N
                if xx[j]<xx[idx]:
                    temp=0
                    j+=1
                    break
            if temp==1:
                max_index.append(j)
                j+=1
        max_index=np.array(max_index,dtype=np.int32)
        max_value=xx[max_index]
        if len(max_index)>=4:
            corners=[]
            for j in range(0,len(max_index)):
                if max_value[j]>0.5:
                    corners.append(max_index[j])
            corners=np.array(corners)
            if len(corners)<=4:
                arg = np.argsort(max_value)[-4:]
                result[i, :] = max_index[arg]
            else:
                L_min, corner_L_min=1e10,[]
                for corner in combinations(corners,4):
                    corner=np.array(corner)
                    corner.sort()
                    L=0
                    for k in range(0,4):
                        if k<3:
                            L+= angle[corner[k]+1:corner[k+1]].sum().abs()
                        else:
                            L+= (angle[corner[3]+1:].sum()+angle[:corner[0]].sum()).abs()
                        L-=np.abs(angle[corner[k]])
                    if L<L_min:
                        L_min=L
                        corner_L_min=corner
                result[i,:]=corner_L_min
        elif len(max_index)==3:
            d1 = angle[max_index[0] + 1:max_index[1]].sum()+(max_index[1]-max_index[0]-1)/1024
            d2 = angle[max_index[1] + 1:max_index[2]].sum()+(max_index[2]-max_index[1]-1)/1024
            d3 = angle[max_index[2] + 1:].sum()+angle[:max_index[0]].sum()+(1023-max_index[2]+max_index[0])/1024
            if d1>=d2 and d1>=d3:
                d=max_index[1]-max_index[0]
                dis=torch.empty([2,d-1])
                dis[0, :]=torch.linspace(1,d-1,d-1)
                dis[1, :] = torch.linspace(d - 1,1, d - 1)
                dis=dis.min(0)[0]
                temp = torch.argmax(angle[max_index[0] + 1:max_index[1]]*dis) + max_index[0] + 1
            elif d2>=d1 and d2>=d3:
                d = max_index[2] - max_index[1]
                dis = torch.empty([2, d - 1])
                dis[0, :] = torch.linspace(1, d - 1, d - 1)
                dis[1, :] = torch.linspace(d - 1, 1, d - 1)
                dis = dis.min(0)[0]
                temp = torch.argmax(angle[max_index[1] + 1:max_index[2]] * dis) + max_index[1] + 1
            else:
                d = N-(max_index[2] - max_index[0])
                dis = torch.empty([2, d - 1])
                dis[0, :] = torch.linspace(1, d - 1, d - 1)
                dis[1, :] = torch.linspace(d - 1, 1, d - 1)
                dis = dis.min(0)[0]
                if max_index[2]==N-1:
                    temp = torch.argmax(angle[0:max_index[0]] * dis)
                else:
                    temp = (torch.argmax(torch.cat([angle[max_index[2] + 1:],angle[:max_index[0]]]) * dis) + max_index[2] + 1)%N
                #temp = round((max_index[0] + N + max_index[2]) / 2)%N
            result[i, :] = np.array([max_index[0], max_index[1], max_index[2], temp],dtype=np.int32)
        elif len(max_index)==2:
            temp0=round((max_index[0]+max_index[1])/2)
            temp1=round((max_index[0]+max_index[1]+N)/2)%N
            result[i, :] = np.array([max_index[0], max_index[1],temp0,temp1],dtype=np.int32)
        elif len(max_index)==1:
            result[i, :] = (np.array([0,N//4,2*N//4,3*N//4],dtype=np.int32)+max_index[0])%N
        else:
            result[i, :] = np.array([0,N//4,2*N//4,3*N//4], dtype=np.int32)
    result.sort(axis=1)
    shape=list(shape[:-1])
    shape.append(4)
    result=result.reshape(shape)
    return result
def test_work(net,test_set,testloader,turn=0,move_range=0,plot=False):
    if plot and (not os.path.exists('./image')):
        os.mkdir('./image')
    if not os.path.exists('./val'):
        os.mkdir('./val')
    points2=pd.read_csv('./val/points_val'+str(turn)+'.csv').values
    model_path=os.path.join(opt.save_dir,str(turn)+'.pth')
    net.load_state_dict(torch.load(model_path))
    net.eval()
    smooth = Smooth().to(device)
    smooth.load_state_dict(torch.load('./model/smooth.pth'))
    smooth.eval()
    n=test_set.__len__()
    indexes_all=np.zeros([n,4])
    num=0
    for data in testloader:
        points, indexes = data
        points, indexes = points.to(device), indexes.to(device)
        outputs_ori = net(points)
        outputs=smooth(outputs_ori)
        for i in range(0,len(outputs)):
            num+=1
            print("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b" + str(num) + " / " + str(n), end="")
            axis_y = outputs[i, 0, :].detach().cpu().numpy()
            axis_y_ori=outputs_ori[i,0,:].detach().cpu().numpy()
            index=max4(axis_y,points[i,0,:].detach().cpu())
            index.sort()
            indexes_all[num - 1, :] = index
            for j in range(0, 4):
                ind = index[j]
                for k in range(1, move_range * 2 + 2):
                    if k % 2 == 0:
                        t = ind + (k // 2)
                    else:
                        t = ind - (k // 2)
                    t = t % (points.shape[2])
                    if points[i, :, t] > 0.4:
                        index[j] = t
                        break
            index.sort()
            if (index[0]!=index[1]) and (index[1]!=index[2]) and (index[2]!=index[3]):
                indexes_all[num-1,:] = index
            if plot:
                point2=points2[num*2-2:num*2,:]
                plt.plot(axis_y)
                plt.plot(axis_y_ori)
                plt.scatter(index,axis_y[index],s=30,c='r')
                plt.savefig("./image/"+str(turn)+'_'+str(num)+"_1.png")
                plt.clf()
                plt.plot(point2[0, :], point2[1, :])
                plt.scatter(point2[0,index], point2[1,index], s=30, c='r')
                plt.savefig("./image/" + str(turn) + '_'+str(num) + "_2.png")
                plt.clf()
    indexes_all=indexes_all[0:num,:]
    df = pd.DataFrame(indexes_all)
    df.to_csv('./val/indexes_val'+str(turn)+'.csv',index=False)


#========================================================
#  主程序

#========================================================

if __name__ == "__main__":
    # 数据集
    print('Preparing')
    file_preparation(opt)
    print('Training')
    start=time.time()
    for turn in range(0,10):
        train_csv_angles = os.path.join(train_dir, 'train_angles'+str(turn)+'.csv')
        test_csv_angles = os.path.join(test_dir, 'test_angles'+str(turn)+'.csv')
        val_csv_angles = os.path.join(val_dir, 'val_angles'+str(turn)+'.csv')
        train_csv_corners = os.path.join(train_dir, 'train_corners'+str(turn)+'.csv')
        test_csv_corners = os.path.join(test_dir, 'test_corners'+str(turn)+'.csv')
        val_csv_corners = os.path.join(val_dir, 'val_corners'+str(turn)+'.csv')
        # 定义DataSet
        train_set = FPDataset(train_csv_angles, train_csv_corners,rotate=True)
        test_set = FPDataset(test_csv_angles, test_csv_corners,rotate=True)
        val_set = FPDataset(val_csv_angles, val_csv_corners)
        # 神经网络
        net = DNet().to(device)
        for opt.LR,opt.BATCH_SIZE,opt.EPOCH,optimizer,wd,mode in \
                [[0.0001,16,100,'Adam',0.0003,None]]:
            trainloader = torch.utils.data.DataLoader(train_set, batch_size=opt.BATCH_SIZE, shuffle=True)
            testloader = torch.utils.data.DataLoader(test_set, batch_size=opt.BATCH_SIZE, shuffle=False)
            valloader = torch.utils.data.DataLoader(val_set, batch_size=128, shuffle=False)
            text = 'batchsize=' + str(opt.BATCH_SIZE) + 'epoch=' + str(opt.EPOCH)
            print("\n"+text)
            # 定义损失函数loss function 和优化方式（采用Adam）
            if optimizer=='SGD':
                optimizer = optim.SGD(net.parameters(), lr=opt.LR,momentum=0.8)
            elif optimizer=='Adam':
                optimizer = optim.Adam(net.parameters(), lr=opt.LR,weight_decay=wd)
            else:
                optimizer = optim.RMSprop(net.parameters(), lr=opt.LR)
            criterion=Criterion()
            # 训练
            training_loss, test_loss = train_work(net, trainloader, testloader, criterion, optimizer,mode,turn)
            show_loss(training_loss,'10\'s of batches', text + ' loss'+str(turn)+'.png')
            show_loss(test_loss,'Epoches', text + ' test_loss'+str(turn)+'.png')
        print('Testing - ',turn)
        test_work(net,val_set,valloader,turn,move_range=20,plot=False)
    end=time.time()
    print("耗时 = ",round(end - start, 2),' secs')
