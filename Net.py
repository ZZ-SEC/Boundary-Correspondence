import torch
from torch import nn


class Smooth(nn.Module):
    def __init__(self):
        super(Smooth, self).__init__()
        kernal_size = 51
        padding = kernal_size // 2
        layer = 8
        self.cnn = nn.Sequential(
            nn.Conv1d(1, layer, kernel_size=kernal_size, padding=padding, padding_mode='circular'),
            nn.ReLU(inplace=True),
            nn.Conv1d(layer, layer, kernel_size=kernal_size, padding=padding, padding_mode='circular'),
            nn.ReLU(inplace=True),
            nn.Conv1d(layer, layer, kernel_size=kernal_size, padding=padding, padding_mode='circular'),
            nn.ReLU(inplace=True),
            nn.Conv1d(layer, layer, kernel_size=kernal_size, padding=padding, padding_mode='circular'),
            nn.ReLU(inplace=True),
            nn.Conv1d(layer, layer, kernel_size=kernal_size, padding=padding, padding_mode='circular'),
            nn.ReLU(inplace=True),
            nn.Conv1d(layer, 1, kernel_size=kernal_size, padding=padding, padding_mode='circular'),
        )

    def forward(self, points):
        # batch*2*1024
        ret = self.cnn(points)
        # batch*4*1024
        return ret


class DNet(nn.Module):
    def __init__(self):
        super(DNet, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        kernal_size = 11
        padding = kernal_size // 2
        kernal_size_8 = 102
        padding_8 = 50
        kernal_size_4 = 52
        padding_4 = 25
        kernal_size_2 = 26
        padding_2 = 12
        layer = 24
        self.cnn1 = nn.Sequential(
            nn.Conv1d(1, layer, kernel_size=kernal_size_8,stride=2, padding=padding_8, padding_mode='circular'),
            # (N-102+2*50)/2+1
            # nn.BatchNorm1d(2*layer),
            nn.ReLU(inplace=True),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv1d(layer, 2 * layer, kernel_size=kernal_size_4,stride=2, padding=padding_4, padding_mode='circular'),
            # nn.BatchNorm1d(2*layer),
            nn.ReLU(inplace=True),
        )
        self.cnn3 = nn.Sequential(
            nn.Conv1d(2 * layer, 2 * layer, kernel_size=kernal_size_4,stride=2, padding=padding_4, padding_mode='circular'),
            # nn.BatchNorm1d(2*layer),
            nn.ReLU(inplace=True),
        )
        self.cnn4 = nn.Sequential(
            nn.Conv1d(2 * layer, 2 * layer, kernel_size=kernal_size_2,stride=2, padding=padding_2, padding_mode='circular'),
            # nn.BatchNorm1d(2 * layer),
            nn.ReLU(inplace=True),
        )
        self.cnn5 = nn.Sequential(
            nn.Conv1d(2 * layer, 4 * layer, kernel_size=kernal_size_2,stride=2, padding=padding_2, padding_mode='circular'),
            # nn.BatchNorm1d(2 * layer),
            nn.ReLU(inplace=True),
        )
        self.cnn6 = nn.Sequential(
            nn.Conv1d(4 * layer, 4 * layer, kernel_size=kernal_size, padding=padding, padding_mode='circular'),
            # nn.BatchNorm1d(2*layer),
            nn.ReLU(inplace=True),
        )
        self.tcnn1 = nn.Sequential(
            nn.ConvTranspose1d(4 * layer, 2 * layer, kernel_size=kernal_size_2,stride=2, padding=0),
            #(input-1)*stride+kernal-2*padding=(N-1)*2+26-2*12
            # nn.BatchNorm1d(2*layer),
            nn.ReLU(inplace=True),
        )
        self.tcnn2 = nn.Sequential(
            nn.ConvTranspose1d(2 * layer, 2 * layer, kernel_size=kernal_size_2,stride=2, padding=0),
            # nn.BatchNorm1d(2 * layer),
            nn.ReLU(inplace=True),
        )
        self.tcnn3 = nn.Sequential(
            nn.ConvTranspose1d(4 * layer, 2 * layer, kernel_size=kernal_size_4,stride=2, padding=0),
            # nn.BatchNorm1d(2 * layer),
            nn.ReLU(inplace=True),
        )
        self.tcnn4 = nn.Sequential(
            nn.ConvTranspose1d(2 * layer, layer, kernel_size=kernal_size_4,stride=2, padding=0),
            # nn.BatchNorm1d(2*layer),
            nn.ReLU(inplace=True),
        )
        self.tcnn5=nn.Sequential(
            nn.ConvTranspose1d(layer, layer, kernel_size=kernal_size_8, stride=2, padding=0),
            nn.ReLU(inplace=True),
        )
        self.cnnli = nn.Sequential(
            nn.Conv1d(2*layer, 2 * layer, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(2 * layer, 1, kernel_size=1)
        )
        self.tcnn32=nn.Sequential(
            nn.ConvTranspose1d(4*layer, layer, kernel_size=102, stride=32, padding=0),
            #pad=35
            nn.ReLU(inplace=True),
        )
        torch.autograd.set_detect_anomaly(True)
    def circular_tconv(self,x,tcnn,padding):
        x=tcnn(x)
        y=x.clone()
        y[:,:,padding:2*padding]+=x[:,:,-padding:]
        y[:, :, -2*padding:-padding] += x[:, :, :padding]
        return y[:,:,padding:-padding]
    def forward(self, points):
        # batch*2*1024
        ret = self.cnn1(points)
        ret = self.cnn2(ret)
        ret = self.cnn3(ret)
        ret3=ret.clone()
        ret = self.cnn4(ret)
        ret = self.cnn5(ret)
        ret5=self.circular_tconv(ret,self.tcnn32,35).clone()
        ret = self.cnn6(ret)
        ret = self.circular_tconv(ret,self.tcnn1,12)
        ret = self.circular_tconv(ret,self.tcnn2,12)
        ret = torch.cat([ret3, ret], 1)
        ret = self.circular_tconv(ret,self.tcnn3,25)
        ret = self.circular_tconv(ret, self.tcnn4,25)
        ret = self.circular_tconv(ret, self.tcnn5,50)
        ret = torch.cat([ret5, ret], 1)
        ret = self.cnnli(ret)
        # batch*4*1024
        return ret
