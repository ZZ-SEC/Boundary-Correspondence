import random
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import  torch

class FPDataset(Dataset):
    def __init__(self, csv_points, csv_corners,rotate=False):
        self.points = pd.read_csv(csv_points).values.astype(np.float32)
        self.corners = pd.read_csv(csv_corners).values.astype(np.int)
        self.N=len(self.points)
        self.rotate=rotate

    def __len__(self):
        """
        继承 Dataset 类后,必须重写的一个方法
        返回数据集的大小
        :return:
        """
        return self.N

    def __getitem__(self, idx):
        """
        继承 Dataset 类后,必须重写的一个方法
        返回第 idx 个图像及相关信息
        :param idx:
        :return:
        """
        point = self.points[idx:idx+1, :]
        point = torch.from_numpy(point)
        corner = self.corners[idx, :] - 1
        if self.rotate:
            rand=random.randint(0,1023)
            point=torch.cat([point[:,rand:],point[:,0:rand]],1)
            corner=(corner-rand)%1024
        corner.sort()
        corner=torch.from_numpy(corner)
        return point,corner
class PreDataset(Dataset):

    def __init__(self, csv_corners,rotate=False):
        self.corners = pd.read_csv(csv_corners).values.astype(np.int)
        self.N=len(self.corners)
        self.rotate=rotate
        self.points = np.random.random([self.N,1024]).astype(np.float32) * 2 - 1

    def __len__(self):
        """
        继承 Dataset 类后,必须重写的一个方法
        返回数据集的大小
        :return:
        """
        return self.N

    def __getitem__(self, idx):
        """
        继承 Dataset 类后,必须重写的一个方法
        返回第 idx 个图像及相关信息
        :param idx:
        :return:
        """
        point = self.points[idx:idx+1, :]
        point = torch.from_numpy(point)
        corner = self.corners[idx, :] - 1
        if self.rotate:
            rand=random.randint(0,1023)
            point=torch.cat([point[:,rand:],point[:,0:rand]],1)
            corner=(corner-rand)%1024
        corner.sort()
        corner=torch.from_numpy(corner)
        return point,corner