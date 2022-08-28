#========================================================
#  超参数设置
#========================================================
import os
class DefaultConfig(object):
    def __init__(self):
        self.EPOCH = 100   # 遍历数据集次数
        self.BATCH_SIZE = 100   # 批处理尺寸(batch_size)
        self.LR = 0.001    # 学习率
        self.original_csv_points = './data/points1024.csv'
        self.original_csv_angles = './data/angles1024.csv'
        self.original_csv_corners = './data/index_min.csv'
        self.base_dir = './data'
        self.save_dir = './model/'
        if not os.path.exists('model'):
            os.mkdir('model')


    def parse(self, kwargs):
        # 更新配置参数
        for k, v in kwargs.items():
            if not hasattr(self, k):
                print('Warning: opt has not attribut {}'.format(k))
            else:
                setattr(self, k, v)

        # 打印配置信息
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if k == '__dict__':
                print(getattr(self, k))
