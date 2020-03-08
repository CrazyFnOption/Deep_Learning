import torch as t
from torch.utils import data
import os
from PIL import Image
import torchvision as tv
import numpy as np

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def create_collate_fn(padding, eos, max_length=50):

    """
        将多个样本拼接在一起成一个batch
        输入： list of data，形如
        [(img1, cap1, index1), (img2, cap2, index2) ....]

        拼接策略如下：
        - batch中每个样本的描述长度都是在变化的，不丢弃任何一个词\
          选取长度最长的句子，将所有句子pad成一样长
        - 长度不够的用</PAD>在结尾PAD
        - 没有START标识符
        - 如果长度刚好和词一样，那么就没有</EOS>

        返回：
        - imgs(Tensor): batch_size*2048(2048这里就是卷积层最后的结果)
        - cap_tensor(Tensor): batch_size*max_length
        - lengths(list of int): 长度为batch_size
        - index(list of int): 长度为batch_size
        """

    def collate_fn(img_cap):
        img_cap.sort(key=lambda p: len(p[1]), reverse=True)
        imgs, caps, indexs = zip(*img_cap)
        imgs = t.cat([img.unsqueeze(0) for img in imgs], 0)
        # 限制最大不能超过50个词 一句话，如果超过就修建成50个词
        lengths = [min(len(c) + 1, max_length) for c in caps]
        batch_length = max(lengths)
        # 直接先创建一个空向量
        cap_tensor = t.LongTensor(batch_length, len(caps)).fill_(padding)
        for i, c in enumerate(caps):
            end_cap = lengths[i] - 1
            if end_cap < batch_length:
                cap_tensor[end_cap, i] = eos
            cap_tensor[:end_cap, i].copy_(c[:end_cap])
        return (imgs, (cap_tensor, lengths), indexs)

    return collate_fn


class CaptionDataset(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        data = t.load(opt.caption_data_path)
        word2ix = data['word2ix']
        self.captions = data['caption']
        self.padding = word2ix.get(data.get('padding'))
        self.end = word2ix.get(data.get('end'))
        self._data = data
        self.ix2id = data['ix2id']
        self.all_imgs = t.load(opt.img_feature_path)
        self.train(True)

    def __getitem__(self, index):
        """
        返回：
        - img: 图像features 2048的向量
        - caption: 描述，形如LongTensor([1,3,5,2]),长度取决于描述长度
        - index: 下标，图像的序号，可以通过ix2id[index]获取对应图片文件名
        """
        index = index + self._start
        img = self.all_imgs[index]

        caption = self.captions[index]
        # 5句描述随机选一句
        rdn_index = np.random.choice(len(caption), 1)[0]
        caption = caption[rdn_index]
        return img, t.LongTensor(caption), index

    def __len__(self):
        return self.len_

    def train(self, training=True):
        self.training = training
        if self.training:
            self._start = 0
            self.len_ = len(self._data) - 10000
        else :
            self._start = len(self._data) - 10000
            self.len_ = 10000



def get_dataloader(opt):
    dataset = CaptionDataset(opt)
    dataloader = data.DataLoader(dataset,
                                 batch_size=opt.batch_size,
                                 shuffle=opt.shuffle,
                                 num_workers=opt.num_workers,
                                 collate_fn=create_collate_fn(dataset.padding, dataset.end))
    return dataloader


