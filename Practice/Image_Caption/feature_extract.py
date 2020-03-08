from config import Config
import tqdm
import torch as t
from torch.autograd import Variable
import torchvision as tv
from torch.utils import data
import os
from PIL import Image
import numpy as np




IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
normalize = tv.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

class CaptionDataset(data.Dataset):
    def __init__(self, opt):
        self.transforms = tv.transforms.Compose([
            tv.transforms.Resize(256),
            tv.transforms.CenterCrop(256),
            tv.transforms.ToTensor(),
            normalize
        ])

        data = t.load(opt.caption_data_path)
        self.ix2id = data['ix2id']

        self.imgs = [os.path.join(opt.img_path, self.ix2id[ix]) for ix in range(len(self.ix2id))]

    def __getitem__(self, index):
        img = Image.open(self.imgs[index]).convert('RGB')
        img = self.transforms(img)
        return img, index

def get_loader(opt):
    dataset = CaptionDataset(opt.caption_data_path)
    dataloader = data.DataLoader(dataset,
                                 batch_size=opt.batch_size,
                                 shuffle=False, # 这里是需要与后面描述那一块的图像进行比对校准，所以不能打乱
                                 num_workers=opt.num_workers
                                 )
    return dataloader

def extract():
    opt = Config()
    t.set_grad_enabled(False)
    dataloader = get_loader(opt)
    results = t.Tensor(len(dataloader.dataset), 2048).fill_(0)
    batch_size = opt.batch_size

    resnet50 = tv.models.resnet50(pretrained=True)
    del resnet50.fc
    resnet50.fc = lambda x:x
    if opt.use_gpu:
        resnet50.cuda()

    for ii, (imgs, indexs) in tqdm.tqdm(enumerate(dataloader)):
        assert indexs[0] == batch_size * ii
        if opt.use_gpu:
            imgs = imgs.cuda()
        features = resnet50(imgs)
        results[batch_size*ii : batch_size*(ii+1),:] = features.data.cpu()

    t.save(results, 'results.pth')