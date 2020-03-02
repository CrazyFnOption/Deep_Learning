import fire
import os
import tqdm
import ipdb
import torchnet as tnt
import utils
import torch as t
import torchvision as tv
from config import Config
from torch.utils import data
from models.transformer_net import TransformerNet
from models.vgg16 import Vgg16
from torch.nn import functional as F


def train(**kwargs):
    opt = Config()
    for k, v in kwargs.items():
        setattr(opt, k, v)

    device = t.device('cuda') if opt.use_gpu else t.device('cpu')
    vis = utils.Visualizer(opt.env)

    transforms = tv.transforms.Compose([
        tv.transforms.Resize(opt.image_size),
        tv.transforms.CenterCrop(opt.image_size),
        tv.transforms.ToTensor(),
        tv.transforms.Lambda(lambda x: x * 255)
    ])

    dataset = tv.datasets.ImageFolder(opt.data_root, transforms)
    dataloader = data.DataLoader(dataset, opt.batch_size, num_workers=opt.num_works)

    transformer = TransformerNet()
    if opt.model_path:
        transformer.load_state_dict(t.load(opt.model_path, map_location=lambda storage, loc: storage))
    transformer.to(device)

    vgg = Vgg16.eval()
    vgg.to(device)

    ## 这里是为了保护Vgg里面的参数不要去变动， 直接将里面的参数改成不能反向传播
    for param in vgg.parameters():
        param.requires_grad = False

    optimizer = t.optim.Adam(transformer.parameters(), opt.lr)

    style = utils.get_style_data(opt.style_path)
    vis.img('style', (style.data[0] * 0.225 + 0.45).clamp(min=0, max=1))
    style = style.to(device)

    with t.no_grad():
        feature_style = vgg(style)
        gram_style = list(utils.gram_matrix(y) for y in feature_style)

    style_meter = tnt.meter.AverageValueMeter()
    content_meter = tnt.meter.AverageValueMeter()

    for epoch in range(opt.max_epoch):
        content_meter.reset()
        style_meter.reset()

        for ii, (x, _) in tqdm.tqdm(enumerate(dataloader)):
            optimizer.zero_grad()
            x = x.to(device)
            y = transforms(x)
            y = utils.normalize_batch(y)
            x = utils.normalize_batch(x)

            ## 这个地方得在仔细敲定一下，没有弄清楚这个features_y的结构
            features_y = vgg(y)
            features_x = vgg(x)
            content_loss = opt.content_weight * F.mse_loss(features_y.relu2_2, features_x.relu2_2)

            style_loss = 0.
            for ft_y, gm_s in zip(features_y, gram_style):
                gram_y = utils.gram_matrix(ft_y)
                style_loss += F.mse_loss(gram_y, gm_s.expand_as(gram_y))
            style_loss *= opt.style_weight

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            content_meter.add(content_loss.item())
            style_meter.add(style_loss.item())

            if (ii + 1) % opt.plot_every == 0:
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()

                vis.plot('content_loss', content_meter.value()[0])
                vis.plot('style_loss', style_meter.value()[0])
                vis.img('output', (y.data.cpu()[0] * 0.225 + 0.45).clamp(min=0, max=1))
                vis.img('input', (x.data.cpu()[0] * 0.225 + 0.45).clamp(min=0, max=1))

        vis.save([opt.env])
        t.save(transformer.state_dict(), 'checkpoints/%s_style.pth' % epoch)


def stylize(**kwargs):
    opt = Config()

    for k, v in kwargs.items():
        setattr(opt, k, v)
    device = t.device('cuda') if opt.use_gpu else t.device('cpu')

    content_image = tv.datasets.folder.default_loader(opt.content_path)
    content_transform = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Lambda(lambda x: x * 255)
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device).detach()

    style_model = TransformerNet().eval()
    style_model.load_state_dict(t.load(opt.model_path, map_location=lambda _s, _: _s))
    style_model.to(device)

    output = style_model(content_image)
    output_data = output.cpu().data[0]
    tv.utils.save_image((output_data / 255).clamp(max=1, min=0), opt.result_path)


if __name__ == '__main__':
    fire.Fire()