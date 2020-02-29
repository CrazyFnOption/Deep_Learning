import os
import tqdm
import fire
import ipdb
import visdom
import torch as t
import torchvision as tv
from torch.autograd import Variable as V
from models import Net_G,Net_D
from config import opt
from visualize import Visualize
from torch.utils.data import DataLoader
from torchnet.meter import AverageValueMeter


def train(**kwargs):
    global vis, fix_fake_img
    for k_,v_ in kwargs.items():
        setattr(opt, k_, v_)
    device = t.device('cuda') if opt.gpu else t.device('cpu')

    vv = visdom.Visdom(env='image')
    print(opt.netd_path)
    print(opt.netg_path)
    if opt.vis:
        from visualize import Visualize
        vis = Visualize(opt.env)

    transforms = tv.transforms.Compose([
        tv.transforms.Resize(opt.image_size),
        tv.transforms.CenterCrop(opt.image_size),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    dataset = tv.datasets.ImageFolder(opt.data_path,transform=transforms)
    dataloader = DataLoader(dataset,
                            opt.batch_size,
                            shuffle=True,
                            num_workers=opt.num_works,
                            drop_last=True)

    net_g,net_d = Net_G(opt),Net_D(opt)
    if opt.netd_path:
        net_d.load_state_dict(t.load(opt.netd_path, map_location=lambda storage, loc: storage))
    if opt.netg_path:
        net_g.load_state_dict(t.load(opt.netg_path, map_location=lambda storage, loc:storage))

    net_g.to(device)
    net_d.to(device)

    optimizer_g = t.optim.Adam(net_g.parameters(), opt.lr1, betas=(opt.beta1, 0.999))
    optimizer_d = t.optim.Adam(net_d.parameters(), opt.lr2, betas=(opt.beta1, 0.999))

    criterion = t.nn.BCELoss().to(device)

    true_labels = V(t.ones(opt.batch_size).to(device))
    fake_labels = V(t.zeros(opt.batch_size).to(device))
    noises = V(t.randn(opt.batch_size, opt.nc, 1, 1).to(device))
    fix_noises = V(t.randn(opt.batch_size, opt.nc, 1, 1).to(device))

    error_d_meter = AverageValueMeter()
    error_g_meter = AverageValueMeter()

    epochs = range(opt.max_epoch)
    for epoch in epochs:
        for ii, (img,_) in tqdm.tqdm(enumerate(dataloader)):
            real_img = img.to(device)

            if ii % opt.d_every == 0:
                optimizer_d.zero_grad()
                output = net_d(real_img)
                error_d_real = criterion(output,true_labels)
                error_d_real.backward()

                noises.data.copy_(t.randn(opt.batch_size, opt.nc, 1, 1))
                fake_img = net_g(noises).detach()
                output = net_d(fake_img)
                error_d_fake = criterion(output, fake_labels)
                error_d_fake.backward()
                optimizer_d.step()
                error_d = error_d_fake + error_d_real
                error_d_meter.add(error_d.item())

            if ii % opt.g_every == 0:
                optimizer_g.zero_grad()
                noises.data.copy_(t.randn(opt.batch_size, opt.nc, 1, 1))
                fake_img = net_g(noises)
                output = net_d(fake_img)
                error_g = criterion(output, true_labels)
                error_g.backward()
                optimizer_g.step()
                error_g_meter.add(error_g.item())

            if opt.vis and ii % opt.plot_every == opt.plot_every - 1:
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()
                fix_fake_img = net_g(fix_noises)
                # vis.img('fixfake',fix_fake_img.detach().data.cpu().numpy()[:64] * 0.5 + 0.5)
                # vis.img('real', (real_img.cpu().numpy()[:64] * 0.5 + 0.5))
                vv.images(fix_fake_img.detach().cpu().numpy()[:64] * 0.5 + 0.5, win='fixfake')
                vv.images(real_img.data.cpu().numpy()[:64] * 0.5 + 0.5, win='real')

                vis.plot('errord', error_d_meter.value()[0])
                vis.plot('errorg', error_g_meter.value()[0])
            print("current epoch: %d" % epoch)
        if (epoch + 1) % opt.save_every == 0:
            tv.utils.save_image(fix_fake_img.data[:64], '%s/%s.png' %(opt.save_path, epoch)
                                , normalize=True,range=(-1,1))
            t.save(net_g.state_dict(),'checkpoints/netg_%s.pth' % epoch)
            t.save(net_d.state_dict(),'checkpoints/netd_%s.pth' % epoch)
            error_g_meter.reset()
            error_d_meter.reset()


def generate(**kwargs):
    for k,v in kwargs.items():
        setattr(opt, k, v)

    device = t.device('cuda') if opt.gpu else t.device('cpu')
    net_g,net_d = Net_G(opt).eval(),Net_D(opt).eval()
    noises = t.randn(opt.gen_search_num, opt.nc, 1, 1).normal_(opt.gen_mean, opt.gen_std)
    noises = noises.to(device)

    net_g.load_state_dict(t.load(opt.netg_path, map_location=lambda storage,loc:storage))
    net_d.load_state_dict(t.load(opt.netd_path, map_location=lambda storage,loc:storage))
    net_g.to(device)
    net_d.to(device)

    fake_img = net_g(noises)
    scores = net_d(fake_img).detach()

    indexs = scores.topk(opt.gen_num)[1]
    result = []

    for ii in indexs:
        result.append(fake_img.data[ii])

    tv.utils.save_image(t.stack(result), opt.gen_num, normalize=True, range=(-1,1))

if __name__ == '__main__':
    fire.Fire()


