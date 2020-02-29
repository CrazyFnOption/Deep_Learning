from torch import nn


class Net_G(nn.Module):
    def __init__(self, opt):
        super(Net_G,self).__init__()

        ngf = opt.ngf # 卷积层中间用来描述给予图像的特征

        self.net = nn.Sequential(
            # 100 * 1 * 1 -> (ngf * 8) * 4 * 4
            nn.ConvTranspose2d(opt.nc, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            # (ngf * 8) * 4 * 4 -> (ngf * 4) * 8 * 8
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # (ngf * 4) * 8 * 8 -> (ngf * 2) * 16 * 16
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),

            # (ngf * 2) * 16 * 16 -> ngf * 32 * 32
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # ngf * 32 * 32 -> 3 * 96 * 96
            nn.ConvTranspose2d(ngf, 3, 5, 3, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


class Net_D(nn.Module):
    def __init__(self, opt):
        super(Net_D,self).__init__()
        ndf = opt.ndf

        self.net = nn.Sequential(
            # 3 * 96 * 96 -> ndf * 32 * 32
            nn.Conv2d(3, ndf, 5, 3, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2,True),

            # ndf * 32 * 32 -> (ndf * 2) * 16 * 16
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2,True),

            # (ndf * 2) * 16 * 16 -> (ndf * 4) * 8 * 8
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, True),

            # (ndf * 4) * 8 * 8 -> (ndf*8) * 4 * 4
            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, True),

            # (ndf*8) * 4 * 4 -> 100 * 1 * 1,
            nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).view(-1)

