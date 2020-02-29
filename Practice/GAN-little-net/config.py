import torch as t


class Config(object):
    data_path = 'data/'
    num_works = 4
    image_size = 96
    batch_size = 256
    max_epoch = 200
    lr1 = 2e-4
    lr2 = 2e-4
    beta1 = 0.5
    gpu = False
    nc = 100
    ngf = 64
    ndf = 64

    save_path = 'imgs/'
    vis = True
    env='GAN'
    plot_every = 20

    debug_file = '/tmp/debug'
    d_every = 1
    g_every = 5
    save_every = 10
    netd_path = None
    netg_path = None

    gen_img = 'result.png'
    gen_num = 64
    gen_search_num = 512
    gen_mean = 0
    gen_std = 1



opt = Config()