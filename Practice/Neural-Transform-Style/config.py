
class Config(object):
    use_gpu = False
    model_path = None

    image_size = 256
    batch_size = 8
    data_root = 'data/'
    num_works = 4

    lr = 1e-3
    max_epoch = 2
    content_weight = 1e5
    style_weight = 1e10

    style_path = 'style.png'
    env = 'neural-style'
    plot_every = 10

    debug_file = '/tmp/debug'

    content_path = 'input.png'
    result_path = 'output.png'


