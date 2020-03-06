class Config(object):
    pickle_path = 'data/tang.npz'
    author = None
    constrain = None
    category = 'poet.tang'
    lr = 1e-3
    weight_dacay = 1e-4
    use_gpu = False
    max_epoch = 251
    batch_size = 128
    maxlen = 125
    plot_erevy = 20
    env = 'poetry'
    max_gen_len = 200       # 生成诗歌的长度
    debug_file = '/tmp/debug'
    model_prefix = 'checkpoints/tang'

    acrostic = False
    model_path = 'checkpoints/tang_model.pth'
    prefix_words = None
    start_words = None


opt = Config()
