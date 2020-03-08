
class Config:
    caption_data_path = 'caption.pth'
    img_path = None

    img_feature_path = 'result.pth'
    scale_size = 300
    img_size = 224
    batch_size = 8
    shuffle = True
    num_workers = 4
    rnn_hidden = 256
    embedding_dim = 256
    num_layers = 2
    share_embedding_weights = False

    prefix = 'checkpoints/caption'
    env='caption'
    plot_every=10
    debug_file='/tmp/debug'

    model_ckpt = "ImageCaption"
    lr = 1e-3
    use_gpu = False
    epoch = 1

    test_img = 'example.jpeg'
    test_prefix = 'img/'

