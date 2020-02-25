import warnings
import torch as t


class DefaultConfig(object):
    env = 'default'
    vis_port = 8097
    model = 'ResNet34' #使用训练模型

    train_data_root = './data/train/'
    test_data_root = './data/test1/'
    load_model_path = None # 加载保存的模型路径

    batch_size = 32
    use_gpu = False
    num_workers = 4 # 读取工作的路径
    print_freg = 20 # 后面打印的行数

    debug_file = '/tmp/debug'
    result_file = 'result.csv'

    max_epoch = 10
    lr = 0.001
    lr_decay = 0.5
    weight_decay = 0e-5

    def _parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        opt.device = t.device('cuda') if opt.use_gpu else t.device('cpu')

        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))

opt = DefaultConfig()
