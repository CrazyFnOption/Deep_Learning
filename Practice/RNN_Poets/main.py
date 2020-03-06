import ipdb
import tqdm
import sys
import fire
import os
import torch as t
import torch.utils.data
from data import get_data
from model import PoetModel
from torch import nn
from utils import Visualizer
from torchnet import meter
from config import opt


def train(**kwargs):
    for k, v in kwargs.items():
        setattr(opt, k, v)
    opt.device = t.device('cuda') if opt.use_gpu else t.device('cpu')
    vis = Visualizer(env=opt.env)

    data, word2ix, ix2word = get_data(opt)
    data = t.from_numpy(data)
    dataloader = t.utils.data.DataLoader(data, batch_size=opt.batch_size, shuffle=True, num_workers=1)

    model = PoetModel(len(word2ix), 128, 256)
    optim = t.optim.Adam(model.parameters(), lr= opt.lr)
    criterion = nn.CrossEntropyLoss()

    if opt.model_path:
        model.load_state_dict(t.load(opt.model_path))
    model.to(opt.device)

    loss_meter = meter.AverageValueMeter()

    for epoch in range(opt.max_epoch):
        loss_meter.reset()
        for ii, data_ in tqdm.tqdm(enumerate(dataloader)):
            data_ = data_.long().transpose(1, 0).contiguous()
            data_ = data_.to(opt.device)
            optim.zero_grad()
            input_, target_ = data_[:-1, :], data_[1:,:]
            output_, _ = model(input_)
            loss = criterion(output_, target_.view(-1))
            loss.backward()
            optim.step()

            # 这里必须写上item() 因为生成的是Tensor类型，防止动态图的显存会炸
            loss_meter.add(loss.item())

            if (ii + 1) % opt.plot_erevy == 0:
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()

                vis.plot('loss', loss_meter.value()[0])
                poetrys = [[ix2word[_word] for _word in data_[:, _iii].tolist()]
                          for _iii in range(data_.shape[1])][:16]
                vis.text('</br>'.join([''.join(poetry) for poetry in poetrys]), win=u'origin_poem')

                gen_poetries = []
                #接下来就是验证 模型在训练过程中生成的诗词
                for word in list(u'春江花月夜凉如水'):
                    gen_poetry = ''.join(generate(model, word, ix2word, word2ix))
                    gen_poetries.append(gen_poetry)

                vis.text('</br>'.join([''.join(poetry) for poetry in gen_poetries]), win=u'gen_poem')

        t.save(model.state_dict(), '%s_%s.pth' %(opt.model_prefix, epoch))


def gen(**kwargs):
    for k, v in kwargs.items():
        setattr(opt, k, v)
    data, word2ix, ix2word = get_data(opt)
    device = t.device('cuda') if opt.use_gpu else t.device('cpu')
    model = PoetModel(len(word2ix), 128, 256)
    model.load_state_dict(t.load(opt.model_path, map_location= lambda s, _: s))
    model.to(device)

    # 处理字符串
    # 并且考虑到两个版本之间的差异，将其兼容了一下
    if sys.version_info.major  == 3:
        if opt.start_words.isprintable():
            start_words = opt.start_words
            prefix_words = opt.prefix_words
        else:
            start_words = opt.start_words.encode('ascii', 'surrogateescape').decode('utf8')
            prefix_words = opt.prefix_words.encode('ascii', 'surrogateescape').decode('utf8') if opt.prefix_words else None
    else:
        start_words = opt.start_words.decode('utf8')
        prefix_words = opt.prefix_words.decode('utf8') if opt.prefix_words else None

    start_words = opt.start_words.replace(',', u'，').replace('?', u'？').replace('.', u'。')
    gen_poetry = gen_acrostic if opt.acrostic else generate
    result = gen_poetry(model, start_words, ix2word, word2ix, prefix_words)
    print(''.join(result))



def generate(model, start_words, ix2word, word2ix, prefix_words= None):
    result = list(start_words)
    start_words_len = len(start_words)
    input = t.Tensor([word2ix['<START>']]).view(1,1).long()
    if opt.use_gpu : input = input.cuda()
    else: input = input.cpu()
    hidden = None

    if prefix_words is not None:
        for word in prefix_words:
            output, hidden = model(input, hidden)
            input = input.data.new([word2ix[word]]).view(1, 1)

    for i in range(opt.max_gen_len):
        output, hidden = model(input, hidden)
        if i < start_words_len:
            w = result[i]
            input = input.data.new([word2ix[w]]).view(1, 1)
        else:
            top_index = output.data[0].topk(1)[1][0].item()
            w = ix2word[top_index]
            result.append(w)
            input = input.data.new([top_index]).view(1, 1)
        if w == '<EOP>':
            del result[-1]
            break
    return result


def gen_acrostic(model, start_words, ix2word, word2ix, prefix_words= None):
    result = []
    start_word_len = len(start_words)
    input = (t.Tensor([word2ix['<START>']]).view(1, 1).long())
    if opt.use_gpu: input = input.cuda()
    hidden = None
    index = 0
    # 这里与前面相比 主要是多了一个 向前的词汇
    pre_word = '<START>'

    if prefix_words:
        for word in prefix_words:
            output, hidden = model(input, hidden)
            input = input.data.new([word2ix[word]]).view(1, 1)

    for i in range(opt.max_gen_len):
        output, hidden = model(input, hidden)
        top_index = output.data[0].topk(1)[1][0].item()
        w = ix2word[top_index]

        if pre_word in {u'。', u'！', '<START>'} :
            if index == start_word_len :
                # 这里就已经将藏头诗全部给用完了
                break
            else:
                w = start_words[index]
                index += 1
                input = (input.data.new([word2ix[w]])).view(1, 1)
        else:
            input = input.data.new([word2ix[w]]).view(1, 1)
        result.append(w)
        pre_word = w
    return result


if __name__ == '__main__':
    fire.Fire()