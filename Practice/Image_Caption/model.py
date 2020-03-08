import torch as t
import torchvision as tv
from utils.beam_search import CaptionGenerator
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
import time


class CaptionModel(nn.Module):
    def __init__(self, opt, word2ix, ix2word):
        super(CaptionModel, self).__init__()
        self.ix2word = ix2word
        self.word2ix = word2ix
        self.opt = opt
        self.fc = nn.Linear(2048, opt.rnn_hidden)

        self.rnn = nn.LSTM(opt.embedding_dim, opt.rnn_hidden, num_layers=opt.num_layers)
        self.classifier = nn.Linear(opt.rnn_hidden, len(word2ix))
        #  其的参数为 所有词语以及词语的词向量
        self.embedding = nn.Embedding(len(word2ix), opt.embedding_dim)

    def forward(self, img_feats, captions, lengths):
        # 将相应的图片解释加入到256维的词向量
        # seq * batch_size * word_vec
        embeddings = self.embedding(captions)
        # 将图片根据全链接层得到256维的向量
        # 1 * batch_size * 256
        img_feats = self.fc(img_feats).unsqueeze(0)
        # 这里直接将图片的信息直接看作成 第一个词的词向量
        # 记住cat的作用是将tuple或者list中的tensor链接在一起
        embeddings = t.cat([img_feats, embeddings], 0)

        pack_embeddings = pack_padded_sequence(embeddings, lengths)
        output, state = self.rnn(pack_embeddings)
        pred = self.classifier(output[0])
        return pred, state

    def generate(self, img, eos_token='</EOS>', beam_size=3, max_caption_length=30,
                 length_normalization_factor=0.0):
        cap_gen = CaptionGenerator(embedder=self.embedding,
                                   rnn=self.rnn,
                                   classifier=self.classifier,
                                   eos_id=self.word2ix[eos_token],
                                   beam_size=beam_size,
                                   max_caption_length=max_caption_length,
                                   length_normalization_factor=length_normalization_factor)

        if next(self.parameters()).is_cuda:
            img = img.cuda()
        # img.size = 1 * 2048
        img = img.unsqueeze(0)
        # img.size = 1 * 1 * 2048
        img = self.fc(img).unsqueeze(0)
        # img.size = 1 * 1 * 256
        sentences, score = cap_gen.beam_search(img)
        sentences = [sent for sent in sentences]
        res = sentences[score.index(max(score))]
        del res[-1]
        res = ''.join([self.ix2word[ii.item()] for ii in res]) + u'。'
        return res

    def save(self, path=None, **kwargs):
        if path is None:
            path = '{prefix}_{time}'.format(prefix=self.opt.prefix,
                                            time=time.strftime('%m%d_%H%M'))
        states = self.states()
        states.update(kwargs)
        t.save(states, path)
        return path

    def load(self, path, load_opt=False):
        data = t.load(path, map_location=lambda s, l: s)
        state_dict = data['state_dict']
        self.load_state_dict(state_dict)

        if load_opt:
            for k, v in data['opt'].items():
                setattr(self.opt, k, v)

        return self

    def get_optimizer(self, lr):
        return t.optim.Adam(self.parameters(), lr=lr)

    # 将模型的一切进行一个封装
    def state(self):
        opt_state_dict = {
            attr: getattr(self.opt, attr)
            for attr in dir(self.opt)
            if not attr.startswith('__')
        }

        return {
            'state_dict':self.state_dict(),
            'opt': opt_state_dict
        }


