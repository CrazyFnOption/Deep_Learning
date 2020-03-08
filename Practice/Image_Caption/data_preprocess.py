import torch as t
import numpy as np
import json
import jieba
import tqdm


class Config:
    annotation_file = 'caption_train_annotations_20170902.json'
    unknown = '</UNKNOWN>'
    end = '</EOS>'
    padding = '</PAD>'
    max_words = 10000
    min_appear = 2
    save_path = 'caption.pth'


def process(**kwargs):
    opt = Config()
    for k, v in kwargs.items():
        setattr(opt, k, v)

    with open(opt.annotation_file) as f:
        data = json.load(f)

    id2ix = {item['image_id']: ix for ix, item in enumerate(data)}
    ix2id = {ix: pic for pic, ix in id2ix.items()}
    assert id2ix[ix2id[10]] == 10

    captions = [item['caption'] for item in data]
    cut_captions = [[list(jieba.cut(ii, cut_all=False) for ii in item)] for item in tqdm.tqdm(captions)]

    word_nums = {}  # 每一个词出现的次数

    # 这里的语法结构很神奇，需要后面自己去加深一下理解
    # 下面就是过滤低频词语
    def update(word_nums):
        def fun(word):
            word_nums[word] = word_nums.get(word, 0) + 1  # 每出现一次就增加一次
            return None
        return fun

    lambda_ = update(word_nums)
    # 需要去测试一下，这一句是否与下面一句是相同的效果
    #_ = {[[lambda_(word) for word in sentence] for sentence in sentences]for sentences in cut_captions}
    _ = {lambda_(word) for sentences in cut_captions for sentence in sentences for word in sentence}
    word_nums_list = sorted([(num, word) for word, num in word_nums.items()], reverse=True)  # 倒序排列
    # 排在前1w个高频词汇 直接将其放入到可以操作的字典李
    words = [word[1] for word in word_nums_list[:opt.max_words] if word[0] >= opt.min_appear]
    words = [opt.unknown, opt.padding, opt.end] + words
    word2ix = {word: ix for ix, word in enumerate(words)}
    ix2word = {ix: word for word, ix in word2ix.items()}
    assert word2ix[ix2word[123]] == 123

    #这里default的意思就是如果不存在 就返回opt.unknown
    ix_captions = [[[word2ix.get(word, default=word2ix.get(opt.unknown)) for word in sentence] for sentence in item] for item in cut_captions]

    readme = u"""
    word：词
    ix:index
    id:图片名
    caption: 分词之后的描述，通过ix2word可以获得原始中文词
    """
    results = {
        'caption': ix_captions,
        'word2ix': word2ix,
        'ix2word': ix2word,
        'ix2id': ix2id,
        'id2ix': id2ix,
        'padding': '</PAD>',
        'end': '</EOS>',
        'readme': readme
    }
    t.save(results, opt.save_path)
    print('save file in %s' % opt.save_path)

    def test(ix, ix2= 4):
        results = t.load(opt.save_path)
        ix2word = results['ix2word']
        examples = results['caption'][ix][4] # 第ix张图的第四句话
        sentences_p = (''.join([ix2word[ii] for ii in examples]))
        sentences_r = data[ix]['caption'][ix2]
        assert sentences_p == sentences_r, 'test failed'
    test(1000)
    print('test success')
