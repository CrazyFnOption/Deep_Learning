from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch as t
from torch.autograd import Variable
from torch.nn.functional import log_softmax
import heapq

class Caption(object):
    def __init__(self, sentence, state, logprob, score, metadata=None):
        self.sentence = sentence
        self.state = state
        self.logprob = logprob
        self.score = score
        self.metadata = metadata

    # 在 python3 上已经被移除了
    # 下面是新的写法，只需要写两个 小于或等于
    def __cmp__(self, other):
        assert isinstance(other, Caption)
        if self.score == other.score:
            return 0
        elif self.score < other.score:
            return -1
        else:
            return 1

    def __lt__(self, other):
        assert isinstance(other, Caption)
        return self.score < other.score

    def __eq__(self, other):
        assert isinstance(other, Caption)
        return self.score == other.score


class TopN(object):
    def __init__(self, n):
        self._n = n
        self._data = []

    def size(self):
        assert self._data is not None
        return len(self._data)

    def push(self, x):
        assert self._data is not None
        if len(self._data) < self._n:
            heapq.heappush(self._data, x)
        else:
            heapq.heappushpop(self._data, x)

    def extract(self, sort=False):
        assert self._data is not None
        data = self._data
        self._data = None
        if sort:
            data.sort(reverse=True)
        return data

    def reset(self):
        self._data = []


class CaptionGenerator(object):
    def __init__(self, embedder, rnn, classifier, eos_id, beam_size=3,
                 max_caption_length=20, length_normalization_factor=0.0):
        self.embedder = embedder
        self.rnn = rnn
        self.classifier = classifier
        self.eos_id = eos_id
        self.beam_size = beam_size
        self.max_caption_length = max_caption_length
        self.length_normalization_factor = length_normalization_factor

    def beam_search(self, rnn_input, initial_state=None):
        """Runs beam search caption generation on a single image.
        Args:
          initial_state: An initial state for the recurrent model
        Returns:
          A list of Caption sorted by descending score.
        """

        def get_topk_words(embeddings, state):
            output, new_states = self.rnn(embeddings, state)
            output = self.classifier(output.squeeze(0))
            logprobs = log_softmax(output, dim=1)
            logprobs, words = logprobs.topk(self.beam_size, 1)
            return words.data, logprobs.data, new_states

        partial_captions = TopN(self.beam_size)
        complete_captions = TopN(self.beam_size)

        words, logprobs, new_state = get_topk_words(rnn_input, initial_state)
        for k in range(self.beam_size):
            cap = Caption(
                sentence=[words[0, k]],
                state=new_state,
                logprob=logprobs[0, k],
                score=logprobs[0, k])
            partial_captions.push(cap)

        for _ in range(self.max_caption_length - 1):
            partial_captions_list = partial_captions.extract()
            partial_captions.reset()

            input_feed = t.LongTensor([c.sentence[-1] for c in partial_captions_list])
            if rnn_input.is_cuda:
                input_feed = input_feed.cuda()

            input_feed.detach_()
            state_feed = [c.state for c in partial_captions_list]

            # 这个地方可以重点关注一下
            if isinstance(state_feed[0], tuple):
                # 这个地方为什么会用到 t.cat
                state_feed_h, state_feed_c = zip(*state_feed)
                state_feed = (t.cat(state_feed_h, 1),
                              t.cat(state_feed_c, 1))
            else:
                state_feed = t.cat(state_feed, 1)

            embeddings = self.embedder(input_feed).view(1, len(input_feed), -1)
            words, logprobs, new_states = get_topk_words(embeddings, state_feed)

            for i, partial_caption in enumerate(partial_captions_list):
                if isinstance(new_states, tuple):
                    state = (new_states[0].narrow(1, i, 1),
                             new_states[1].narrow(1, i, 1))
                else:
                    state = new_states[i]

                for k in range(self.beam_size):
                    w = words[i, k]
                    sentence = partial_caption.sentence + [w]
                    logprob = partial_caption.logprob + logprobs[i, k]
                    score = logprob
                    if w == self.eos_id:
                        if self.length_normalization_factor > 0:
                            score /= len(sentence)**self.length_normalization_factor
                        beam = Caption(sentence, state, logprob, score)
                        complete_captions.push(beam)
                    else:
                        beam = Caption(sentence, state, logprob, score)
                        partial_captions.push(beam)
            if partial_captions.size() == 0:
                break
        if not complete_captions.size():
            complete_captions = partial_captions

        caps = complete_captions.extract(sort=True)

        return [c.sentence for c in caps], [c.score for c in caps]



