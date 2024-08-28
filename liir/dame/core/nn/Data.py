from itertools import zip_longest
from operator import itemgetter
from random import shuffle

import numpy as np

__author__ = "Quynh Do"
__copyright__ = "Copyright 2017, DAME"


# manage data for training
class SimpleDataIteratorSEQ():
    def __init__(self, dt, need_shuffle=False, feed_last=False):
        self.dt = dt
        self.size = len(self.dt)
        self.indexes = list(range(self.size))
        self.epochs = 0
        self.fields = len(dt[0][0])
        self.cursor = 0
        self.need_shuffle = need_shuffle
        self.feed_last = feed_last
        if self.need_shuffle:
            self.shuffle()

    def shuffle(self):
        shuffle(self.indexes)
        self.cursor = 0

    def next_batch(self, n):
        if self.cursor + n > self.size:
            res = [self.dt[i] for i in self.indexes[self.cursor:]]
            if self.feed_last:
                res.extend([self.dt[i] for i in self.indexes[0:  self.cursor + n - self.size]])
            self.epochs += 1
            if self.need_shuffle:
                self.shuffle()
            else:
                self.cursor = 0
        else:
            res = [self.dt[i] for i in self.indexes[self.cursor:self.cursor + n]]
            self.cursor += n
        seq_lens = [len(seq) for seq in res]

        return self.prepare(res), seq_lens

    def all(self):
        seq_lens = [len(seq) for seq in self.dt]

        return self.prepare(self.dt), seq_lens

    def prepare(self, input):

        dt = ([[[i for i in element if i is not None] for element in list(zip_longest(*ix))] for ix in input])

        rs = [[i for i in element if i is not None] for element in list(zip_longest(*dt))]
        return [np.asarray(rst, dtype="int32") for rst in rs]


class PaddedDataIteratorSEQ(SimpleDataIteratorSEQ):
    def prepare(self, input):
        maxlen = max([len(seq) for seq in input])

        input = [seq + [tuple([0] * self.fields)] * (maxlen - len(seq)) for seq in input]
        dt = ([[[i for i in element if i is not None] for element in list(zip_longest(*ix))] for ix in input])

        rs = [[i for i in element if i is not None] for element in list(zip_longest(*dt))]
        return [np.asarray(rst, dtype="int32") for rst in rs]


class BucketedDataIteratorSEQ():
    def __init__(self, dt, num_buckets=5):
        self.dt = dt
        seq_lens = [len(seq) for seq in dt]
        df = [i[0] for i in sorted(enumerate(seq_lens), key=itemgetter(1))]

        self.size = int(len(df) / num_buckets)
        self.index_dts = []
        for bucket in range(num_buckets):
            self.index_dts.append(df[bucket * self.size: (bucket + 1) * self.size + 1])
        self.num_buckets = num_buckets

        self.cursor = np.array([0] * num_buckets)
        self.shuffle()

        self.epochs = 0

        self.fields = len(dt[0][0])

    def shuffle(self):
        for i in range(self.num_buckets):
            shuffle(self.index_dts[i])
            self.cursor[i] = 0

    def next_batch(self, n):

        while True:
            i = np.random.randint(0, self.num_buckets)
            if self.cursor[i] + n <= self.size:
                res = [self.dt[j] for j in self.index_dts[i][self.cursor[i]:self.cursor[i] + n]]
                self.cursor[i] += n

                if np.all(self.cursor + n > self.size):
                    self.epochs += 1
                    self.shuffle()

                seq_lens = [len(seq) for seq in res]
                return self.prepare(res), seq_lens

    def prepare(self, input):
        maxlen = max([len(seq) for seq in input])
        input = [seq + [tuple([0] * self.fields)] * (maxlen - len(seq)) for seq in input]
        dt = ([[[i for i in element if i is not None] for element in list(zip_longest(*ix))] for ix in input])

        rs = [[i for i in element if i is not None] for element in list(zip_longest(*dt))]
        return [np.asarray(rst, dtype="int32") for rst in rs]
