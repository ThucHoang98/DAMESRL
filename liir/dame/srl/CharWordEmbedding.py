# Learning word embeddings based on characters
import numpy as np


class CharProcessor:
    def __init__(self, word_vob):
        self.UNK_ID = 1
        self.PAD_ID = 0
        self.NUM_ID = 2

        self.vob = {"": self.UNK_ID}
        self.word_vob = word_vob

        for w in word_vob:
            self.process_word(w)

    def process_word(self, w):
        if w != "**UNK**" and w != "**PAD**" and w != "**NUM**":
            for char in w:
                if not char in self.vob:
                    self.vob[char] = len(self.vob)

    def get_word_presentation(self, w_id):
        if w_id == self.PAD_ID:
            return [0]
        if w_id == self.UNK_ID:
            return [1]
        if w_id == self.NUM_ID:
            return [2]

        w = self.word_vob[w_id]

        return [self.vob.get(chr, self.UNK_ID) for chr in w]

    def get_data(self, word_arrays):
        # word_arrays list of sequences of words
        rss = []
        wlens = []
        for sen in word_arrays.tolist():
            rss.append([self.get_word_presentation(w_id) for w_id in sen])
            wlens.append([len(self.word_vob[wid]) if wid > 2 else 1 for wid in sen])
        # do padding:

        maxlen = max([x for y in wlens for x in y])

        ins = [[seq + [0] * (maxlen - len(seq)) for seq in sen] for sen in rss]

        return np.asarray(ins, dtype="int32"), np.asarray(wlens, dtype="int32")
