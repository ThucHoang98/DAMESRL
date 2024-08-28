import os
import re
from random import shuffle

from  DAMESRL.liir.dame.core.representation.Sentence import Sentence
from  DAMESRL.liir.dame.core.representation.Text import Text
from  DAMESRL.liir.dame.core.representation.Word import Word

__author__ = "Quynh Do"
__copyright__ = "Copyright 2017, DAME"


class TextReader:
    def __init__(self, sources):
        self.sources = sources

    def read_file(self, fn):
        f = open(fn, "r")
        sentences = []
        words = []
        for line in f.readlines():
            line = line.strip()
            if re.match("\\s+", line) or line == "":
                if len(words) > 0:
                    sentences.append(Sentence([Word(w) for w in words]))
                    words = []
            else:
                words.append(re.split("\\s+", line)[1])

        f.close()
        if len(words) > 0:
            sentences.append(Sentence([Word(w) for w in words]))
        return Text(sentences)

    def read(self, fn):

        if os.path.isfile(fn):
            return self.read_file(fn)

        if os.path.isdir(fn):
            rs = Text()
            for file_name in os.listdir(fn):
                rs.extend(self.read(os.path.join(fn, file_name)))
            return rs
        if isinstance(fn, list):
            rs = Text()
            for source in fn:
                rs.extend(self.read(source))
            return rs

    def read_all(self):
        return self.read(self.sources)

    def select(self, percent, need_shuffer=False):
        all_sentences = self.read_all()
        indexes = [x for x in range(len(all_sentences))]
        if need_shuffer:
            shuffle(indexes)

        return [all_sentences[idx] for idx in indexes[:int(len(all_sentences) * percent / 100)]]
