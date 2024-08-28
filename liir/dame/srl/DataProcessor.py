import os

import numpy as np

from  DAMESRL.liir.dame.core.io.CoNLL2005Reader import CoNLL2005Reader
from  DAMESRL.liir.dame.core.nn.WEDict import WEDict
from  DAMESRL.liir.dame.core.representation.Predicate import Predicate


class DataProcessor:
    def __init__(self, model_dir=None):

        self.data = None
        self.UNK = "**UNK**"
        self.PAD = "**PAD**"
        self.NUM = "**num**"
        self.labels = [self.PAD, self.UNK]
        self.words = [self.PAD, self.UNK, self.NUM]
        self.word_count = {self.PAD: 1000, self.UNK: 1000, self.NUM: 1000}
        self.max_len = 0
        self.extra_words = []

        self.LABEL_FILE = "vob.label.txt"
        self.WORD_FILE = "vob.word.txt"
        self.do_lowercase = True

        if os.path.exists(os.path.join(model_dir, self.LABEL_FILE)) and os.path.exists(
                os.path.join(model_dir, self.WORD_FILE)):
            self.labels = []
            self.words = []
            self.import_vobs(model_dir)

    def get_num_words(self):
        return len(self.words)

    def get_num_labels(self):
        return len(self.labels)

    def generate_vob(self, train_path):
        def generate_for_file(path, is_train=True):
            for sen in CoNLL2005Reader(path).read_all():
                to_process = True
                if 0 < self.max_len < len(sen):
                    to_process = False
                if to_process:
                    if len(sen.get_predicates()) > 0:
                        for w in sen:
                            wf = w.form.lower() if self.do_lowercase else w.form
                            self.add_to_vob(wf, self.words, self.word_count) if is_train else self.add_to_vob(wf,
                                                                                                              self.words)

                        for p in sen.get_predicates():
                            for arg in p.arguments:
                                self.add_to_vob(arg, self.labels)

        if isinstance(train_path, list):
            for i in range(len(train_path)):
                generate_for_file(train_path[i], True) if i == 0 else generate_for_file(train_path[i], False)
        else:
            generate_for_file(train_path)

    def extend_vob(self, eval_path):
        def generate_for_file(path):
            for sen in CoNLL2005Reader(path).read_all():
                to_process = True
                if self.max_len > 0 and len(sen) > self.max_len:
                    to_process = False
                if to_process:
                    if len(sen.get_predicates()) > 0:
                        for w in sen:
                            wf = w.form.lower() if self.do_lowercase else w.form
                            if wf not in self.words:
                                self.add_to_vob(wf, self.extra_words)

        if isinstance(eval_path, list):
            for i in range(len(eval_path)):
                generate_for_file(eval_path[i]) if i == 0 else generate_for_file(eval_path[i])
        else:
            generate_for_file(eval_path)

    def add_to_vob(self, val, vob, vob_count=None):
        if not val in vob:
            vob.append(val)
        if vob_count is not None:
            vob_count[val] = 1 if not val in vob_count else vob_count[val] + 1

    def get_index(self, val, vob):
        return vob.index(val) if val in vob else len(vob)

    def get_index_word(self, val):
        return self.words.index(val) if val in self.words else self.words.index(self.UNK)

    def is_empty(self):
        return len(self.labels) == 2

    def export_vobs(self, output):
        def write_dict(vob, path, count_vob=None):

            f = open(path, "w", encoding="utf-8")
            for x in vob:
                f.write(x)
                if count_vob is not None:
                    if x in count_vob:
                        f.write(" " + str(count_vob[x]))
                f.write("\n")
            f.close()

        write_dict(self.labels, os.path.join(output, self.LABEL_FILE))
        write_dict(self.words + self.extra_words, os.path.join(output, self.WORD_FILE), self.word_count)

    def import_vobs(self, dir_path):
        def read_dict(vob, path, vob_count=None):

            f = open(path, "r", encoding="utf-8")

            for line in f.readlines():
                line = line.strip()

                if line != "":
                    if vob_count is not None:
                        tmps = line.split(" ")

                        vob.append(tmps[0])
                        if len(tmps) > 1:
                            vob_count[tmps[0]] = int(tmps[1])
                    else:
                        vob.append(line)
            f.close()

        read_dict(self.labels, os.path.join(dir_path, self.LABEL_FILE))
        read_dict(self.words, os.path.join(dir_path, self.WORD_FILE), self.word_count)

    def smooth_word_vob(self, rare_frequency=3):
        self.words = [w for w in self.words if self.word_count[w] > rare_frequency]

    def process_sentence(self, sen):
        dt = []
        for pred in sen.get_predicates():
            this_pred = []
            for pos, word in enumerate(sen):
                wf = word.form.lower() if self.do_lowercase else word.form
                word_index = self.get_index_word(wf)

                this_pred.append((word_index, int(word == pred), self.get_index(pred.arguments[pos], self.labels)))

            dt.append(this_pred)
        return dt

    def get_data_train(self, train_path, data=None):
        train = []
        if data is None:
            data = CoNLL2005Reader(train_path).read_all()

        for sen in data:
            if self.max_len > 0:
                if len(sen) > self.max_len:
                    continue

            dt = self.process_sentence_not_in_vob(sen)

            if len(dt) > 0:
                train.extend(dt)
        return train

    def get_data_eval(self, eval_path, data=None):
        eval = []
        if data is None:
            data = CoNLL2005Reader(eval_path).read_all()
        for sen in data:

            dt = self.process_sentence_not_in_vob(sen)

            if len(dt) > 0:
                eval.extend(dt)
        return eval

    def process_sentence_not_in_vob(self, sen):
        dt = []
        for pred in sen.get_predicates():
            this_pred = []
            for pos, word in enumerate(sen):

                form = word.form.lower() if self.do_lowercase else word.form
                if form in self.words:
                    word_index = self.words.index(form)
                elif form in self.extra_words:
                    word_index = len(self.words) + self.extra_words.index(form)
                else:
                    word_index = self.words.index(self.UNK)

                this_pred.append((word_index, int(word == pred), self.get_index(pred.arguments[pos], self.labels)))

            dt.append(this_pred)
        return dt

    def get_data_eval_not_in_vob(self, eval_path, data=None):
        eval = []
        if data is None:
            data = CoNLL2005Reader(eval_path).read_all()
        for sen in data:

            dt = self.process_sentence_not_in_vob(sen)

            if len(dt) > 0:
                eval.extend(dt)
        return eval

    def get_we_dict(self, path):
        wedict = WEDict(path)
        arr = []
        for word in self.words:
            arr.append(wedict.get_we(word.lower()))
        return np.asarray(arr)

    def get_we_dict_extra(self, path):
        wedict = WEDict(path)
        arr = []
        for word in self.extra_words:
            arr.append(wedict.get_we(word.lower()))
        return np.asarray(arr)


class PredDataProcessor(DataProcessor):
    def __init__(self, model_dir=None):
        DataProcessor.__init__(self, model_dir)

        self.labels = ['False', 'True']

    def generate_vob(self, train_path):
        def generate_for_file(path, is_train=True):
            for sen in CoNLL2005Reader(path).read_all():
                to_process = True
                if 0 < self.max_len < len(sen):
                    to_process = False
                if to_process:

                    for w in sen:
                        wf = w.form.lower() if self.do_lowercase else w.form
                        if is_train:
                            self.add_to_vob(wf, self.words, self.word_count)
                        else:
                            self.add_to_vob(wf, self.words)

        if isinstance(train_path, list):
            for i in range(len(train_path)):
                generate_for_file(train_path[i], True) if i == 0 else generate_for_file(train_path[i], False)
        else:
            generate_for_file(train_path)
        self.add_to_vob("True", self.labels)
        self.add_to_vob("False", self.labels)

    def extend_vob(self, eval_path):
        def generate_for_file(path):
            for sen in CoNLL2005Reader(path).read_all():
                to_process = True
                if self.max_len > 0 and len(sen) > self.max_len:
                    to_process = False
                if to_process:

                    for w in sen:
                        wf = w.form.lower() if self.do_lowercase else w.form
                        if wf not in self.words:
                            self.add_to_vob(wf, self.extra_words)

        if isinstance(eval_path, list):
            for i in range(len(eval_path)):
                generate_for_file(eval_path[i]) if i == 0 else generate_for_file(eval_path[i])
        else:
            generate_for_file(eval_path)

    def process_sentence_not_in_vob(self, sen):
        dt = []
        for word in sen:

            form = word.form.lower() if self.do_lowercase else word.form
            if form in self.words:
                word_index = self.words.index(form)
            elif form in self.extra_words:
                word_index = len(self.words) + self.extra_words.index(form)
            else:
                word_index = self.words.index(self.UNK)

            dt.append((word_index, self.get_index(str((isinstance(word, Predicate))), self.labels)))

        return [dt]

    def process_sentence(self, sen):
        dt = []
        for word in sen:
            wf = word.form.lower() if self.do_lowercase else word.form

            word_index = self.get_index_word(wf)

            dt.append((word_index, self.get_index(str(isinstance(word, Predicate))), self.labels))

        return [dt]
