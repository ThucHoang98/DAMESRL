import os
import re

from  DAMESRL.liir.dame.core.io.CoNLL2005Writer import write_short_conll2005_format
from  DAMESRL.liir.dame.core.io.TextReader import TextReader
from  DAMESRL.liir.dame.core.representation.Predicate import Predicate
from  DAMESRL.liir.dame.core.representation.Sentence import Sentence
from  DAMESRL.liir.dame.core.representation.Text import Text
from  DAMESRL.liir.dame.core.representation.Word import Word

__author__ = "Quynh Do"
__copyright__ = "Copyright 2017, DAME"


class CoNLL2012Reader(TextReader):
    def __init__(self, sources, predicate_pos=6):  # predicate_pos: position of the predicate column in the data file
        TextReader.__init__(self, sources)
        self.predicate_pos = predicate_pos

    def read_file(self, fn):
        f = open(fn, "r", encoding="utf-8")
        sentences, words = [], []
        for line in f.readlines():
            line = line.strip()
            if line.startswith("#"):
                continue
            if re.match("\\s+", line) or line == "":
                if len(words) > 0:
                    sentences.append(words)
                    words = []
            else:
                words.append(line)
        f.close()
        print(sentences)
        if len(words) > 0:
            sentences.append(words)

        txt = Text()
        for lines in sentences:
            txt.append(self.generate_sentence(lines))
        return txt

    def generate_sentence(self, lines):
        pred_id = 0
        dt = []
        sen = Sentence()
        for line in lines:
            temps = re.split("\\s+", line)
            dt.append(temps)
            w = Word(temps[3])
            # if w.is_numeric():
            #    w.form = "**NUM**"

            if temps[self.predicate_pos + 1] != "-":
                w = Predicate(w)
                w.lemma = temps[self.predicate_pos]
            sen.append(w)

        # read srl information
        for pred in sen:
            if isinstance(pred, Predicate):
                args = []
                j = 0
                while j < len(lines):
                    tmps = dt[j]
                    lbl = tmps[5 + self.predicate_pos + pred_id]
                    match = re.match('\((.+)\*\)', lbl)

                    if match:
                        args.append("B-" + match.group(1))
                        j += 1
                    else:
                        match = re.match('\((.+)\*', lbl)
                        if match:
                            args.append("B-" + match.group(1))
                            for k in range(j + 1, len(lines)):
                                l1 = lines[k]
                                tmps1 = re.split("\\s+", l1)
                                match1 = re.match('\*\)', tmps1[5 + self.predicate_pos + pred_id])
                                args.append("I-" + match.group(1))
                                if match1:
                                    j = k + 1
                                    break
                        else:
                            args.append("O")
                            j += 1
                args2 = []
                for arg in args:
                    if "(" in arg:
                        args2.append(arg.split("(")[0])
                    else:
                        args2.append(arg)
                pred.arguments = args2
                pred_id += 1
        return sen

    def read(self, fn, tail="v4_gold_conll"):
        if os.path.isfile(fn) and fn.endswith(tail):
            return self.read_file(fn)

        if os.path.isdir(fn):
            rs = Text()
            for file_name in os.listdir(fn):

                if file_name is not None:
                    rs.extend(self.read(os.path.join(fn, file_name)))
            return rs
        if isinstance(fn, list):
            rs = Text()
            for source in fn:
                rs.extend(self.read(source))
            return rs
        return Text()



if __name__=="__main__":
    import sys
    data_folder = sys.argv[1]
    output = sys.argv[2]
    reader=CoNLL2012Reader(data_folder)
    txt = reader.read_all()
    write_short_conll2005_format(reader.read_all(), output)