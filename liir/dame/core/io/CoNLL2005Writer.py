import re
from itertools import zip_longest

from  DAMESRL.liir.dame.core.representation.Predicate import Predicate
from  DAMESRL.liir.dame.core.representation.Text import Text

__author__ = "Quynh Do"
__copyright__ = "Copyright 2017, DAME"


def write_short_conll2005_format(txt, output):
    def write_conll_sentence(sen):
        columns = [[w.form for w in sen], [to_string_word(w) for w in sen]]
        for pred in sen.get_predicates():
            columns.append(
                [to_string_label(pred.arguments[i], pred.arguments[i + 1]) for i in
                 range(len(pred.arguments) - 1)]
                + [to_string_label(pred.arguments[-1], None)])

        columns = [[i for i in element if i is not None] for element in list(zip_longest(*columns))]
        return ["\t\t".join(l) for l in columns]

    f = open(output, "w", encoding="utf-8")
    for sen in txt:
        for l in write_conll_sentence(sen):
            f.write(l)
            f.write("\n")
        f.write("\n")
    f.close()


def write_props(txt, output):
    if isinstance(txt, Text):
        f = open(output, "w", encoding="utf-8")
        for sen in txt:
            for l in write_props_sentence(sen):
                f.write(l)
                f.write("\n")
            f.write("\n")
        f.close()


def to_string_word(w):
    return w.lemma if isinstance(w, Predicate) else "-"


def to_string_label(lbl, lbl_next=None):
    if lbl == "O":
        return "*"
    match = re.match('B-(.+)', lbl)
    if match:
        if lbl_next is None:
            return "(" + match.group(1) + "*)"
        if lbl_next[0] == "I":
            return "(" + match.group(1) + "*"
        else:
            return "(" + match.group(1) + "*)"

    match = re.match('I-(.+)', lbl)
    if match:
        if lbl_next is None:
            return "*)"
        if lbl_next[0] == "B" or lbl_next[0] == "O":
            return "*)"
        if lbl_next[0] == "I":
            return "*"

    return "*"


def write_props_sentence(sen):
    columns = [[to_string_word(w) for w in sen]]
    for pred in sen.get_predicates():
        columns.append(
            [to_string_label(pred.arguments[i], pred.arguments[i + 1]) for i in range(len(pred.arguments) - 1)]
            + [to_string_label(pred.arguments[-1], None)])

    columns = [[i for i in element if i is not None] for element in list(zip_longest(*columns))]
    return ["\t".join(l) for l in columns]
