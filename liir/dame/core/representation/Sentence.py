from  DAMESRL.liir.dame.core.representation.Predicate import Predicate

__author__ = "Quynh Do"
__copyright__ = "Copyright 2017, DAME"


class Sentence(list):
    def __init__(self, words=None):  # value should be  a list of Word
        list.__init__(self)
        if words is not None:
            assert isinstance(words, list)
            for w in words:
                self.append(w)

    def __str__(self):
        return " ".join([w.form for w in self])

    def get_predicates(self):
        return [w for w in self if isinstance(w, Predicate)]
