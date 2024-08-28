from  DAMESRL.liir.dame.core.representation.Word import Word

__author__ = "Quynh Do"
__copyright__ = "Copyright 2017, DAME"


class Predicate(Word):
    def __init__(self, w):
        super().__init__()
        self.__dict__ = w.__dict__
        self.arguments = None
        self.lemma = None
