__author__ = "Quynh Do"
__copyright__ = "Copyright 2017, DAME"


class Text(list):
    def __init__(self, sens=None):  # value should be  a list of Sentences
        list.__init__(self)
        if sens is not None:
            assert isinstance(sens, list)
            for s in sens:
                self.append(s)

    def __str__(self):
        return " ".join([str(s) for s in self])
