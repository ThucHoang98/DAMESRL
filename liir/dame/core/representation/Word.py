__author__ = "Quynh Do"
__copyright__ = "Copyright 2017, DAME"


class Word(object):
    def __init__(self, form=None):
        self.form = form

    def is_numeric(self):
        try:
            val = self.form
            val = val.replace(",", "")
            num = float(val)
        except ValueError:
            return False
        return True
