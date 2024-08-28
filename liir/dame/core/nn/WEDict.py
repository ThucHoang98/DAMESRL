import re

import numpy as np

__author__ = "Quynh Do"
__copyright__ = "Copyright 2017, DAME"


class WEDict:

    def __init__(self, full_dict_path=None):
        self.full_dict = {}
        self.we_size = -1
        if full_dict_path is not None:
            f = open(full_dict_path, "r" , encoding='utf8')
            for l in f.readlines(): # read the full dictionary
                tmps = re.compile('\s+', re.UNICODE).split(l.strip())
                if len(tmps) > 1:
                    we = []
                    if self.we_size == -1:
                        self.we_size = len(tmps)-1
                    else:
                        if len(tmps)-1 != self.we_size:
                            continue
                    error = False
                    for i in range(1, len(tmps)):
                        try:
                            we.append(float(tmps[i].strip()))
                        except ValueError:
                            error = True
                    if not error:
                        self.full_dict[tmps[0]]= np.asarray(we)
            f.close()

    def get_we(self, w):
        return self.full_dict[w] if w in self.full_dict.keys() else np.random.normal(0,0.01, self.we_size)
