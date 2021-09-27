import os
import sys

import pprint
import random
import numpy as np

import spacy

sys.path.append("../")
from datatools.analyzer import *
from error_tools import *

class LM:
    def __init__(self, n=4) -> None:
        self.n = n
        self.LM_types = [
        ]

if __name__ == '__main__':
    texts = [
 '休日に行きたいと思います。',
 'はい。',]

    # text = texts
    # pos = sentence2pos(text)
    # print(pos)
    # print(fill_SYMBOL(pos))
    sen_id = np.arange(13)
    print(sen_id)
    split_sequence(sen_id)
    # print(make_fake_error(text, mode="--change", split_option="pos"))
