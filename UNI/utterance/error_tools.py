import sys

from numpy.core.defchararray import add
sys.path.append("../")
from datatools.analyzer import *

import random
import numpy as np

pos_dict = get_all_pos_dict()
pos_all_list = list(pos_dict.keys())

# import spacy

# nlp = spacy.load('ja_ginza')

def doc2number(sen, word_dict, unk="unk"):
    return  [ np.array( word_dict[w] ) if w in word_dict else word_dict[unk] for w in sen]   

def make_fake_error(text:str, mode:str, split_option="pos", vocab_list=None):
    """
    text : str
        エラー元になるテキスト
    mode : str
        エラー作成の方法
        --delete
            必要な形態素が消失
        --change
            文中のある形態素を変更
        --duplicate
            文中のある形態素を重複
    split_option : str
        形態素解析の方法
        pos:
            品詞タグによる解析
        nnv: (normalize noun and verb)
            品詞の名詞と動詞を正規化
    """
    if mode == "--delete":
        if split_option=="pos":
            pos = sum(sentence2pos(text), [] )
            pos.pop(random.randint(0, len(pos)-1))
            return pos
        elif split_option=="nnv":
            nnv = sentence2normalize_nv(text)[0]
            nnv.pop(random.randint(0, len(nnv)-1))
            return nnv

    elif mode == "--change":
        if split_option=="pos":
            pos = sum(sentence2pos(text), [] )
            i = random.randint(0, len(pos)-1)
            prev = pos[i]
            while prev == pos[i]:
                pos[i] = random.choice(vocab_list)
            return pos
        elif split_option=="nnv":
            nnv = sum(sentence2normalize_nv(text), [] )
            i = random.randint(0, len(nnv)-1)
            prev = nnv[i]
            while prev == nnv[i]:
                nnv[i] = random.choice(vocab_list)
            return nnv
    
    elif mode == "--duplicate":
        pos = sentence2pos(text)[0]
        pass
    
    else:
        return None


def split_sequence(sen_id:np.ndarray, seq_len=4):
    splited = []
    targets = []
    sen_id = np.concatenate([sen_id, np.zeros(seq_len//2)])
    for i in range(0, len(sen_id)-seq_len, 2):
        splited.append( sen_id[i:i+seq_len] )
        targets.append( sen_id[i+seq_len] )
        # print(sen_id[i:i+seq_len], sen_id[i+seq_len])
    targets[-1] = 2
    # print(targets)
    return splited, targets
