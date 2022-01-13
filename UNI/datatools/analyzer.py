

from numpy.lib.arraysetops import isin
import spacy
import ginza

import pandas as pd
import numpy as np
import json
from pathlib import Path

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn import metrics

from tqdm import tqdm

nlp = spacy.load('ja_ginza')
import collections

from tqdm import tqdm

import MeCab
from wakame.tokenizer import Tokenizer
from wakame.analyzer import Analyzer
from wakame.charfilter import *
from wakame.tokenfilter import *
# tokenizer_ = Tokenizer(use_neologd=True)

# tokenizer_  = None

# ginza のプリセット
pos_preset = [
    "pad",
    "FOS",
    "EOS",
    
    "名詞-普通名詞-一般",
    "名詞-普通名詞-サ変可能" ,
    "名詞-普通名詞-形状詞可能" ,
    "名詞-普通名詞-サ変形状詞可能" ,
    "名詞-普通名詞-副詞可能",
    "名詞-普通名詞-助数詞可能",
    "名詞-固有名詞-一般",
    "名詞-固有名詞-人名-一般",
    "名詞-固有名詞-人名-姓",
    "名詞-固有名詞-人名-名",
    "名詞-固有名詞-地名-一般",
    "名詞-固有名詞-地名-国",
    "名詞-数詞",
    "名詞-助動詞語幹",
    "代名詞",

    "形状詞-一般",
    "形状詞-タリ",
    "形状詞-助動詞語幹",

    "連体詞",
    "副詞",
    "接続詞",

    "感動詞-一般" ,
    "感動詞-フィラー" ,

    "動詞-一般" ,
    "動詞-非自立可能",

    "形容詞-一般",
    "形容詞-非自立可能",

    "助動詞",

    "助詞-格助詞",
    "助詞-副助詞",
    "助詞-係助詞",
    "助詞-接続助詞",
    "助詞-終助詞",
    "助詞-準体助詞",

    "接頭辞",
    "接尾辞-名詞的-一般",
    "接尾辞-名詞的-サ変可能",
    "接尾辞-名詞的-形状詞可能",
    "接尾辞-名詞的-サ変形状詞可能",
    "接尾辞-名詞的-副詞可能",
    "接尾辞-名詞的-助数詞",
    "接尾辞-形状詞的",
    "接尾辞-動詞的",
    "接尾辞-形容詞的",

    "記号-一般",
    "記号-文字",

    "補助記号-一般",
    "補助記号-句点",
    "補助記号-読点",
    "補助記号-括弧開",
    "補助記号-括弧閉",
    "補助記号-ＡＡ-一般",
    "補助記号-ＡＡ-顔文字",
    "空白",
]

filler_func = lambda L: ["FOS", "FOS",  *L, "EOS", "EOS"]
filler_func_one = lambda L: ["FOS",  *L, "EOS"]
# filler_func_google = lambda L: ["FOS",  *L, "EOS"]
filler_func_sep = lambda L: [*L, "[SEP]"]


independent_set = set("NOUN PROPN VERB ADJ ADV PRON NUM".split())
toyoshima_set = set("NOUN PROPN VERB ADJ".split())

import neologdn
import re
from sudachipy import tokenizer
from sudachipy import dictionary
tokenizer_obj = dictionary.Dictionary().create()
tmode = tokenizer.Tokenizer.SplitMode.C

def clean_text(text):
    text_ = neologdn.normalize(text)
    text_ = re.sub(r'\(.*\)', "", text_)
    text_ = re.sub(r'\d+', "0", text_)
    text_  = "".join( [m.normalized_form() if m.part_of_speech()[0]=="名詞" else m.surface() for m in tokenizer_obj.tokenize(text_, tmode)] )
    if "？？" in text_:
        text_ = text_.replace("？？", "？")
    return text_

def normalized_span(text):
    return [m.normalized_form() for m in tokenizer_obj.tokenize(text, tmode)]

def mecab_tokenize(text):
    return tokenizer_.tokenize(text, wakati=True)

def fill_SYMBOL(L):
    return list(map(filler_func, L))

def fill_SYMBOL_ONE(L):
    return list(map(filler_func_one, L))

def fill_SYMBOL_SEP(L):
    sep = list(map(filler_func_sep, L[:-1]))
    sep.append(L[-1])
    return sep

def get_all_pos_dict():
    return dict( zip(pos_preset, range(len(pos_preset))) )

# 修辞
from tqdm import tqdm
def rhetoricasl_and_words(sen):
    # docs = sentence2docs(sen, sents_span=False)
    rhetoricasl = []
    for s in tqdm( sen ) :
        doc = nlp(s)
        phrases = ginza.bunsetu_phrase_spans(doc)
        phrase_otrh = [ str(p) for p in phrases ]
        rhetoricasl.append( list ( set( phrase_otrh + [token.lemma_ for token in doc] )  ) )
        # rhetoricasl.extend( )
    return rhetoricasl
    # return rhetoricasl

def extract_independet(sen):
    docs = sentence2docs(sen, sents_span=False)
    independent = []
    for doc in docs:
        words = []
        for token in doc:
            if token.pos_ in independent_set:
                    # print(token.lemma_)
                words.append(token.lemma_)
            # else:
            #      words.append(token.orth_)
        independent.append(words)
    return independent

def sentence2docs(sen, sents_span=True):
    if isinstance(sen, str):
        doc = nlp(sen)
        # 普通の処理
        if sents_span:
            texts = [str(s)  for s in doc.sents]
        # 文章で区切らない
        else:
            texts = [sen]
    
    elif isinstance(sen, list):
        texts = []
        if sents_span:
            docs = list(nlp.pipe(sen, disable=['ner']))
            for doc in docs:
                texts.extend( [str(s) for s in doc.sents] )
        # 区切らない
        else:
            texts = sen
    else:
        return None
    
    docs = list(nlp.pipe(texts, disable=['ner']))

    return docs

def sentence2pos(sen, sents_span=True) -> list:
    pos_list = []
    docs = sentence2docs(sen, sents_span)
    for doc in docs:
        pos_list.append([ token.tag_ for token in doc ])    
    return pos_list

def sentence2morpheme(sen, sents_span=True)-> list:
    docs = sentence2docs(sen, sents_span)
    morpheme_list = []
    for doc in docs:
        morpheme = []
        for token in doc:
            morpheme.append(token.orth_)
        morpheme_list.append(morpheme)
    return morpheme_list

def sentence2normalize_nv(sen, sents_span=True) -> list:
    normalize_sen = []
    docs = sentence2docs(sen, sents_span)
    for doc in docs:
        words = []
        for token in doc:
            tag = token.tag_.split("-")[0]
                # print(tag)
            if tag in ["名詞", "動詞"]:
                    # print(token.lemma_)
                words.append(token.tag_)
            else:
                 words.append(token.orth_)
        normalize_sen.append(words)
    return normalize_sen

def sentence2normalize_noun(sen, sents_span=True) -> list:
    normalize_sen = []
    docs = sentence2docs(sen, sents_span)
    for doc in docs:
        words = []
        for token in doc:
            tag = token.tag_.split("-")[0]
                # print(tag)
            if tag in ["名詞"]:
                    # print(token.lemma_)
                words.append(token.tag_)
            else:
                 words.append(token.orth_)
        normalize_sen.append(words)
    return normalize_sen

def sentence2normalize_independent(sen, sents_span=True) -> list:
    normalize_sen = []
    docs = sentence2docs(sen, sents_span)
    for doc in docs:
        words = []
        for token in doc:
            # tag = token.tag_.split("-")[0]
                # print(tag)
            if token.pos_ in independent_set:
                    # print(token.lemma_)
                words.append(token.tag_)
            else:
                 words.append(token.orth_)
        normalize_sen.append(words)
    return normalize_sen              

def is_contain_independent(text:str) -> bool:
    doc = nlp(text)
    for token in doc:
        for token in doc:
            if token.pos_ in independent_set:
                return True
    
    return False


def read_json_with_NoErr(path:str, datalist:list) -> pd.DataFrame:
    cols = ['did', 'tid', 'usr', 'sys', 'ec']
    df = pd.DataFrame(index=[], columns=cols)

    for p in datalist:
        datapath = Path(path + p + '/')
        for file in datapath.glob("*.json"):
                # print(file)
            with open(file, "r") as f:
                json_data = json.load(f)
                did = json_data["dialogue-id"]
                for t in json_data["turns"]:
                    if t["turn-index"] == 0:
                        continue
                    if t["speaker"] == "U":
                        usr = t["utterance"]
                        continue
                    if t["speaker"] == "S" :
                        tid = t["turn-index"]
                        sys = t["utterance"]
                        if t["error_category"]:
                            ec = t["error_category"]
                        else:
                            ec = ["No-Err"]
                        df = df.append(pd.DataFrame([did, tid, usr, sys, ec], index = cols).T)
    df.reset_index(inplace=True, drop=True)
    return df
    

class Utterance:
    utt_level = ["Wrong information", "Semantic error", "Uninterpretable", "Grammatical error"]
    def __init__(self, did, sp, utt, errors, type_) -> None:
        self.did = did
        self.sp = sp
        self.utt = utt
        self.errors = errors
        self.type_ = type_

        self.is_utt_error = False
        for e in Utterance.utt_level:
            if self.is_error_included(e):
                self.is_utt_error = True
    
    def __str__(self):
        return "{0}: {1}".format(self.sp, self.utt)

    def is_system(self):
        return True if self.sp=="S" else False
    
    def is_error_included(self, error):
        # Null 対応
        if not self.errors:
            return False
        if isinstance(error, list):
            is_inclued = False
            for e in error:
                is_inclued = is_inclued or (e in self.errors)
            return is_inclued
        elif isinstance(error, str):
            return error in self.errors
        else:
            return False

    def is_exist_error(self):
        return True if self.errors else False
    
    def is_type_included(self, type_):
        return type_ in self.type_
    
    def is_exist_type(self):
        return True if self.type_ else False
    
    def is_utt_level_error(self):
        return self.is_utt_error
    


def read_conv(path:str, datalist:list):
    convs = []
    for p in datalist:
        datapath = Path(path + p + '/')
        for file_ in datapath.glob("*.json"):
            conv = []
            with open(file_, "r") as f:
                json_data = json.load(f)
                did = json_data["did"]
                for t in json_data["turns"]:
                    sp = t["speaker"]
                    utt = t["utterance"]
                    errors = t["error_category"]
                    type_ = t["type"]
                    one = Utterance(did, sp, utt, errors, type_)
                    conv.append(one)
            convs.append(conv)
    return convs  


def score(test, pred):
    if len(collections.Counter(pred)) <= 2:
        print('confusion matrix = \n', confusion_matrix(y_true=test, y_pred=pred))
        print('accuracy = ', accuracy_score(y_true=test, y_pred=pred))
        print('precision = ', precision_score(y_true=test, y_pred=pred))
        print('recall = ', recall_score(y_true=test, y_pred=pred))
        print('f1 score = ', f1_score(y_true=test, y_pred=pred))
    else:
        print('confusion matrix = \n', confusion_matrix(y_true=test, y_pred=pred))
        print('accuracy = ', accuracy_score(y_true=test, y_pred=pred))



if __name__ == '__main__':
    texts = ['そうですね。',
 '最近とても暑いですから。',
 '休日に行きたいと思います。',
 'はい。',
 'あなたは海に行きますか？',
 '何故ですか？',
 'そうですか。',
 '山に行くのはどうでしょうか？',
 '山はお好きなのですか？',
 '山のおすすめのスポットはご存知ですか？',
 'どこに行くといいですか？',
 '明日はとても暑くなるみたいですね。',
 '涼しくなってきたら、一緒に山へ行きたいですね。']

    pos = sentence2pos(texts)
    print(pos)