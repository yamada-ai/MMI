from pathlib import Path
import json
import pandas as pd
import numpy as np
from numpy.lib.function_base import select
import spacy
import torch
import re

# from transformers import RobertaTokenizer, RobertaForMaskedLM, RobertaConfig
# from transformers import BertTokenizer, BertModel

from sklearn.model_selection import train_test_split
# from sklearn.multioutput import MultiOutputClassifier
# from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics

import torch.nn as nn

# from pyknp import Juman
# from transformers import BertTokenizer, BertForMaskedLM, BertConfig

# from sentence_transformers import SentenceTransformer

class preprocessor:

    def __init__(self) -> None:
        self.nlp = spacy.load('ja_ginza')
        self.model_path = "/home/yamada/Downloads/training_bert_japanese"
        self.model = SentenceTransformer(self.model_path, show_progress_bar=False)

        # 半角全角英数字
        # self.DELETE_PATTERN_1 = re.compile(r'[0-9０-９a-zA-Zａ-ｚＡ-Ｚ]+')
        # 記号
        self.DELETE_PATTERN_2 = re.compile(
            r'[\．_－―─！＠＃＄％＾＆\-‐|\\＊\“（）＿■×+α※÷⇒—●★☆〇◎◆▼◇△□(：〜～＋=)／*&^%$#@!~`){}［］…\[\]\"\'\”\’:;<>?＜＞〔〕〈〉？、。・,\./『』【】「」→←○《》≪≫\n\u3000]+')
        
        self.emb_size = self.get_sentence_vec("emb").shape[0]
        print(self.emb_size)

    def get_sentence_vec(self, sen) -> np.array:
        # sen_ = self.DELETE_PATTERN_1.sub(sen)
        sen_ = self.DELETE_PATTERN_2.sub("", sen)
        sentence_vec = self.nlp(sen_).vector
        # sentence_vec = self.model.encode(sen)[0]
        return sentence_vec

    def read_json(self, path:str, datalist:list) -> pd.DataFrame:
        cols = ['did', 'tid', 'usr', 'sys', 'ec']
        df = pd.DataFrame(index=[], columns=cols)
        # datalist = ['DCM', 'DIT', 'IRS']
        for p in datalist:
            datapath = Path(path + p + '/')
            print(datapath)
            # print(list(datapath.glob("*.json")))
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
                        if t["speaker"] == "S" and t["error_category"] != None:
                            tid = t["turn-index"]
                            sys = t["utterance"]
                            ec = t["error_category"]
                            df = df.append(pd.DataFrame([did, tid, usr, sys, ec], index = cols).T)
        df.reset_index(inplace=True, drop=True)
        return df

    def read_json_with_NoErr(self, path:str, datalist:list) -> pd.DataFrame:
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
    
    def feature_extraction(self, df:pd.DataFrame) -> np.array:
        return np.array([np.concatenate([self.get_sentence_vec(u), self.get_sentence_vec(s)]) for u,s in zip(df.usr, df.sys)])
    
    def feature_extraction_context2(self, df:pd.DataFrame) -> np.array:
        # nlp = spacy.load('ja_ginza')
        feature = []
        did = 0
        for d, u, s, e in zip(df.did, df.usr, df.sys, df.ec):
            if did != d:
                u_prev_vec = self.get_sentence_vec(u)
                s_prev_vec = self.get_sentence_vec(s)
                did = d
                if e[0] != "No-Err":
                    each = np.array(
                        [np.concatenate(
                            [np.zeros(self.emb_size),
                            np.zeros(self.emb_size), 
                            u_prev_vec, 
                            s_prev_vec]
                        )]
                    ) 
                    feature.append(each[0])
            else:
                # エラーである
                if e[0] != "No-Err":
                    u_vec = self.get_sentence_vec(u)
                    s_vec = self.get_sentence_vec(s)
                    each = np.array(
                        [np.concatenate(
                            [u_vec,
                            s_vec, 
                            u_prev_vec, 
                            s_prev_vec]
                        )]
                    )
                    feature.append(each[0])
                    u_prev_vec = u_vec
                    s_prev_vec = s_vec
                # エラーではない
                else:    
                    u_prev_vec = self.get_sentence_vec(u)
                    s_prev_vec = self.get_sentence_vec(s)
        return np.array(feature)
    
    def extract_y(self, df:pd.DataFrame, error_types) -> np.array:
        y = []
        for ec in df.ec:
            if ec[0] == "No-Err":
                continue
            y_each_err = np.zeros(len(error_types))
            for i, err in enumerate( error_types ):
                if err in ec:
                    y_each_err[i] = 1
            y.append(y_each_err)
        return np.array(y)