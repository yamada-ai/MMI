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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import loss
import torch.optim as optim

# from pyknp import Juman
# from transformers import BertTokenizer, BertForMaskedLM, BertConfig

# from sentence_transformers import SentenceTransformer

class preprocessor:

    def __init__(self) -> None:
        self.nlp = spacy.load('ja_ginza')
        # self.model_path = "/home/yamada/Downloads/training_bert_japanese"
        # self.model = SentenceTransformer(self.model_path, show_progress_bar=False)

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

    def make_error_dict(self, error_types):
        error_dict = {}
        for e in error_types:
            error_dict[e] = len(error_dict)
        return error_dict

    def div_did_error(self, df:pd.DataFrame, error_types) -> np.array:
        # nlp = spacy.load('ja_ginza')
        
        did = df.did[0]
        # 全体
        X_data = []
        y_data = []
        # 各 did 
        sequence_did = []
        y_did = []
        
        # エラーの辞書定義
        error_dict = self.make_error_dict(error_types)
        for d, u, s, e in zip(df.did, df.usr, df.sys, df.ec):
            # did で学習データを分割してみる？
            y_one_conv = np.zeros(len(error_types))
            
            if did != d:
                did = d
                # 登録用データ修正
                sequence_did = np.array(sequence_did)
                y_did = np.array(y_did)

                # training_data.append([sequence_did, y_did])
                X_data.append(sequence_did)
                y_data.append(y_did)
                sequence_did = []
                y_did = []

            for e_ in e:
                y_one_conv[error_dict[e_]] = 1

            sequence_did.append(
                np.concatenate(
                    [self.nlp(u).vector,
                    self.nlp(s).vector]
                )
            )
            y_did.append(y_one_conv)

        sequence_did = np.array(sequence_did)
        y_did = np.array(y_did)
        X_data.append(sequence_did)
        y_data.append(y_did)
        return np.array(X_data), np.array( y_data )


class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, tagset_size):
        # 親クラスのコンストラクタ。決まり文句
        super(LSTMTagger, self).__init__()
        # 隠れ層の次元数。これは好きな値に設定しても行列計算の過程で出力には出てこないので。
        self.hidden_dim = hidden_dim
        # インプットの単語をベクトル化するために使う
        # self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # LSTMの隠れ層。これ１つでOK。超便利。
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        # LSTMの出力を受け取って全結合してsoftmaxに食わせるための１層のネットワーク
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        # softmaxのLog版。dim=0で列、dim=1で行方向を確率変換。
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x, None)
        y = self.hidden2tag(lstm_out)
        y = self.softmax(y)
        return y


if __name__ == "__main__":
    print("start")
    EMBEDDING_DIM = 600
    HIDDEN_DIM = 600
    pre = preprocessor()
    path = './error_category_classification/dbdc5_ja_dev_labeled/'
    datalist = ['DCM']
    # List of error types
    error_types = ['Ignore question', 'Unclear intention', 
            'Wrong information', 'Topic transition error', 
            'Lack of information', 'Repetition', 
            'Semantic error', 'Self-contradiction', 
            'Contradiction', 'Grammatical error', 
            'Ignore offer', 'Ignore proposal', 
            'Lack of sociality', 'Lack of common sense',
            'Uninterpretable', 'Ignore greeting', 
            'No-Err'
            ]
    df = pre.read_json_with_NoErr(path, datalist)
    # df = pre.read_json(path, datalist)
    print(df.shape)

    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(error_types))
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    X_data, y_data = pre.div_did_error(df, error_types)
    print(X_data.shape, y_data.shape)
    # for epoch in range(500):

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.20, random_state=5)

    for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
        for X_t, y_t in zip(X_train, y_train):
            model.zero_grad()
            score = model(X_t)
            loss_ = loss_function(score, y_t)
            loss_.backward()
            optimizer.step()