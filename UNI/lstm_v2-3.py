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
    
    def make_error_dict(self, error_types):
        error_dict = {}
        for e in error_types:
            error_dict[e] = len(error_dict)
        return error_dict
    
    def extract_X_y(self, df:pd.DataFrame, error_types, prev_num) -> np.array:
        # nlp = spacy.load('ja_ginza')
        
        did = df.did[0]
        n = prev_num
        # print(did)
        # 全体
        X_data = []
        y_data = []
        # 各 did 
        sequence_did = []
        y_did = []
        # エラーの辞書定義
        error_dict = self.make_error_dict(error_types)

        # 初期の調整 padding
        for i in range(n-1):
            sequence_did.append(
                    np.concatenate( [np.zeros(300), np.zeros(300)])
                )

        # didごとに返却する？
        # エラーが発生したら、開始からエラーまでの文脈を入力とする(N=5の固定長でも可能)
        # 先にこのベクトル列を作成し，Tensorに変換して， List に保持
        for d, u, s, e in zip(df.did, df.usr, df.sys, df.ec):
            if did != d:
                did = d
                sequence_did = []
                y_did = []
                # 初期の調整 padding
                for i in range(n-1):
                    sequence_did.append(
                            np.concatenate( [np.zeros(self.emb_size), np.zeros(self.emb_size)])
                        )
                # break

            # sequence_did.append([u, s])
            sequence_did.append(
                    np.concatenate(
                        [self.get_sentence_vec(u), self.get_sentence_vec(s)]
                    )
                # [u, s]
            )
            if e[0] == "No-Err":
                continue
            else:
                y_each_error_label = np.zeros(len(error_types))
                for e_ in e:
                    y_each_error_label[error_dict[e_]] = 1
                X_data.append(sequence_did[-n:])
                # y_did = np.array(y_each_error_label)
                y_data.append(y_each_error_label)
        return np.array(X_data), np.array(y_data)


def predict_at_least_oneClass(clf, X, error_types) -> np.array:
    y_pred = clf.predict(X)
    p = clf.predict_proba(X)
    # print(y_pred)
    proba = np.array([[p[c][i][1] if (p[c][i].shape[0]!=1) else 0 
                     for c in range(len(error_types))] for i in range(len(X))])
    # print(proba)
  # replace [] to the highest probability label
    y_pred2 = np.empty((0, len(error_types)), int)
    for y, pr in zip(y_pred, proba):
        if  (sum(y) == 0):
            ans = np.zeros_like(y)
            ans[np.argmax(pr)] = 1
        else:
            ans = y
        y_pred2 = np.append(y_pred2, np.array([ans]), axis=0)
    return y_pred2


class LSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, tagset_size, batch_size):
        super(LSTMClassifier, self).__init__()    
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.softmax = nn.LogSoftmax()
    
    def forward(self, x):
        #embeds.size() = (batch_size × len(sentence) × embedding_dim)
        batch_size, seq_len = x.shape[0], x.shape[1]
        _, hidden_layer = self.lstm(x)
        # print(hidden_layer)
        y = self.hidden2tag(hidden_layer[0].view(batch_size, -1))
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