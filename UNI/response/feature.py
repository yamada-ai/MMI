from logging import fatal
import os, sys, pprint, random
import numpy as np
from spacy.util import normalize_slice
sys.path.append('../')
from datatools.analyzer import *

class Feature:
    def __init__(self) -> None:
        # self.pre = preprocess.Preprocessor()

        # 用いる素性テンプレート(関数名)
        self.feature_types = [
            # self.f_is_contain_independent,
            self.f_contain_keyword, 
            self.f_pos_order, 
            self.f_normalize_Noun_order,
            # self.f_order,
            # self.f_normalize_independent_order,
            self.f_normalize_Noun_Verb_order
            
        ]
        self.features = {}
        self._init_feature_functions()

        # 句点を50%で削除する
        # self.delete_func = lambda L: L[:-1] if (L[-1] == "補助記号-句点" and random.random() > 0.5) else L
        # FOS, EOS で囲む
        # self.filler_func = lambda L: ["FOS", *L, "EOS"]

        # キーワード
        self.pro_keyword = "どうですか　どう　どうか　の？　みたら　うか？".split()
        self.wh_keyword = "だれ　誰　どこ　何　いつ　どれくらい　どんな　なぜ　どうやって".split()
        self.f_keyword = self.pro_keyword + self.wh_keyword


    def fill_SYMBOL(self, L):
        return ["FOS", *L, "EOS"]

    # def set_preprocessor(self, pre):
    #     self.pre = pre
    
    def _init_feature_functions(self):
        for type_ in self.feature_types:
            type_name = type_.__name__
            self.features[type_name] = set()

    def make_features(self, sentences_, len_=4):
        sentences = [clean_text(sent) for sent in sentences_]
        self.feature_number_dict = {}
        for feature_func in self.feature_types:
            type_name = feature_func.__name__
            if "order" in type_name:
                for i in range(2, len_+1):
                    features = feature_func(sentences, int(i))
                    self.features[type_name].update(features)
            else:
                if "keyword"  in type_name or "contain_independent" in type_name:
                    features = feature_func(sentences, len_, make=True)
                else:
                    features = feature_func(sentences, len_)
                    self.features[type_name].update(features)
                self.features[type_name].update(features)
            # for sentence in sentence_list:
        
        self.numbering_features()
                

    def numbering_features(self):
        
        for feature_func in self.feature_types:
            type_name = feature_func.__name__
            for f in self.features[type_name]:
                if f not in self.feature_number_dict.keys():
                    self.feature_number_dict[f] = len(self.feature_number_dict)
        self.feature_num = len(self.feature_number_dict)


    def featurization(self, sentences_, len_=4):
        if isinstance(sentences_, str):
            sentences_ = [sentences_]
        # print("feature num: ", self.feature_num)
        sentences = [clean_text(sent) for sent in sentences_]
        x = np.zeros( self.feature_num )
        for feature_func in self.feature_types:
            type_name = feature_func.__name__
            if "order" in type_name:
                for i in range(2, len_+1):
                    # feature_func : 素性関数(ex. f_pos_order)
                    features = feature_func(sentences, int(i))
                    for f in features:
                        if f in self.feature_number_dict.keys():
                            x[self.feature_number_dict[f]] = 1
            else:
                # self.features[type_name].update(features)
                features = feature_func(sentences, len_)
                for f in features:
                    if f in self.feature_number_dict.keys():
                        x[self.feature_number_dict[f]] = 1
        return x

    # 品詞のn-gram
    def f_pos_order(self, sentences, len_=3):
        # 素性タイプ名
        type_ = self.f_pos_order.__name__

        # pos = self.pre.get_POS(sentences)
        # 句読点の句点を50%で削除
        # random_deleted = map(self.delete_func, pos)
        # filled = map(self.filler_func, random_deleted)
        filled = fill_SYMBOL(sentence2pos(sentences))
        
        feature_set = set()
        for L in filled:
            for i in range(len(L)-len_+1):
                f = "_".join(L[i:i+len_])
                # self.features[type_].add(f)
                feature_set.add(f)
        return feature_set
    
    # 名詞の正規化
    def f_normalize_Noun_order(self, sentences, len_=3):
        type_ = self.f_normalize_Noun_order.__name__

        # normal = self.pre.noun2normal(sentences)
        # random_deleted = map(self.delete_func, normal)
        # filled = map(self.filler_func, random_deleted)
        # filled = self.fill_SYMBOL(normal)
        filled = fill_SYMBOL(sentence2normalize_noun(sentences))

        feature_set = set()
        for L in filled:
            for i in range(len(L)-len_+1):
                f = "_".join(L[i:i+len_])
                # self.features[type_].add(f)
                feature_set.add(f)
        return feature_set

    # 普通の n-gram
    def f_order(self, sentences, len_):
        type_ = self.f_order.__name__

        # doc = self.pre.nlp(senteces)
        # token_lem = self.pre.get_lemma(sentences)

        # random_deleted = map(self.delete_func, token_lem)
        # filled = map(self.filler_func, random_deleted)
        # filled = self.fill_SYMBOL(token_lem)

        feature_set = set()
        # for L in filled:
        #     for i in range(len(L)-len_+1):
        #         f = "_".join(L[i:i+len_])
        #         # self.features[type_].add(f)
        #         feature_set.add(f)
        return feature_set

    def f_normalize_independent_order(self, sentences, len_=3):
        type_ = self.f_normalize_Noun_order.__name__

        # normal = self.pre.independent2normal(sentences)
        # random_deleted = map(self.delete_func, normal)
        # filled = map(self.filler_func, random_deleted)
        # filled = self.fill_SYMBOL(normal)
        filled = fill_SYMBOL(sentence2normalize_independent(sentences))

        feature_set = set()
        for L in filled:
            for i in range(len(L)-len_+1):
                f = "_".join(L[i:i+len_])
                # self.features[type_].add(f)
                feature_set.add(f)
        return feature_set
    
    def f_normalize_Noun_Verb_order(self, sentences, len_=3):
        type_ = self.f_normalize_Noun_order.__name__

        # normal = self.pre.noun_verb_2normal(sentences)
        # random_deleted = map(self.delete_func, normal)
        # filled = map(self.filler_func, random_deleted)
        # filled = self.fill_SYMBOL(normal)
        filled = fill_SYMBOL(sentence2normalize_nv(sentences))

        feature_set = set()
        for L in filled:
            for i in range(len(L)-len_+1):
                f = "_".join(L[i:i+len_])
                # self.features[type_].add(f)
                feature_set.add(f)
        return feature_set

    def f_sentence_len(self, sentences, len_):
        type_ = self.f_sentence_len.__name__
        
        # token_lem = self.pre.get_lemma(sentences)

        feature_set = set()
        # for L in token_lem:
        #     f = "len:{0}".format(len(L))
        #     feature_set.add(f)
        return feature_set
    
    def f_is_contain_independent(self, sentence:str, len_, make=False):
        type_ = self.f_is_contain_independent.__name__

        if make:
            return set(["independet", "not independet"])

        is_contain = is_contain_independent(sentence)

        if is_contain:
            return set(["independet"])
        else:
            return set(["not independet"])

        # return is_contain

    def f_contain_keyword(self, sentences, len_, make=False):
        docs = sentence2docs(sentences)

        feature_set = set()

        if make:
            feature_set.update(self.make_keyword_feature())

        for doc in docs:
            for i, token in enumerate(doc):
                
                # ---質問---

                # ---提案---
    
                # ---依頼---
                # 命令形
                if token.pos_=="VERB" or token.pos_=="AUX" :
                    # 連用形
                    inflection = ginza.inflection(token)
                    if inflection=="":
                        continue
                    conjugation = inflection.split(",")[1].split("-")[0]
                    # 文末付近(?, !を考慮)
                    if conjugation=="命令形" and i>=(len(doc)-5):
                        # print(token)
                        f = "-".join( [token.orth_ for token in doc[i:] ] )
                        feature_set.add(f)
                    
                    elif conjugation=="連用形" and i>=len(doc)-6 and i<len(doc)-1:
                    # if conjugation=="連用形" and i< (len(doc)-1):
                        # print(token)
                        if doc[i+1].orth_=="て": 
                            # print(token, doc[i+1], min(3, len(doc)-1-i))
                            f = "-".join( [token.orth_ for token in doc[i:i+min(3, len(doc)-1-i)] ] )
                            feature_set.add(f)
                        

            
            # 他のキーワードはここでチェック！
            sentence = "".join([token.orth_ for token in doc ])
            for keyword in self.f_keyword:
                if keyword in sentence:
                    feature_set.add(keyword)

            return feature_set

    def make_keyword_feature(self):
        feature_set = set()
        for keyword in self.f_keyword:
            feature_set.add(keyword)
        return feature_set

    def show_features(self):
        print("features num :", self.feature_num)
        for type_ in self.feature_types:
            type_name = type_.__name__
            print("feature type : {0}, nums : {1}".format(type_name, len(self.features[type_name])))
            # pprint.pprint(self.features[type_name])

            for f in self.features[type_name]:
                print("{0} : {1}".format(self.feature_number_dict[f], f))
            print()
    

# if __name__ == '__main__':
#     texts = ['そうですね。最近とても暑いですから。', '休日に行きたいと思いますが，あなたもいかがですか？']
#     texts = ['そうですね。',
#  '最近とても暑いですから。',
#  '休日に行きたいと思います。',
#  'はい。',
#  'あなたは海に行きますか？',
#  '何故ですか？',
#  'そうですか。',
#  '山に行くのはどうでしょうか？',
#  '山はお好きなのですか？',
#  '山のおすすめのスポットはご存知ですか？',
#  'どこに行くといいですか？',
#  '明日はとても暑くなるみたいですね。',
#  '涼しくなってきたら、一緒に山へ行きたいですね。']
    # texts = texts[0]
    # F = Feature()
    # F.set_preprocessor(preprocess.Preprocessor())
    # # pos = features.f_pos_order(texts)
    # # normal = features.f_normalize_noun_order(texts)
    # F.make_features(texts)
    # F.show_features()
