from pathlib import Path
import json
import pandas as pd
import collections
import numpy as np

import spacy

def read_json(path:str, datalist:list) -> pd.DataFrame:
    cols = ['did', 'tid', 'usr', 'sys', 'ec']
    datalist = ['DCM', 'DIT', 'IRS']
    df = pd.DataFrame(index=[], columns=cols)

    for p in datalist:
        datapath = Path(path + p + '/')
        for file_ in datapath.glob("*.json"):
            # print(file_)
            with open(file_, "r") as f:
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


if __name__ == "__main__":
    # Path to the development data
    path = './error_category_classification/dbdc5_ja_dev_labeled/'
    # Names of the dialogue systems
    datalist = ['DCM', 'DIT', 'IRS']
    # List of error types
    error_types = ['Ignore question', 'Unclear intention', 'Wrong information', 'Topic transition error', 'Lack of information', 
    'Repetition', 'Semantic error', 'Self-contradiction', 'Contradiction', 'Grammatical error', 'Ignore offer', 
    'Ignore proposal', 'Lack of sociality', 'Lack of common sense', 'Uninterpretable', 'Ignore greeting']
    #print('Number of error types:', len(error_types))
    df = read_json(path, datalist)

    print(df)