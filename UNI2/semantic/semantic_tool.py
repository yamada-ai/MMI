import json
import re
import neologdn

def clean_text_plain(text):
    text_ = neologdn.normalize(text)
    text_ = re.sub(r'\(.*\)', "", text_)
    text_ = re.sub(r'\d+', "0", text_)
    if "……" in text_:
        text_ = text_.replace("……", "…")
    return text_

def format_EOS(sentence):
    if sentence[-1] in ["。", "?", "!", "．"]:
        return sentence
    else:
         return sentence+"。"

def is_valid_utt(utt:str):
    if "「" in utt or "」" in utt:
        return False
    return True

def format_utt(utt:str):
    utt = utt.replace("-", "")
    return utt


def load_novel(path):
    with open(path, mode="r") as f:
        data = json.load(f)
    utt_list = data["utt"]
    return sorted( [ format_utt(u) for u in utt_list if is_valid_utt(u)] )
