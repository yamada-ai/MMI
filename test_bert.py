import torch
from transformers import BertTokenizer, BertForMaskedLM, BertConfig
import numpy as np
from pyknp import Juman

if __name__ == "__main__":
    print("bert test")
    bert_JPN_root = "~/Downloads/Japanese_L-12_H-768_A-12_E-30_BPE/"
    bert_JPN_root = "/home/yamada/Downloads/Japanese_L-12_H-768_A-12_E-30_BPE/"
    
    config = BertConfig.from_json_file(bert_JPN_root+'bert_config.json')
    
    model = BertForMaskedLM.from_pretrained(bert_JPN_root+'pytorch_model.bin', config=config)

    bert_tokenizer = BertTokenizer(bert_JPN_root+'vocab.txt',do_lower_case=False, do_basic_tokenize=False)


    text = "スタビジは＊を発信するサイトです"
    jumanpp = Juman()
    result = jumanpp.analysis(text)
    tokenized_text = [mrph.midasi for mrph in result.mrph_list()]
    print(tokenized_text)


    tokenized_text.insert(0, '[CLS]')
    tokenized_text.append('[SEP]')
    masked_index = 4
    tokenized_text[masked_index] = '[MASK]'
    print(tokenized_text)
    

    tokens = bert_tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([tokens])

    
    model.eval()
    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]
    _,predicted_indexes = torch.topk(predictions[0, masked_index], k=10)
    predicted_tokens = bert_tokenizer.convert_ids_to_tokens(predicted_indexes.tolist())
    print(predicted_tokens)

