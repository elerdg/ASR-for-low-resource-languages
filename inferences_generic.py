## imports 
import pandas as pd
import re
import torch
import json
from IPython.display import display, HTML
from transformers import Wav2Vec2ForCTC
from transformers import Wav2Vec2CTCTokenizer
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from transformers import AutoModelForCTC, Wav2Vec2Processor
from datasets.utils.version import Version
from datasets import load_dataset, load_metric, Audio
import os
import numpy as np
import sys
import warnings
import argparse
from torch import Tensor

parser = argparse.ArgumentParser(description="Code to fine-tune pre-trained model wav2vec-xls-r with small amount of data")
parser.add_argument('-l', '--lang_code', type=str, help='select the first language from Common Voice', default='it')
parser.add_argument('-tp', '--test_pct', type=int, help='pct train', default=20)
parser.add_argument('-m', '--model_name', type=str, help='select the fine-tuned model', default='wav2vec2-large-xls-r-300m-it-100')
parser.add_argument('-ck', '--checkpoint', type=int, help='select the last checkpoint', default=26685)
parser.add_argument('-t', '--tokenizer_name', type=str, help='select tokenizer', default='it')
parser.add_argument('-cs', '--corpus', type=int, help='version of common voice corpus', default=8)

arg= parser.parse_args()

lang = arg.lang_code
test_pct = arg.test_pct
model_name = arg.model_name
checkpoint = arg.checkpoint
tokenizer_name = arg.tokenizer_name
corpus = arg.corpus

"""import the model, processor, tokenizer"""
print("loading saved model")
saved_model = AutoModelForCTC.from_pretrained(f"/data/disk1/data/erodegher/wav2vec2-large-xls-r-300m-{model_name}/checkpoint-{checkpoint}/", local_files_only = True)
saved_model.to("cuda")
print("loading tokenizer")
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(f"/data/disk1/data/erodegher/tokenizer_{tokenizer_name}/", local_files_only = True)
print("loading processor")
processor = Wav2Vec2Processor.from_pretrained(f"/data/disk1/data/erodegher/wav2vec2-large-xls-r-300m-{model_name}/checkpoint-{checkpoint}/", local_files_only=True)

print("import test set")
if corpus == 6 :
    data_test= load_dataset("common_voice", lang, split=f"test[:{test_pct}%]")
else:
    data_test= load_dataset(f"mozilla-foundation/common_voice_{cs}_0", lang,  split=f"test[:{test_pct}%]" )
    data_test = data_test.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])

print("preprocess data") 
if lang == "ar":
    chars_to_remove_regex = '[\—\,\?\.\!\-\;\:\"\“\%\�\°\(\)\–\…\¿\¡\,\""\‘\”\჻\~\՞\؟\،\,\॥\«\»\„\,\“\”\「\」\‘\’\《\》\[\]\{\}\=\`\_\+\<\>\‹\›\©\®\→\。\、\﹂\﹁\～\﹏\，\【\】\‥\〽\『\』\〝\⟨\⟩\〜\♪\؛\/\\\−\^\'\ʻ\ˆ\´\ʾ\‧\〟\'ً \'ٌ\'ُ\'ِ\'ّ\'ْ]'
else:
    chars_to_remove_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\°\(\)\–\…\\\[\]\«\»\\\/\^\<\>\~\_\-\¿\¡\—]'

def remove_special_characters(batch):
    batch["sentence"] = re.sub(chars_to_remove_regex, '', batch["sentence"]).lower()
    return batch
data_test = data_test.map(remove_special_characters)

def replace_hatted_characters(batch):
    batch["sentence"] = re.sub('[’]', "'", batch["sentence"])
   # batch["sentence"] = re.sub('(ll)', "gl", batch["sentence"])
   # batch["sentence"] = re.sub('[ñ]', "gn", batch["sentence"])
   # batch["sentence"] = re.sub('[ç]', "s", batch["sentence"])
    return batch

data_test= data_test.map(replace_hatted_characters)
data_test = data_test.cast_column("audio", Audio(sampling_rate=16_000))

"""Prepare Dataset"""
print("prepare dataset")
def prepare_dataset(batch):
    audio = batch["audio"]    
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])
    with processor.as_target_processor():
        batch["labels"] = processor(batch["sentence"]).input_ids
    return batch

common_voice_test = data_test.map(prepare_dataset, remove_columns=data_test.column_names , keep_in_memory=True)
common_voice_test= common_voice_test.filter(lambda x : x < 5.0*16000, input_columns=["input_length"])

"""#Loading original Transcriptions"""
print("loading transcriptions")
common_voice_transcription = data_test
transcription=[ el for el in common_voice_transcription if len(el["audio"]["array"]) < 5.0*16000]

"""# Evaluation"""
wer = load_metric("wer")
cer = load_metric("cer")

print('evaluation')
predictions = [ ]
for el in common_voice_test["input_values"]:
    input_dict = processor(el, return_tensors="pt", padding=True)
    logits= saved_model(input_dict.input_values.cuda()).logits
    #print(logits.shape)
    pred_ids = torch.argmax(logits[0], dim=-1)
    #print(pred_ids)
    predicted_sentences = processor.decode(pred_ids)
    predictions.append(predicted_sentences)
    #print(predictions)

list_sent=[]
list_ref=[]
for i, sentence_ in enumerate(predictions):
    #print(i, sentence_)
    print(i, "Sentence: ",  sentence_)
    print("Reference: ",  transcription[i]["sentence"])
    list_sent.append(sentence_)
    list_ref.append(transcription[i]["sentence"])

result_cer= cer.compute(predictions=[" ".join(list_sent)], references=[" ".join(list_ref)] )
print("CER", result_cer)
result_wer= wer.compute(predictions=[list_sent], references=[list_ref])
print("WER: ", result_wer)

print("creating a csv file with prediction and reference sentences")
d={ "predictions":list_sent, "reference":list_ref }
df = pd.DataFrame(d)
df.to_csv(f"/data/disk1/data/erodegher/Inference_{lang}-{model_name}.csv")
        




