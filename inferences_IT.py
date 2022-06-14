## imports 
import pandas as pd
from datasets import ClassLabel
import random
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

#"""import the model, processor, tokenizer"""

print("loading saved model")
saved_model = AutoModelForCTC.from_pretrained("/data/disk1/data/erodegher/wav2vec2-large-xls-r-300m-italian-new/checkpoint-/", local_files_only = True)
saved_model.to("cuda")
print("loading tokenizer")
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("/data/disk1/data/erodegher/tokenizer_ita/", local_files_only = True)
print("loading processor")
processor = Wav2Vec2Processor.from_pretrained("/data/disk1/data/erodegher/wav2vec2-large-xls-r-300m-italian-new/checkpoint-/", local_files_only=True)

## import test set 
data_test= load_dataset("common_voice", "it", data_dir="./cv-corpus-6.1-2020-12-11", split="test[:20%]")

## lower and no punctuation
print("preprocess data") 
chars_to_remove_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\°\(\)\–\…\\\[\]\«\»\\\/\^\<\>\~]'
def remove_special_characters(batch):
    batch["sentence"] = re.sub(chars_to_remove_regex, '', batch["sentence"]).lower()
    return batch
data_test = data_test.map(remove_special_characters)
## downsampling
data_test = data_test.cast_column("audio", Audio(sampling_rate=16_000))

## import metrics
wer = load_metric("wer")
cer = load_metric("cer")

# Preprocessing the datasets.
print("Preprocessing Dataset")
def prepare_dataset(batch):
    audio = batch["audio"]    
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])
    with processor.as_target_processor():
        batch["labels"] = processor(batch["sentence"]).input_ids
    return batch

common_voice_test = data_test.map(prepare_dataset, remove_columns=data_test.column_names )
common_voice_test= common_voice_test.filter(lambda x : x < 5.0*16000, input_columns=["input_length"])

"""#Loading original Transcriptions"""
print("loading transcriptions")
#common_voice_transcription= load_dataset("common_voice", "it", split="test[:20%]")
#common_voice_transcription = common_voice_transcription.map(remove_special_characters)
#common_voice_transcription=common_voice_transcription.cast_column("audio", Audio(sampling_rate=16_000))
common_voice_transcription= data_test
transcription=[ el for el in common_voice_transcription if len(el["audio"]["array"]) < 5.0*16000]
   
    
"""WER METRICS"""    
def wer_score(hyp, ref):
    wer_dict={ }
    N = len(hyp)
    M = len(ref)
    L = np.zeros((N,M))
    
    for i in range(0, N):
        for j in range(0, M):
            if min(i,j) == 0:
                L[i,j] = max(i,j)
        else:
            deletion = L[i-1,j] + 1
            insertion = L[i,j-1] + 1
            sub = 1 if hyp[i] != ref[j] else 0
            substitution = L[i-1,j-1] + sub
            L[i,j] = min(deletion, min(insertion, substitution))
            print("{} - {}: del {} ins {} sub {} s {}".format(hyp[i], ref[j], deletion, insertion, substitution, sub))
        
            wer_dict[ f'{hyp[i]} - {ref[j]}'] = [deletion, insertion, substitution, sub] 
            with open("/data/disk1/data/erodegher/WER_ita_score.txt","a") as f:
                f.writelines("{} - {}: del {} ins {} sub {} s {}".format(hyp[i], ref[j], deletion, insertion, substitution, sub))

            json_obj = json.dumps(wer_dict)
            file = open('/data/disk1/data/erodegher/ita_file.json', 'a',encoding="utf-8")
            file.write(json_obj)
            
    return int(L[N-1, M-1])    
    
"""# Evaluation"""
print('evaluation')
predictions = [ ]
for el in common_voice_test["input_values"]:
    warnings.filterwarnings("ignore")
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
    print(wer_score(sentence_, transcription[i]["sentence"]))

result_cer= cer.compute(predictions=[" ".join(list_sent)], references=[" ".join(list_ref)] )
print("CER", result_cer)

result_wer= wer.compute(predictions=[list_sent], references=[list_ref])
print("WER: ", result_wer)

d={ "predictions":list_sent, "reference":list_ref }

df = pd.DataFrame(d)
df.to_csv("/data/disk1/data/erodegher/CSV_ITA_INFERENCES-1.csv")

