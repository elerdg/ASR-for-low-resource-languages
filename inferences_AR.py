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
saved_model = AutoModelForCTC.from_pretrained("/data/disk1/data/erodegher/wav2vec2-large-xls-r-300m-arabic-80/checkpoint-10780/", local_files_only = True)
saved_model.to("cuda")
print("loading tokenizer")
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("/data/disk1/data/erodegher/tokenizer_ar/", local_files_only = True)
print("loading processor")
processor = Wav2Vec2Processor.from_pretrained("/data/disk1/data/erodegher/wav2vec2-large-xls-r-300m-arabic-80/checkpoint-10780/", local_files_only=True)

## import test set 
common_voice_test= load_dataset("common_voice", "ar", data_dir="./cv-corpus-6.1-2020-12-11", split="test[:20%]")

## lower and no punctuation
print("preprocess data") 
chars_to_remove_regex = '[\—\,\?\.\!\-\;\:\"\“\%\�\°\(\)\–\…\¿\¡\,\""\‘\”\჻\~\՞\؟\،\,\॥\«\»\„\,\“\”\「\」\‘\’\《\》\[\]\{\}\=\`\_\+\<\>\‹\›\©\®\→\。\、\﹂\﹁\～\﹏\，\【\】\‥\〽\『\』\〝\⟨\⟩\〜\♪\؛\/\\\−\^\'\ʻ\ˆ\´\ʾ\‧\〟\'ً \'ٌ\'ُ\'ِ\'ّ\'ْ]'

def remove_special_characters(batch):
    batch["sentence"] = re.sub(chars_to_remove_regex, ' ', batch["sentence"]).lower()
    return batch
common_voice_test = common_voice_test.map(remove_special_characters)

## downsampling
common_voice_test = common_voice_test.cast_column("audio", Audio(sampling_rate=16_000))

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

common_voice_test = common_voice_test.map(prepare_dataset, remove_columns=common_voice_test.column_names )
common_voice_test= common_voice_test.filter(lambda x : x < 5.0*16000, input_columns=["input_length"])

"""#Loading original Transcriptions"""
print("loading transcriptions")
common_voice_transcription= load_dataset("common_voice", "ar", split="test[:20%]")
common_voice_transcription = common_voice_transcription.map(remove_special_characters)
common_voice_transcription=common_voice_transcription.cast_column("audio", Audio(sampling_rate=16_000))
transcription=[ el for el in common_voice_transcription if len(el["audio"]["array"]) < 5.0*16000]


"""# Evaluation"""
print('evaluation')
d_predictions={}
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
list_cer=[]
list_wer=[]
for i, sentence_ in enumerate(predictions):
    #print(i, sentence_)
    d_predictions[sentence_]= transcription[i]["sentence"]
    print(i, "Sentence: ",  sentence_)
    print("Reference: ",  transcription[i]["sentence"])
    result_cer= cer.compute(predictions=[sentence_], references=[transcription[i]["sentence"]])
    result_wer= wer.compute(predictions=[sentence_], references=[transcription[i]["sentence"]])
    print("CER: ", result_cer, "WER: ", result_wer)

    list_sent.append(sentence_)
    list_ref.append(transcription[i]["sentence"])
    list_cer.append(result_cer)
    list_wer.append(result_wer)

#print(len(list_sent), len(list_ref), len(list_cer))

d= {"predictions":list_sent, "reference":list_ref }
df = pd.DataFrame(d)

d2={"CER score": sum(list_cer)/len(list_cer), "WER score": sum(list_wer)/len(list_wer) }
print(d2)

#df2= pd.DataFrame(d2)
#df.append(df2)

df.to_csv("/data/disk1/data/erodegher/inference_ar.csv")
        
