## imports 
import pandas as pd
from datasets import ClassLabel
import random
import re
import torch
import librosa
import json
from IPython.display import display, HTML
from transformers import Wav2Vec2ForCTC
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from transformers import AutoModelForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer
from datasets.utils.version import Version
from datasets import load_dataset, load_metric, Audio
import os
import numpy as np
import sys

#"""import the model, processor, tokenizer"""

print("loading saved model")
saved_model = AutoModelForCTC.from_pretrained("/data/disk1/data/erodegher/wav2vec2-large-xls-r-300m-italian-colab/checkpoint-36000/", local_files_only = True)
saved_model.to("cuda")

print("loading tokenizer")
tokenizer =  Wav2Vec2CTCTokenizer.from_pretrained("/data/disk1/data/erodegher/wav2vec2-large-xls-r-300m-italian-colab/checkpoint-36000/", local_files_only = True)

print("loading processor")
processor = Wav2Vec2Processor.from_pretrained("/data/disk1/data/erodegher/wav2vec2-large-xls-r-300m-italian-colab/checkpoint-36000/", local_files_only = True)

## import test set 
common_voice_test= load_dataset("common_voice", "it", data_dir="./cv-corpus-6.1-2020-12-11", split="test[:10%]")
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

common_voice_test = common_voice_test.map(prepare_dataset, remove_columns=common_voice_test.column_names)
common_voice_test= common_voice_test.filter(lambda x : x < 5.0*processor.feature_extractor.sampling_rate, input_columns=["input_length"])

"""# Evaluation"""
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
    
    for i, sentence_ in enumerate(predictions):
        print("Prediction: ",  sentence_)
        #print("Reference", common_voice_test[i]["sentence"])

#print("WER: {:2f}".format(100 * wer.compute(predicted_sentence=sentence_, references=common_voice_test[i]["sentence"])))
#print("CER: {:2f}".format(100 * cer.compute(predicted_sentence=result["pred_strings"], references=result["sentence"])))



