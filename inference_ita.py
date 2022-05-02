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
#saved_model= torch.load("/data/disk1/data/erodegher/model-wav2vec2-ita")  #saved epoch 50
saved_model = AutoModelForCTC.from_pretrained("/data/disk1/data/erodegher/wav2vec2-large-xls-r-300m-italian-colab/checkpoint-36000/", local_files_only = True)
saved_model.to("cuda")

print("loading tokenizer")
tokenizer =  Wav2Vec2CTCTokenizer.from_pretrained("/data/disk1/data/erodegher/wav2vec2-large-xls-r-300m-italian-colab/checkpoint-36000/", local_files_only = True)

print("loading processor")
#processor= torch.load("/data/disk1/data/erodegher/processor_wav2vec-it")
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


"""# Evaluation"""
print('evaluation')
input_dict = processor(common_voice_test["input_values"], return_tensors="pt", padding=True)
logits = saved_model(input_dict.input_values.cuda()).logits
pred_ids = torch.argmax(logits, dim=-1)
predicted_sentences = processor.decode(pred_ids)

for i, predicted_sentence in enumerate(predicted_sentences):
    print("-" * 100)
    print("Reference:", common_voice_test[i]["sentence"])
    print("Prediction:", predicted_sentence)

print("WER: {:2f}".format(100 * wer.compute(predicted_sentence=result["pred_strings"], references=result["sentence"])))
print("CER: {:2f}".format(100 * cer.compute(predicted_sentence=result["pred_strings"], references=result["sentence"])))



