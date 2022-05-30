# -*- coding: utf-8 -*-
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
from transformers import AutoModelForCTC, Wav2Vec2Processor, Wav2Vec2ProcessorWithLM
from datasets.utils.version import Version
from datasets import load_dataset, load_metric, Audio
import os
import numpy as np
import sys
import warnings

"""import the model, processor, tokenizer"""

print("loading saved model")
saved_model = AutoModelForCTC.from_pretrained("/data/disk1/data/erodegher/wav2vec2-large-xls-r-300m-italian-30/checkpoint-11847/", local_files_only = True)
saved_model.to("cuda")
print("loading tokenizer")
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("/data/disk1/data/erodegher/tokenizer_ita/", local_files_only = True)
print("loading processor")
processor = Wav2Vec2Processor.from_pretrained("/data/disk1/data/erodegher/wav2vec2-large-xls-r-300m-italian-30/checkpoint-11847/", local_files_only=True)

## import test set 
data_test= load_dataset("common_voice", "it", data_dir="./cv-corpus-6.1-2020-12-11", split="test[:10]")

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
common_voice_references = data_test
references=[ el for el in common_voice_references if len(el["audio"]["array"]) < 5.0*16000]


"""# Evaluation"""
print('evaluation')
predictions = []

for el in common_voice_test["input_values"]:
    input_dict = processor(el, return_tensors="pt", padding=True)
    logits= saved_model(input_dict.input_values.cuda()).logits
    print(logits.shape)
    pred_ids = torch.argmax(logits[0], dim=-1)
    #print(pred_ids)
    predicted_sentences = processor.decode(pred_ids)
    predictions.append(predicted_sentences)
    #print(predictions)

"""create vocab from tokenizer"""
#vocab_dict = processor.tokenizer.get_vocab()
#sorted_vocab_dict = {k.lower(): v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}
#print(vocab_dict)
#print(sorted_vocab_dict)

vocab_dict = tokenizer.get_vocab()
sorted_vocab = sorted((value, key) for (key,value) in vocab_dict.items())
vocab = [x[1].replace("|", " ") if x[1] not in tokenizer.all_special_tokens else "_" for x in sorted_vocab]
print(vocab)
#sorted_vocab_dict = [''.join(vocab)]
sorted_vocab_dict = ["abcdefghilmnopqrstuvzxkyèòàóúùìé"]
print(sorted_vocab_dict)


print("""decoder from language model""")
from pyctcdecode import build_ctcdecoder

decoder = build_ctcdecoder(labels=list(sorted_vocab_dict),
                            kenlm_model_path="/data/disk1/data/erodegher/ita.arpa",)


from transformers import Wav2Vec2ProcessorWithLM

processor_with_lm = Wav2Vec2ProcessorWithLM(
    feature_extractor=processor.feature_extractor,
    tokenizer=processor.tokenizer,
    decoder=decoder
)

#processor = Wav2Vec2ProcessorWithLM.from_pretrained("/data/disk1/data/erodegher/wav2vec2-base-100h-with-lm")
transcription = processor_with_lm.batch_decode(logits.numpy()).text
transcription[0].lower()

print("LM Prediction: ", transcription)
#print("Reference: ", test_dataset[:10]["sentence"])

sys.exit()

"""compute CER WER"""
list_sent=[]
list_ref=[]

for i, sentence_ in enumerate(transcription):
    #text = [lm_postprocess(x) for x in transcription]
    print("LM Prediction: ", sentence_)
    #print(i, "Sentence: ",  sentence_)
    print("Reference: ",  references[i]["sentence"])

    list_sent.append(sentence_)
    list_ref.append(references[i]["sentence"])
    result_cer= cer.compute(predictions=[sentence_], references=[references[i]["sentence"]])
    result_wer= wer.compute(predictions=[sentence_], references=[references[i]["sentence"]])

"""Write inference file """

#d={ "predictions":list_sent, "reference":list_ref }
#df = pd.DataFrame(d)
#df.to_csv("/data/disk1/data/erodegher/CSV_ITA_INFERENCES.csv")
