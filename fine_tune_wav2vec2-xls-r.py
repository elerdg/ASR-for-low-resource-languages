# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
# %%capture 
# !pip install datasets==2.1 
# !pip install transformers==4.18 
# !pip install huggingface_hub==0.5.1 
# !pip install torchaudio==0.11  
# !pip install librosa 
# !pip install jiwer   
# ! git config --global credential.helper store 
# ! apt install git-lfs

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
import argparse
from torch import Tensor

parser = argparse.ArgumentParser(description="Code to fine-tune pre-trained model wav2vec-xls-r with small amount of data")
parser.add_argument('-l', '--lang_code', type=str, help='select language from Common Voice', default='it')
parser.add_argument('-tp', '--train_pct', type=int, help='pct train set', default=10)
parser.add_argument('-cs', '--corpus', type=int, help='version of common voice corpus', default=8)
parser.add_argument('-t', '--tokenizer_name', type=str, help='select the tokenizer of the language', default='tokenizer_it')

arg= parser.parse_args()

lang = arg.lang_code
train_pct = arg.train_pct
corpus = arg.corpus
tokenizer_name = arg.tokenizer_name

"""load dataset"""
if corpus == 6 : 
    common_voice_train = load_dataset("common_voice", lang , split=f"train[:{train_pct}%]")
    common_voice_train = common_voice_train.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])
    common_voice_test = load_dataset("common_voice", lang, split="test[:10%]")
    common_voice_test = common_voice_test.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])
    common_voice_validation = load_dataset("common_voice", lang, split="validation[:10%]")
    common_voice_validation = common_voice_validation.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])
else: 
    common_voice_train = load_dataset(f"mozilla-foundation/common_voice_{cs}_0", lang , split=f"train[:{train_pct}%]", use_auth_token=True )
    ommon_voice_train = common_voice_train.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])
    common_voice_test = load_dataset(f"mozilla-foundation/common_voice_{cs}_0", lang  , split="test[:10%]")    
    common_voice_test = common_voice_test.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])
    common_voice_validation = load_dataset(f"mozilla-foundation/common_voice_{cs}_0", lang , split="validation[:10%]")
    common_voice_validation = common_voice_validation.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])

"""take only path, audio, sentence """
len_train = len(common_voice_train)
len_test = len(common_voice_test)
len_validation=len(common_voice_validation)

print(f"Audio Files per each set: train: {len_train},    test: {len_test},  validation: {len_validation}")

"""Preprocessing Dataset"""
print("preprocess data")
import re
if lang == "ar":
    chars_to_remove_regex = '[\—\,\?\.\!\-\;\:\"\“\%\�\°\(\)\–\…\¿\¡\,\""\‘\”\჻\~\՞\؟\،\,\॥\«\»\„\,\“\”\「\」\‘\’\《\》\[\]\{\}\=\`\_\+\<\>\‹\›\©\®\→\。\、\﹂\﹁\～\﹏\，\【\】\‥\〽\『\』\〝\⟨\⟩\〜\♪\؛\/\\\−\^\'\ʻ\ˆ\´\ʾ\‧\〟\'ً \'ٌ\'ُ\'ِ\'ّ\'ْ]'
else:
    chars_to_remove_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\°\(\)\–\…\\\[\]\«\»\\\/\^\<\>\~\_\-\¿\¡\—]'

def remove_special_characters(batch):
    batch["sentence"] = re.sub(chars_to_remove_regex, '', batch["sentence"]).lower()
    return batch

common_voice_train = common_voice_train.map(remove_special_characters)
common_voice_test = common_voice_test.map(remove_special_characters)
common_voice_validation = common_voice_validation.map(remove_special_characters)

def replace_characters(batch):
    batch["sentence"] = re.sub('[’]', "'", batch["sentence"])
    return batch
common_voice_train = common_voice_train.map(replace_characters)
common_voice_test = common_voice_test.map(replace_characters)
common_voice_validation = common_voice_validation.map(replace_characters)

def extract_all_chars(batch):
  all_text = " ".join(batch["sentence"])
  vocab = list(set(all_text))
  return {"vocab": [vocab], "all_text": [all_text]}

vocab_train = common_voice_train.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=common_voice_train.column_names)
vocab_test = common_voice_test.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=common_voice_test.column_names)
vocab_validation = common_voice_validation.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=common_voice_test.column_names)

vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))
vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}
print(vocab_dict)             
vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]
vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)
len(vocab_dict)

import json
"""uncomment this part if you want to create a tokenizer of the language from the characters in the transcriptions"""
#print("Creating the tokenizer")
#with open('vocab.json', 'w') as vocab_file:
    #json.dump(vocab_dict, vocab_file)
#tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("./", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
#print("saving tokenizer")
#tokenizer.save_pretrained(f"./wav2vec2-large-xls-r-300m-{lang}")

print("load tokenizer form local folder")
tokenizer= Wav2Vec2CTCTokenizer.from_pretrained(f"./{tokenizer_name}/", local_files_only=True)

"""Feature Extractor"""
from transformers import Wav2Vec2FeatureExtractor
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
"""Processor"""
from transformers import Wav2Vec2Processor
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

"""Downsampling"""
print("downsampling from 48000 to 16000")
common_voice_train = common_voice_train.cast_column("audio", Audio(sampling_rate=16_000))
common_voice_test = common_voice_test.cast_column("audio", Audio(sampling_rate=16_000))
common_voice_validation = common_voice_validation.cast_column("audio", Audio(sampling_rate=16_000))

"""Prepare Dataset"""
print("prepare dataset")
def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])
    with processor.as_target_processor():
        batch["labels"] = processor(batch["sentence"]).input_ids
    return batch

common_voice_train = common_voice_train.map(prepare_dataset, remove_columns=common_voice_train.column_names)
common_voice_test = common_voice_test.map(prepare_dataset, remove_columns=common_voice_test.column_names)
common_voice_validation = common_voice_validation.map(prepare_dataset, remove_columns=common_voice_validation.column_names)

"""filter for the audio length"""
print("keep only audios < 5 seconds length")
max_input_length_in_sec = 5.0
common_voice_train = common_voice_train.filter(lambda x: x < max_input_length_in_sec*processor.feature_extractor.sampling_rate, input_columns=["input_length"])
common_voice_test = common_voice_test.filter(lambda x: x < max_input_length_in_sec*processor.feature_extractor.sampling_rate, input_columns=["input_length"])
common_voice_validation = common_voice_validation.filter(lambda x: x < max_input_length_in_sec*processor.feature_extractor.sampling_rate, input_columns=["input_length"])

common_voice_train[0]["input_length"]

"""see the length filtered"""
def Audio_len_filter(common_voice_set):
    len_t =0
    for el in common_voice_set:
        T= el["input_length"]/16000
        len_t= len_t+T

    return len_t

len_tr_filter = Audio_len_filter(common_voice_train)
len_ts_filter = Audio_len_filter(common_voice_test)
len_val_filter=Audio_len_filter(common_voice_validation)
print("filtered duration in seconds:  Train Set:", len_tr_filter, "Test Set:", len_ts_filter , "Validation Set:", len_val_filter)

import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

"""Data Collator"""
@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

"""Metrics CER WER"""
wer_metric = load_metric("wer")
cer_metric = load_metric("cer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    print("wer:", wer )
    print("cer:", cer)

    return {"wer": wer, 
            "cer": cer,}

from transformers import Wav2Vec2ForCTC
#"""load the pretrained checkpoint of Wav2Vec2-XLS-R-300M.
#   Note: When using this notebook to train XLS-R on another language of Common Voice those hyper-parameter settings might not work very well. 
#   Feel free to adapt those depending on your use case."""
print('loading pretrained model')

model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-xls-r-300m", 
    attention_dropout=0.0,
    hidden_dropout=0.0,
    feat_proj_dropout=0.0,
    mask_time_prob=0.05,
    layerdrop=0.0,
    ctc_loss_reduction="mean", 
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
) 

"""Freeze feauture extractor"""
model.freeze_feature_extractor()

"""Parameters for training"""
from transformers import TrainingArguments
training_args = TrainingArguments(
  output_dir= f"wav2vec2-large-xls-r-300m-{lang}-{train_pct}",
  group_by_length=True,
  per_device_train_batch_size=4,  ##16
  per_device_eval_batch_size=4,  
  gradient_accumulation_steps=2,
  evaluation_strategy="epoch",    ## changed from steps to epoch
  num_train_epochs=30,
  gradient_checkpointing=True,
  fp16=True,
  #save_steps=400,
  #eval_steps=400,
  #logging_steps=400,
  learning_rate=3e-4,
  warmup_steps=500,
  save_total_limit=2,
  save_strategy= "epoch",            
  metric_for_best_model="eval_loss", 
  load_best_model_at_end = True,  
  )


"""Trainer"""
from transformers import Trainer, EarlyStoppingCallback 

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=common_voice_train,
    eval_dataset=common_voice_validation,
    tokenizer=processor.feature_extractor,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

"""Training
if "out-of-memory" error: reduce per_device_train_batch_size to 8 or even less and increase gradient_accumulation."""

print("TRAINING")
trainer.train()
#trainer.train(resume_from_checkpoint = True)   #uncomment to restart the traing from last checkpoint
print("ENDED TRAINING")

print("model and tokenizer have been saved in the output_dir directory")

"""Evaluation"""
print("running evaluation")
input_dict = processor(common_voice_test[0]["input_values"], return_tensors="pt", padding=True)
logits = model(input_dict.input_values.to("cuda")).logits
pred_ids = torch.argmax(logits, dim=-1)[0]
print("PRED_IDS", pred_ids)

print("Prediction:")
print(processor.decode(pred_ids))
