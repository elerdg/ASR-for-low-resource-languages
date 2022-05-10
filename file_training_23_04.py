# -*- coding: utf-8 -*-
"""file_per_finetuning.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1xSfgmn0ZSixG-1QmYA4YcGdKpjetM4Gw

#Install datasets and transformers
"""

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

os.environ["WANDB_DISABLED"] = "true"

#parser = argparse.ArgumentParser()
#parser.add_argument("epoch", "output_dir", type=int, str)
#args = parser.parse_args()

common_voice_train = load_dataset("common_voice", "it", split="train[:5%]")
common_voice_test = load_dataset("common_voice", "it", split="test[:10%]") ##NON SCARICARE OGNI VOLTA.
common_voice_validation = load_dataset("common_voice", "it", split="validation[:10%]")

"""the information are about : client id, path, audio file, the transcribed sentence , votes , age, gender , accent, the locale of the speaker, and segment """

print('creating dataframe')
pd.DataFrame(common_voice_train)
pd.DataFrame(common_voice_test)
pd.DataFrame(common_voice_validation)

len_train = len(pd.DataFrame(common_voice_train)["audio"])
len_test = len(pd.DataFrame(common_voice_test)["audio"])
len_validation= len(pd.DataFrame(common_voice_validation)["audio"])

print(f" FILE AUDIO PER DATAFRAME train: {len_train},    test: {len_test},   validation: {len_validation}")

"""take only path, audio, sentence """
common_voice_train = common_voice_train.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])
common_voice_test = common_voice_test.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])
common_voice_validation = common_voice_validation.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])

"""Preprocessing dataset"""
print('preprocessing the dataset')

chars_to_remove_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\°\(\)\–\…\\\[\]\\\/\«\»]'

def remove_special_characters(batch):
    batch["sentence"] = re.sub(chars_to_remove_regex, '', batch["sentence"]).lower()
    return batch

common_voice_train = common_voice_train.map(remove_special_characters)
common_voice_test = common_voice_test.map(remove_special_characters)
common_voice_validation=common_voice_validation.map(remove_special_characters)

#show_random_elements(common_voice_train.remove_columns(["path","audio"]))
#show_random_elements(common_voice_test.remove_columns(["path","audio"]))

def replace_hatted_characters(batch):
    batch["sentence"] = re.sub('[à]', 'a', batch["sentence"])
    batch["sentence"] = re.sub('[ì]', 'i', batch["sentence"])
    batch["sentence"] = re.sub('[ò]', 'o', batch["sentence"])
    batch["sentence"] = re.sub('[ù]', 'u', batch["sentence"])
    batch["sentence"] = re.sub('[é]', 'e', batch["sentence"])
    batch["sentence"] = re.sub('[ó]', 'o', batch["sentence"])
    batch["sentence"] = re.sub('[ú]', 'u', batch["sentence"])
    batch["sentence"] = re.sub('[í]', 'i', batch["sentence"])
    
    return batch

common_voice_train = common_voice_train.map(replace_hatted_characters)
common_voice_test = common_voice_test.map(replace_hatted_characters)
common_voice_validation= common_voice_validation.map(replace_hatted_characters)

def extract_all_chars(batch):
  all_text = " ".join(batch["sentence"])
  vocab = list(set(all_text))
  return {"vocab": [vocab], "all_text": [all_text]}

vocab_train = common_voice_train.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=common_voice_train.column_names)
vocab_test = common_voice_test.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=common_voice_test.column_names)

vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))
vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}
print(vocab_dict)

"""for character not present in the test"""
vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]

"""### Vocab of Characters"""
vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)
len(vocab_dict)

import json
with open('vocab.json', 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)
 

"""## Tokenizer"""
print("Tokenizer")
from transformers import Wav2Vec2CTCTokenizer
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("./", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
repo_name = "wav2vec2-large-xls-r-300m-italian-colab"
print("saving tokenizer")
tokenizer.save_pretrained("./wav2vec2-large-xls-r-300m-italian-colab")

"""## FeatureExtractor"""
from transformers import Wav2Vec2FeatureExtractor
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)

"""## Processor = feature extractor + tokenizer"""
from transformers import Wav2Vec2Processor, Wav2Vec2CTCTokenizer
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

print("## Check and resampling")
#common_voice_test[0]['audio']   #sr = 48000
common_voice_train = common_voice_train.cast_column("audio", Audio(sampling_rate=16_000))
common_voice_test = common_voice_test.cast_column("audio", Audio(sampling_rate=16_000))
common_voice_validation = common_voice_validation.cast_column("audio", Audio(sampling_rate=16_000))

#common_voice_test[0]['audio'] #sr = 16000

print("## Prepare Dataset")
def prepare_dataset(batch):
    audio = batch["audio"]
    # batched output is "un-batched"
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])
    with processor.as_target_processor():
        batch["labels"] = processor(batch["sentence"]).input_ids
    return batch

common_voice_train = common_voice_train.map(prepare_dataset, remove_columns=common_voice_train.column_names)
common_voice_test = common_voice_test.map(prepare_dataset, remove_columns=common_voice_test.column_names)
common_voice_validation = common_voice_validation.map(prepare_dataset, remove_columns=common_voice_validation.column_names)
"""#take only first 5 seconds = 89000 number of samplings"""
common_voice_train = common_voice_train.filter(lambda x: x < 5.5* processor.feature_extractor.sampling_rate, input_columns=["input_length"])
common_voice_test = common_voice_test.filter(lambda x: x < 5.5* processor.feature_extractor.sampling_rate, input_columns=["input_length"])
common_voice_validation = common_voice_validation.filter(lambda x: x < 5.5* processor.feature_extractor.sampling_rate, input_columns=["input_length"])

"""## Data Collator """
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

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


"""## Cer Metric"""
cer_metric = load_metric("cer")
wer_metric = load_metric("wer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    print("wer", wer)
    print("cer", cer)
    
    return {"cer": cer, 
           "wer": wer,
           }

"""# load the pretrained checkpoint of Wav2Vec2-XLS-R-300M"""

print('loading pretrained model')
from transformers import Wav2Vec2ForCTC

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
).cuda()

model.freeze_feature_extractor()

from transformers import TrainingArguments
training_args = TrainingArguments(
  output_dir="wav2vec2-large-xls-r-300m-italian-colab",
  group_by_length=True,
  per_device_train_batch_size=4,
  per_device_eval_batch_size=4,
  gradient_accumulation_steps=2,
  evaluation_strategy="epoch",
  num_train_epochs=15,
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

from transformers import Trainer, EarlyStoppingCallback

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=common_voice_train, 
    eval_dataset=common_voice_validation,
    tokenizer=processor.feature_extractor,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
)


#"""# Training """
print("TRAINING")
trainer.train()
#trainer.train(resume_from_checkpoint = True)
print("ENDED TRAINING")

"""#MODEL and TOKENIZER have been saved in the output_dir directory"""
print("model and tokenizer have been saved in the output_dir directory")
#trainer.push_to_hub("wav2vec2-large-xls-r-300m-italian-colab")

"""# Evaluation"""
print('evaluation')
#model = Wav2Vec2ForCTC.from_pretrained(repo_name).cuda()
#processor = Wav2Vec2Processor.from_pretrained(repo_name)

input_dict = processor(common_voice_test[0]["input_values"], return_tensors="pt", padding=True)
logits = model(input_dict.input_values.cuda()).logits
pred_ids = torch.argmax(logits, dim=-1)[0]
print("PRED_IDS", pred_ids)
print("Prediction:")
print(processor.decode(pred_ids))

common_voice_test_transcription = load_dataset("common_voice", "it", data_dir="./cv-corpus-6.1-2020-12-11", split="test[:10%]")

print("\nReference:")
print(common_voice_test_transcription[0]["sentence"].lower())
