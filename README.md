# ASR for Low-resource languages

## Overview
This project aims to perform automatic speech recognition for low-resource languages. To do so we fine-tuned [wav2vec2-xls-r](https://huggingface.co/facebook/wav2vec2-xls-r-300m) [(Babu et al., 2021)](https://arxiv.org/abs/2111.09296) on labeled speech data from different languages from the [Mozilla Common Voice](https://commonvoice.mozilla.org/en/datasets) dataset [(Ardila et al., 2020)](https://github.com/common-voice/cv-dataset).

The scripts in this repo allow to: 
* fine-tune wav2vec2-xls-r on labeled speech data from one language 
* fine-tune wav2vec2-xls-r training jointly two different languages
* test the fine-tuned model 
* test the fine-tuned model incorporating an n-gram language model  