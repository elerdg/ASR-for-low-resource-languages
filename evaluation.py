import torch
import librosa
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

## import the model and processor
print("loading saved model")
model= torch.load("/data/disk1/data/erodegher/wav2vec2-xls-r-ita")  #saved epoch 15
#saved_model = AutoModelForCTC.from_pretrained("/data/disk1/data/erodegher/wav2vec2-large-xls-r-300m-italian-colab")
model.to("cuda")

print("loading processor")
processor= torch.load("/data/disk1/data/erodegher/processor_wav2vec-it")
#processor = Wav2Vec2Processor.from_pretrained("/data/disk1/data/erodegher/wav2vec2-large-xls-r-300m-italian-colab")

## import metrics
wer = load_metric("wer")
cer = load_metric("cer")

## import test set
test_dataset = load_dataset("common_voice", "it", data_dir="./cv-corpus-6.1-2020-12-11", split="test[:10%]")
#test_dataset = common_voice_test.cast_column("audio", Audio(sampling_rate=16_000))

# Preprocessing the datasets.
# We need to read the audio files as arrays
def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = librosa.load(batch["path"], sr=16_000)
    batch["speech"] = speech_array
    batch["sentence"] = batch["sentence"].upper()
    return batch

test_dataset = test_dataset.map(speech_file_to_array_fn)

inputs = processor(test_dataset["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)

with torch.no_grad():
    logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

predicted_ids = torch.argmax(logits, dim=-1)
predicted_sentences = processor.batch_decode(predicted_ids)

for i, predicted_sentence in enumerate(predicted_sentences):
    print("-" * 100)
    print("Reference:", common_voice_test[i]["sentence"])
    print("Prediction:", predicted_sentence)


#print("WER: {:2f}".format(100 * wer.compute(predicted_sentence=result["pred_strings"], references=result["sentence"])))
#print("CER: {:2f}".format(100 * cer.compute(predicted_sentence=result["pred_strings"], references=result["sentence"])))
