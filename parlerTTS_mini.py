import torch
from datasets import load_dataset
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import numpy as np
import time


# read output PATH
file = open("path/output_path.txt", "r")
path = file.read()
file.close()

# TEXT dataset load : pmt
ds = load_dataset("fka/awesome-chatgpt-prompts")

# ParlerTTS mini model load
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")

# mini model's top 5 speaker, ranked by their average speaker similarity scores + background noise
description = {"m1":"Will's voice. And very noisy audio", "m2":"Eric's voice.", "m3":"Laura's voice.", "m4":"Alisa's voice.", "m5":"Patrick's voice."}

print("up to here completed")

# split into 10s
sampling_rate = model.config.sampling_rate
duration = 10  # seconds
chunking = sampling_rate * duration

start = time.time()
past_time = start
for (notation, speaker) in description.items() :
    now_index = 0
    for index in range(0, len(ds['train'])) :
        prompt = ds['train'][index]['prompt']
        # print(prompt)

        input_ids = tokenizer(speaker, return_tensors="pt").input_ids.to(device)
        prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        # print("\nLet's Generate!!\n")
        generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
        audio_arr = generation.cpu().numpy().squeeze()

        # Split the audio array into 10-second segments and Save
        for i in range(0, len(audio_arr), chunking):
            chunk = audio_arr[i:i+chunking]
            sf.write(path + f"/E03_{notation}_pmt_{now_index:06}.wav", chunk, sampling_rate)
            now_index += 1


        # print(f"{time.time()-start:.2f} sec")
        # check
        if index % 10 == 0 :
            print(f'now : {index}')
            print(f"{time.time()-past_time:.2f} sec") 
            past_time = time.time()