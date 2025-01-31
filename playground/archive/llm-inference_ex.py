import sys
sys.path.append("/projects/bdbl/ssastry/bioclip/src")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
import pandas as pd
from tqdm import tqdm
import numpy as np
import os
import json
import itertools

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# quantization_config = BitsAndBytesConfig(load_in_8bit=True)

model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="cuda:0", torch_dtype=torch.bfloat16)

tokenizer = AutoTokenizer.from_pretrained(model_id)
input_text = ["""Coordinates: (38.61467, -90.18976)

Address: "Chouteau Avenue, Kosciusko, Saint Louis, Missouri, United States"

Nearby Places:
"
0.7 km North-West: Gratiot Tower
0.9 km West: LaSalle Park
1.2 km North: Downtown
1.5 km North: Saint Louis
1.5 km West: Peabody-Darst-Webbe
1.6 km South-West: Kosciusko
2.0 km South-West: Soulard
2.2 km West: Lafayette Square
2.2 km North-West: Downtown West
2.5 km West: McKinley Heights
"

How likely does squirrel exist here (On a Scale from 0.0 to 9.9):"""]
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
output = model.generate(**input_ids, max_new_tokens=100, do_sample=False)
print(tokenizer.decode(output[0], skip_special_tokens=True))
"""
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
quantized_model.resize_token_embeddings(len(tokenizer))
for i, chunk in tqdm(enumerate(chunked(prompts.items(), 64))):
    if i%100 == 0:
        d = {}
    batch = chunk
    prompt = [x[1] for x in batch]
    species = [x[0] for x in batch]
    input_ids = tokenizer(prompt, return_tensors="pt", padding=True).to("cuda")
    output = quantized_model.generate(**input_ids, max_new_tokens=100, do_sample=False)
    for j, o in enumerate(output):
        d[species[j]] = tokenizer.decode(o, skip_special_tokens=True)
    if (i+1)%100 == 0:
        with open(f"/projects/bdbl/ssastry/bioclip/data/TreeOfLife-10M/metadata/shard_{(i+1)//100}.json", "w") as f:
            json.dump(d, f)

with open(f"/projects/bdbl/ssastry/bioclip/data/TreeOfLife-10M/metadata/shard_{(i+1)//100+1}.json", "w") as f:
    json.dump(d, f)
#print(tokenizer.decode(output[0], skip_special_tokens=True))
"""