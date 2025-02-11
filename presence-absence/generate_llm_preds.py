import json
import re

import pandas as pd

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig

from llm_prompts import incontext_prompt

def extract_estimate(text):
    match = re.search(r'\\boxed\{([\d.]+)\}', text)
    return float(match.group(1)) if match else None

def extract_coordinates(text):

    # Regular expression to extract coordinates
    match = re.search(r"Coordinates:\s*\(([-\d.]+),\s*([-\d.]+)\)", text)

    if match:
        lat, lon = float(match.group(1)), float(match.group(2))
        return lat, lon
    else:
        return None, None
    
def llm_inference(tokenizer, model, input_prompt, max_tokens = 300):

    # inference LLM
    input_ids = tokenizer(input_prompt, return_tensors="pt").to("cuda")
    output = model.generate(**input_ids, max_new_tokens=max_tokens, do_sample=False)
    response = (tokenizer.decode(output[0], skip_special_tokens=True)).split(input_prompt)[-1]

    return response

def get_species_prompt(text, species):
    # make it species specific
    amended_prompt = text.replace("TASK", f"How likely are you to find {species} here")

    # add to in-context prompting
    input_prompt = incontext_prompt.replace("{SPECIES}", species)
    input_prompt = input_prompt.replace("{NEW_LOCATION}", amended_prompt)

    return input_prompt


def generate_predictions(coordinate_w_prompts, tokenizer, model, output_file, species):

    with open(coordinate_w_prompts, "r", encoding="utf-8") as file:
        for i, line in enumerate(file):
            idx = int(line.split('"text": ')[0].split('"index": ')[-1].split(", ")[0])
            lat, lon = extract_coordinates(line)

            input_prompt = get_species_prompt(text = line.split('"text": ')[-1], species = species)

            response = llm_inference(tokenizer, model, input_prompt)

            if output_file:
                with open(output_file, "a") as file:
                    file.write(json.dumps({"index": idx, "lat":lat, "lon":lon, "text": response}) + "\n")

def get_numeric_predictions_from_llm_text(llm_text_file, output_file):
    
    results = {}
    with open(llm_text_file, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line.strip()) 

            idx = data["index"]
            lat = data["lat"]
            lon = data["lon"]
            text = data['text']

            number = extract_estimate(text)

            results[idx] = {}
            results[idx]['latitude'] = lat
            results[idx]['longitude'] = lon
            results[idx]['prediction'] = number
              
    
    df = pd.DataFrame.from_dict(results, orient='index').reset_index()
    df.columns = ['index', 'latitude', 'longitude' ,'prediction']
    df.to_csv(output_file, index=False)


def main():

    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    coordinate_w_prompts = '/data/cher/geollm-bias/playground/prompts/st_louis_equal.jsonl'
    coordinate_csv = pd.read_csv('/data/cher/geollm-bias/playground/coordinates/st_louis_equal.csv')
    output_file_llm_preds_text = f"./llm-responses/Eastern Gray Squirrel/STL/expert-incontext-{model_id.split('/')[-1]}.jsonl"
    output_file_llm_preds_num = f"./llm-responses/Eastern Gray Squirrel/STL/expert-incontext-{model_id.split('/')[-1]}-numeric.csv"


    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="cuda:0", torch_dtype=torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # generate llm preds
    generate_predictions(coordinate_w_prompts, tokenizer, model, output_file_llm_preds_text, species)

    # pull out the answers
    get_numeric_predictions_from_llm_text(output_file_llm_preds_text, output_file_llm_preds_num)

if __name__ == "__main__":
    main()
