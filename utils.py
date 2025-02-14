import re
import torch
import pandas as pd
import json

from datetime import datetime

from sklearn.metrics.pairwise import cosine_similarity

from llm_prompts import (
    only_coords_prompt,
    basic_prompt,
)

## prompt manipulation
def extract_coordinates(text):
    # Regular expression to extract coordinates
    match = re.search(r"Coordinates:\s*\(([-\d.]+),\s*([-\d.]+)\)", text)

    if match:
        lat, lon = float(match.group(1)), float(match.group(2))
        return lat, lon
    else:
        return None, None
    
def add_date_to_prompt(text, date):
    date = str(date).split('/')[0][:10] 
    date = str(datetime.strptime(date, '%Y-%m-%d'))[:10]  # Get current date in MM/DD/YYYY format
    return re.sub(r'(\n\nAddress:)', f'\n\nDate: {date}\1', text, count=1) 


def get_species_prompt(text, species, date = None, prompt_type = 'basic'):
    # remove <TASK> (On a Scale from 0.0 to 9.9):  from text
    text = re.sub(r'<TASK> \(On a Scale from 0.0 to 9.9\):', '', text)
    
    if prompt_type == 'only_coords':
        coordinates = str(extract_coordinates(text))
        input_prompt = only_coords_prompt.replace("{SPECIES}", species)
        prompt = input_prompt.replace("{COORDINATES}", coordinates)

    if prompt_type == 'basic':
        # add to basic prompting
        input_prompt = basic_prompt.replace("{SPECIES}", species)
        prompt = input_prompt.replace("{NEW_LOCATION}", text)

    # if prompt_type == 'incontext':
    #     # add to in-context prompting
    #     input_prompt = incontext_prompt.replace("{SPECIES}", species)
    #     prompt = input_prompt.replace("{NEW_LOCATION}", text)

    if prompt_type == 'temporal':
        # add date to prompt and instructions
        basic_prompt_w_date = basic_prompt.replace("coordinates,", "coordinates, date,")
        input_prompt = basic_prompt_w_date.replace("{SPECIES}", species)
        prompt = input_prompt.replace("{NEW_LOCATION}", text)
        prompt = add_date_to_prompt(prompt, date)

    return prompt
    

## data manipulation
def _jsonl_to_df(jsonl):
    data = []
    with open(jsonl, "r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line.strip()))

    df = pd.DataFrame(data)

    return df

def flatten_sim_dict(results):
    flattened_data = []

    for i, species_data in results.items():
        for species, values in species_data.items():
            row = {
                "index": i,
                "species": species,
                "latitude": values["latitude"],
                "longitude": values["longitude"],
                "mean": values["mean"],
                "max": values["max"]
            }
            flattened_data.append(row)

    return pd.DataFrame(flattened_data)


def convert_to_serializable(obj):
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict()
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    return obj

## similarity calculations
def get_output_vector(inputs, model):
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # Use the hidden states of the last layer as the output vector
        last_hidden_state = outputs.hidden_states[-1]
        # Average over the sequence dimension to get a single vector
        # Ensure the tensor is in a compatible data type before performing operations
        last_hidden_state = last_hidden_state.to(torch.float32)
        output_vector_mean = last_hidden_state.mean(dim=1).cpu().numpy()
        output_vector_max = last_hidden_state.max(dim=1)[0].cpu().numpy() # get just value. don't need index
    return output_vector_mean, output_vector_max

def get_vector_similarity(species_vector_mean, species_vector_max, expert_coordinate_vector_mean, expert_coordinate_vector_max):

    similarity_mean = cosine_similarity(expert_coordinate_vector_mean, species_vector_mean)[0][0]
    similarity_max = cosine_similarity(expert_coordinate_vector_max, species_vector_max)[0][0]

    return similarity_mean, similarity_max

## llm predictions
def extract_estimate(text):
    
    # List of regex patterns to search for numerical estimates
    patterns = [
        r'\\boxed\{([\d.]+)\}',                # \boxed{4.5}
        r'likelihood of finding .*? is ([\d.]+)',  # likelihood of finding X is 4.5
        r'of finding .*? is ([\d.]+)',        # of finding X is 4.5
        r'([\d.]+) out of 9\.9',              # 4.5 out of 9.9
        r'([\d.]+)/9\.9',                     # 4.5 / 9.9
        r'location is ([\d.]+)',              # location is 4.5
        r'answer is ([\d.]+)',                # the final answer is 4.5
        r'9\.9\) ([\d.]+)',                   # 9.9) 4.5
        r'9\.9\: ([\d.]+)',                   # 9.9: 4.5
        r'9\.9\)\: ([\d.]+)',                 # 9.9): 4.5
        r'\"likelihood\":\s*([\d.]+)'         # { "likelihood": 0.0 }
        r' of ([\d.]+)',                      # of 4.5
    ]
    
    # Pre-compile regex patterns for efficiency
    compiled_patterns = [re.compile(pattern, re.MULTILINE | re.DOTALL) for pattern in patterns]
    
    # Iterate over patterns and return the first match
    for pattern in compiled_patterns:
        match = pattern.search(text)
        if match:
            return float(match.group(1).strip('.'))

    return None  # Return None if no match is found

def batch_llm_inference(tokenizer, model, batch_prompts, batch_metadata, d, max_tokens = 500):
    input_ids = tokenizer(batch_prompts, return_tensors="pt", padding=True).to("cuda")
    output = model.generate(**input_ids, do_sample=False, max_new_tokens=max_tokens)

    for j, (o, prompt) in enumerate(zip(output, batch_prompts)):
        response_text = tokenizer.decode(o, skip_special_tokens=True)

        # Ensure the model-generated response doesn't contain the repeated prompt
        # if response_text.startswith(prompt):
        #     response_text = response_text[len(prompt):].strip()

        batch_metadata[j]["response"] = response_text  # Store response
        batch_metadata[j]["prediction"] = extract_estimate(response_text)  # Extract numeric estimate
        d[batch_metadata[j]["index"]] = batch_metadata[j]  # Use index as key

    # for j, o in enumerate(output):
    #     response_text = tokenizer.decode(o, skip_special_tokens=True).split(batch_prompts)[-1]
    #     batch_metadata[j]["response"] = response_text  # Store response
    #     batch_metadata[j]["prediction"] = extract_estimate(response_text)  # Extract numeric estimate
    #     d[batch_metadata[j]["index"]] = batch_metadata[j]  # Use index as key

    return batch_metadata