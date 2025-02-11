import json
import re

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sklearn.metrics.pairwise import cosine_similarity

from llm_prompts import coordinate_prompt, species_prompt

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

def extract_coordinates(text):

    # Regular expression to extract coordinates
    match = re.search(r"Coordinates:\s*\(([-\d.]+),\s*([-\d.]+)\)", text)

    if match:
        lat, lon = float(match.group(1)), float(match.group(2))
        return lat, lon
    else:
        return None, None

def get_species_vectors(tokenizer, model, species):
    species_prompt = species_prompt.replace("{SPECIES}", species)
    species_inputs = tokenizer(species_prompt, return_tensors="pt").to(model.device)
    return get_output_vector(species_inputs, model)

def get_vector_similarity(species_vector_mean, species_vector_max, expert_coordinate_vector_mean, expert_coordinate_vector_max):

    similarity_mean = cosine_similarity(expert_coordinate_vector_mean, species_vector_mean)[0][0]
    similarity_max = cosine_similarity(expert_coordinate_vector_max, species_vector_max)[0][0]

    return similarity_mean, similarity_max

def main():

    model_id = "meta-llama/Meta-Llama-3.1-8B"
    # model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    coordinate_w_prompts = './prompts/st_louis_equal.jsonl'

    species = "Eastern Gray Squirrel"

    output_file = f"./llm-sim/{species}/cos-sim-{model_id.split('/')[-1]}_STL.csv"


    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="cuda:1", 
        torch_dtype=torch.bfloat16
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # get species output vector
    species_vector_mean, species_vector_max = get_species_vectors(tokenizer, model, species)
    
    # TODO: Accept dataframe with coordinates
    # Go through coordinate prompts
    results = {}
    with open(coordinate_w_prompts, "r", encoding="utf-8") as file:
        for line in file:
            results[idx] = {}
            idx = int(line.split('"text": ')[0].split('"index": ')[-1].split(", ")[0])
            lat, lon = extract_coordinates(line)

            # Add coordinate info to prompt
            coordinate_info = line.split('"text": ')[-1].split('<TASK>')[0]

            expert_coordinate_prompt = coordinate_prompt.replace("{NEW_LOCATION}", coordinate_info)

            # Get the coordinate vector
            expert_coordinate_inputs = tokenizer(expert_coordinate_prompt, return_tensors="pt").to(model.device)
            expert_coordinate_vector_mean, expert_coordinate_vector_max = get_output_vector(expert_coordinate_inputs, model)

            similarity_mean, similarity_max = get_vector_similarity(species_vector_mean, species_vector_max, expert_coordinate_vector_mean, expert_coordinate_vector_max)

            results[idx]['latitude'] = lat
            results[idx]['longitude'] = lon
            results[idx]['mean'] = similarity_mean
            results[idx]['max'] = similarity_max

    # Convert nested dictionary to a DataFrame
    df = pd.DataFrame.from_dict(results, orient='index').reset_index()
    df.columns = ['index', 'latitude', 'longitude' ,'similarity_mean', 'similarity_max']

    # Normalize the similarity columns using MinMaxScaler
    scaler = MinMaxScaler()
    df[['normalized_similarity_mean', 'normalized_similarity_max']] = scaler.fit_transform(df[['similarity_mean', 'similarity_max']])

    df.to_csv(output_file)


if __name__ == "__main__":
    main()