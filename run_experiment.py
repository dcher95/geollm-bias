import json
import os
import re
from tqdm import tqdm
import pandas as pd
from datetime import datetime

from more_itertools import chunked

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig

from generate_geollm_prompts_with_csv import get_prompts

from utils import (
    _jsonl_to_df, 
    get_vector_similarity, 
    get_output_vector, 
    extract_coordinates,
    get_species_prompt,
    batch_llm_inference,
    extract_estimate,
    flatten_sim_dict,
    convert_to_serializable
)

from llm_prompts import (
    coordinate_prompt, 
    species_prompt
)

from sklearn.preprocessing import MinMaxScaler

import config

# Get the places information for each of the coordinates
def _get_geollm_prompts(coordinates_csv, coordinate_prompt_jsonl):
    df = pd.read_csv(coordinates_csv)
    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        coordinates = list(zip(df['Latitude'], df['Longitude']))
    else:
        raise ValueError("CSV file must contain 'Latitude' and 'Longitude' columns")
    
    get_prompts(coordinates, coordinate_prompt_jsonl)

# Use similarity calculations to get the similarity between the species and the coordinates
def _get_llm_sim(coordinates_csv, species_col, tokenizer, model, coordinate_prompt_jsonl):
    # TODO: add temporal aspect. add other species aspect. Will not change prompt for now.
    
    df = pd.read_csv(coordinates_csv)
    species_list = pd.read_csv(coordinates_csv, usecols = [species_col]).drop_duplicates()[species_col].tolist()
    
    # read coordinate_prompt_jsonl --> dictionary + pass to get vector stuff.
    coordinates_df = _jsonl_to_df(coordinate_prompt_jsonl)

    # get coordinates related to each species
    results = {}
    for species in tqdm(species_list, desc="Processing species"):

        species_prompt_s = species_prompt.replace("{SPECIES}", species) # Need new name for variable if importing from another file
        species_inputs = tokenizer(species_prompt_s, return_tensors="pt").to(model.device)
        species_vector_mean, species_vector_max = get_output_vector(species_inputs, model)

        # get relevant coordinate prompts
        species_index = df[df[species_col] == species].index
        species_coordinates_df = coordinates_df[coordinates_df['index'].isin(species_index)]

        # pass relevant coordinate prompts to model & get vector similarity for each coordinate
        for i, row in species_coordinates_df.iterrows():
            results[i] = {}
            results[i][species] = {}
            coordinate_info = row['text'].split('"text": ')[-1].split('<TASK>')[0]
            lat, lon = extract_coordinates(row['text'])

            expert_coordinate_prompt = coordinate_prompt.replace("{NEW_LOCATION}", coordinate_info)

            expert_coordinate_inputs = tokenizer(expert_coordinate_prompt, return_tensors="pt").to(model.device)
            expert_coordinate_vector_mean, expert_coordinate_vector_max = get_output_vector(expert_coordinate_inputs, model)

            similarity_mean, similarity_max = get_vector_similarity(species_vector_mean, 
                                                                    species_vector_max, 
                                                                    expert_coordinate_vector_mean, 
                                                                    expert_coordinate_vector_max)
            
            results[i][species]['latitude'] = lat
            results[i][species]['longitude'] = lon
            results[i][species]['mean'] = similarity_mean
            results[i][species]['max'] = similarity_max
        
    df = flatten_sim_dict(results)

    # Normalize the similarity columns using MinMaxScaler
    scaler = MinMaxScaler()
    df[['normalized_similarity_mean', 'normalized_similarity_max']] = scaler.fit_transform(df[['mean', 'max']])

    return df

def _get_llm_preds_batch(coordinate_prompt_jsonl, coordinates_csv, tokenizer, model, prompt_type, max_tokens, output_file_llm_preds, save_every = 10, batch_size=32):
    """ Perform LLM inference in properly batched mode with tqdm for tracking progress. """
    
    coordinates_df = _jsonl_to_df(coordinate_prompt_jsonl)
    species_df = pd.read_csv(coordinates_csv)

    # Generate prompts and metadata before batching
    prompts = []
    metadata = []

    for _, row in coordinates_df.iterrows():
        idx = row['index']
        text = row['text']

        species = species_df[species_df.index == idx]['species'][idx]
        date = species_df[species_df.index == idx]['date'][idx]

        lat, lon = extract_coordinates(text)
        input_prompt = get_species_prompt(text=text, 
                                          species=species, 
                                          date = date,
                                          prompt_type=prompt_type)

        prompts.append(input_prompt)
        metadata.append({'species': species, 'index': idx, 'latitude': lat, 'longitude': lon})

    # Run batched inference
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

    results = []
    for i, chunk in tqdm(enumerate(chunked(prompts, batch_size)), desc="Processing species"):
        if i%save_every == 0:
            d = {}

        batch_prompts = list(chunk)
        batch_metadata = metadata[i * batch_size : (i + 1) * batch_size]
        
        batch_metadata = batch_llm_inference(tokenizer, 
                                             model, 
                                             batch_prompts, 
                                             batch_metadata,
                                             d,
                                             max_tokens)

        results.extend(batch_metadata)  # Append batch results
        
        # Save every 10 batches
        if (i+1)%save_every == 0:
            with open((output_file_llm_preds + f"_{(i+1)//save_every}.json"), "w") as f:
                json.dump(convert_to_serializable(d), f, indent=4)

    if len(results) % save_every != 0:  # If the final batch wasn't saved
        with open((output_file_llm_preds + f"_{(i+1)//save_every + 1}.json"), "w") as f:
            json.dump(convert_to_serializable(d), f, indent=4)

def main():

    coordinates_csv = config.coordinates_csv
    coordinate_prompt_jsonl = config.coordinate_prompt_jsonl
    llm_sim_output_file = config.llm_sim_output_file
    species_col = config.species_col
    model_id = config.model_id
    output_path_llm_preds = config.output_path_llm_preds
    prompt_types = config.prompt_types
    max_tokens = config.max_tokens

    # generate geollm coordinate prompts
    if not os.path.exists(coordinate_prompt_jsonl):
        _get_geollm_prompts(coordinates_csv, coordinate_prompt_jsonl)
        
    # load models
    hf_access_token = XXX
               
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda:0", torch_dtype=torch.bfloat16, token=hf_access_token)
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_access_token)

    # generate llm-sim results
    if not os.path.exists(llm_sim_output_file):
        llm_sim_df = _get_llm_sim(coordinates_csv, 
                                species_col, 
                                tokenizer, 
                                model, 
                                coordinate_prompt_jsonl)
        llm_sim_df.to_csv(llm_sim_output_file)

    for prompt_type in prompt_types:
        print(f"Running prompt type: {prompt_type}")
        output_file_llm_preds = f"{output_path_llm_preds}llm_params_default/{prompt_type}/{model_id.split('/')[-1]}" # outputs in shards so no json is necessary

        # do llm_preds output
        if not os.path.exists(f"{output_file_llm_preds}_1.json"):
            # create folder
            os.makedirs(os.path.dirname(output_file_llm_preds), exist_ok=True)
            _get_llm_preds_batch(coordinate_prompt_jsonl, 
                                        coordinates_csv, 
                                        tokenizer, 
                                        model, 
                                        prompt_type,
                                        max_tokens,
                                        output_file_llm_preds)


if __name__ == "__main__":
    main()