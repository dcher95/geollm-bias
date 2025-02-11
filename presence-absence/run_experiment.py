import json
import os
import re

import pandas as pd

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig

from generate_geollm_prompts_with_csv import get_prompts
from generate_llm_sim import get_species_vectors, get_vector_similarity, get_output_vector, extract_coordinates
from generate_llm_preds import get_species_prompt, llm_inference, extract_estimate

from llm_prompts import coordinate_prompt, species_prompt

from sklearn.preprocessing import MinMaxScaler

import config

def _get_geollm_prompts(coordinates_csv, coordinate_prompt_jsonl):
    df = pd.read_csv(coordinates_csv)
    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        coordinates = list(zip(df['Latitude'], df['Longitude']))
    else:
        raise ValueError("CSV file must contain 'Latitude' and 'Longitude' columns")
    
    get_prompts(coordinates, coordinate_prompt_jsonl)

def _jsonl_to_df(jsonl):
    data = []
    with open(jsonl, "r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line.strip()))

    df = pd.DataFrame(data)

    return df

def _get_llm_sim(coordinates_csv, species_col, tokenizer, model, coordinate_prompt_jsonl):
    df = pd.read_csv(coordinates_csv)
    species_list = pd.read_csv(coordinates_csv, usecols = [species_col]).drop_duplicates()[species_col].tolist()
    
    # read coordinate_prompt_jsonl --> dictionary + pass to get vector stuff.
    coordinates_df = _jsonl_to_df(coordinate_prompt_jsonl)

    # get coordinates related to each species
    results = {}
    for species in species_list:
        results[species] = {}

        species_vector_mean, species_vector_max = get_species_vectors(tokenizer, model, species)

        # get relevant coordinate prompts
        species_index = df[df[species_col] == species].index
        species_coordinates_df = coordinates_df[coordinates_df['index'].isin(species_index)]

        # pass relevant coordinate prompts to model & get vector similarity for each coordinate
        for i, row in species_coordinates_df.iterrows():
            results[species][i] = {}
            coordinate_info = row['text'].split('"text": ')[-1].split('<TASK>')[0]
            lat, lon = extract_coordinates(row['text'])

            expert_coordinate_prompt = coordinate_prompt.replace("{NEW_LOCATION}", coordinate_info)

            expert_coordinate_inputs = tokenizer(expert_coordinate_prompt, return_tensors="pt").to(model.device)
            expert_coordinate_vector_mean, expert_coordinate_vector_max = get_output_vector(expert_coordinate_inputs, model)

            similarity_mean, similarity_max = get_vector_similarity(species_vector_mean, 
                                                                    species_vector_max, 
                                                                    expert_coordinate_vector_mean, 
                                                                    expert_coordinate_vector_max)
            
            results[species][i]['latitude'] = lat
            results[species][i]['longitude'] = lon
            results[species][i]['mean'] = similarity_mean
            results[species][i]['max'] = similarity_max
        
    df = pd.DataFrame.from_dict(results, orient='index').reset_index()
    df.columns = ['species','index', 'latitude', 'longitude' ,'similarity_mean', 'similarity_max']

    # Normalize the similarity columns using MinMaxScaler
    scaler = MinMaxScaler()
    df[['normalized_similarity_mean', 'normalized_similarity_max']] = scaler.fit_transform(df[['similarity_mean', 'similarity_max']])

    return df

def _get_llm_preds(coordinate_prompt_jsonl, coordinates_csv, tokenizer, model):

    coordinates_df = _jsonl_to_df(coordinate_prompt_jsonl)
    species_df = pd.read_csv(coordinates_csv)

    results = {}
    for i, row in coordinates_df.iterrows():
        idx = row['index']
        text = row['text']
        species = species_df[species_df[species_df.index] == idx]

        lat, lon = extract_coordinates(text)

        # Basic prompt TODO: make the coding of different prompts modularized
        input_prompt = get_species_prompt(text = text, species = species)
        response = llm_inference(tokenizer, model, input_prompt)
        number = extract_estimate(response)

        results[species] = {}
        results[species][idx]['latitude'] = lat
        results[species][idx]['longitude'] = lon
        results[species][idx]['response'] = response
        results[species][idx]['prediction'] = number

    df = pd.DataFrame.from_dict(results, orient='index').reset_index()
    df.columns = ['species','index', 'latitude', 'longitude' ,'response', 'prediction']

    return df


def main():

    coordinates_csv = config.coordinates_csv
    coordinate_prompt_jsonl = config.coordinate_prompt_jsonl
    llm_sim_output_file = config.llm_sim_output_file
    species_col = config.species_col
    model_id = config.model_id
    output_file_llm_preds = config.output_file_llm_preds

    # generate geollm coordinate prompts
    if not os.path.exists(coordinate_prompt_jsonl):
        _get_geollm_prompts(coordinates_csv, coordinate_prompt_jsonl)
        
    # load models
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda:0", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # generate llm-sim results
    if not os.path.exists(llm_sim_output_file):
        llm_sim_df = _get_llm_sim(coordinates_csv, species_col, tokenizer, model, coordinate_prompt_jsonl)
        llm_sim_df.to_csv(llm_sim_output_file)

    # do llm_preds output
    output_file_llm_preds_num = output_file_llm_preds.replace(".jsonl" ,"-numeric.csv")
    if not os.path.exists(output_file_llm_preds_num):
        llm_preds_df = _get_llm_preds(coordinate_prompt_jsonl, coordinates_csv, tokenizer, model)
        llm_preds_df.to_csv(output_file_llm_preds)


if __name__ == "__main__":
    main()