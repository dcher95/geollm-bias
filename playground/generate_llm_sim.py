import json
import re

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sklearn.metrics.pairwise import cosine_similarity

from llm_prompts import coordinate_prompt

def get_output_vector(inputs, model):
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # Use the hidden states of the last layer as the output vector
        last_hidden_state = outputs.hidden_states[-1]
        # Average over the sequence dimension to get a single vector
        # Ensure the tensor is in a compatible data type before performing operations
        last_hidden_state = last_hidden_state.to(torch.float32)
        output_vector = last_hidden_state.mean(dim=1).cpu().numpy()
    return output_vector

def main():

    model_id = "meta-llama/Meta-Llama-3.1-8B"
    coordinate_w_prompts = 'sample_prompts.jsonl'

    species = "Eastern Gray Squirrel"

    output_file = f"./llm-sim/{species}/cos-sim-{model_id.split('/')[-1]}.csv"


    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="cuda:1", 
        torch_dtype=torch.bfloat16
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # get species output vector
    species_prompt = f"You are an expert bioligist. Think about where you would find {species}"
    species_inputs = tokenizer(species_prompt, return_tensors="pt").to(model.device)
    species_vector = get_output_vector(species_inputs, model)

    # Go through coordinate prompts
    results = {}
    with open(coordinate_w_prompts, "r", encoding="utf-8") as file:
        for line in file:
            # TODO: Take out coordinates & add to saved information. Should be available in final file.
            idx = int(line.split('"text": ')[0].split('"index": ')[-1].split(", ")[0])

            # Add coordinate info to prompt
            coordinate_info = line.split('"text": ')[-1].split('<TASK>')[0]

            expert_coordinate_prompt = coordinate_prompt.replace("{NEW_LOCATION}", coordinate_info)

            # Get the coordinate vector
            expert_coordinate_inputs = tokenizer(expert_coordinate_prompt, return_tensors="pt").to(model.device)
            expert_coordinate_vector = get_output_vector(expert_coordinate_inputs, model)

            similarity = cosine_similarity(expert_coordinate_vector, species_vector)[0][0]

            results[idx] = similarity
    
    # Convert results to a DataFrame
    df = pd.DataFrame(list(results.items()), columns=['index', 'similarity'])

    # Normalize the similarity column using MinMaxScaler
    scaler = MinMaxScaler()
    df['normalized_similarity'] = scaler.fit_transform(df[['similarity']])

    df.to_csv(output_file)


if __name__ == "__main__":
    main()