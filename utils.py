import re
import torch
import pandas as pd
import json

from sklearn.metrics.pairwise import cosine_similarity

from llm_prompts import (
    incontext_prompt,
    expert_prompting,
    incontext_temporal_prompt
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
    

def get_species_prompt(text, species, date = None, prompt_type = 'basic'):
    # make it species specific
    prompt = text.replace("TASK", f"How likely are you to find {species} here")

    if ('incontext' in prompt_type) and ('temporal' not in prompt_type):
        # add to in-context prompting
        input_prompt = incontext_prompt.replace("{SPECIES}", species)
        prompt = input_prompt.replace("{NEW_LOCATION}", prompt)

    # it is assumed that temporal is being done with incontext. Hard to think through logic of adding everything.
    if 'temporal' in prompt_type:
        # add date to prompt
        input_prompt = incontext_temporal_prompt.replace("{SPECIES}", species)
        prompt = input_prompt.replace("{NEW_LOCATION}", prompt)

    # order matters. Should do last as others change the base prompt.
    if 'expert' in prompt_type:
        # add expert prompting to beginning
        species_expert_prompt = expert_prompting.replace("{SPECIES}", species)
        prompt = species_expert_prompt + prompt

    return prompt
    

## data manipulation
def _jsonl_to_df(jsonl):
    data = []
    with open(jsonl, "r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line.strip()))

    df = pd.DataFrame(data)

    return df

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
    match = re.search(r'\\boxed\{([\d.]+)\}', text)
    return float(match.group(1)) if match else None

def llm_inference(tokenizer, model, input_prompt, max_tokens = 500):

    # inference LLM
    input_ids = tokenizer(input_prompt, return_tensors="pt").to("cuda")
    output = model.generate(**input_ids, max_new_tokens=max_tokens, do_sample=False)
    response = (tokenizer.decode(output[0], skip_special_tokens=True)).split(input_prompt)[-1]

    return response