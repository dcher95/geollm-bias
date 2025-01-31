import json
import re

import pandas as pd

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig

from llm_prompts import squirrel_incontext_prompt

def extract_estimate(text):
    match = re.search(r'\\boxed\{([\d.]+)\}', text)
    return float(match.group(1)) if match else None

def generate_predictions(coordinate_w_prompts, tokenizer, model, output_file):

    with open(coordinate_w_prompts, "r", encoding="utf-8") as file:
        for line in file:
            idx = int(line.split('"text": ')[0].split('"index": ')[-1].split(", ")[0])
            species = "Eastern Gray Squirrel"

            # make it species specific
            amended_prompt = line.split('"text": ')[-1].replace("TASK", f"How likely are you to find a {species} here")

            # add to in-context prompting
            input_prompt = squirrel_incontext_prompt.replace("{NEW_LOCATION}", amended_prompt)

            # inference LLM
            input_ids = tokenizer(input_prompt, return_tensors="pt").to("cuda")
            output = model.generate(**input_ids, max_new_tokens=300, do_sample=False)
            response = (tokenizer.decode(output[0], skip_special_tokens=True)).split(input_prompt)[-1]

            if output_file:
                with open(output_file, "a") as file:
                    file.write(json.dumps({"index": idx, "text": response}) + "\n")

def get_numeric_predictions_from_llm_text(llm_text_file, output_file):
    
    results = {}
    with open(llm_text_file, "r", encoding="utf-8") as file:
            for line in file:
                idx = int(line.split('"text": ')[0].split('"index": ')[-1].split(", ")[0])
                text = line.split('"text": ')[-1]

                number = extract_estimate(text)

                results[idx] = number
              
    df = pd.DataFrame(list(results.items()), columns=['index', 'prediction'])
    df.to_csv(output_file, index=False)


def main():

    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    coordinate_w_prompts = 'sample_prompts.jsonl'
    output_file_llm_preds_text = f"./llm-responses/Eastern Gray Squirrel/expert-incontext3a-{model_id.split('/')[-1]}.jsonl"
    output_file_llm_preds_num = f"./llm-responses/Eastern Gray Squirrel/expert-incontext3a-{model_id.split('/')[-1]}-numeric.csv"


    # model = AutoModelForCausalLM.from_pretrained(
    #     model_id, device_map="cuda:0", torch_dtype=torch.bfloat16)

    # tokenizer = AutoTokenizer.from_pretrained(model_id)

    # # generate llm preds
    # generate_predictions(coordinate_w_prompts, tokenizer, model, output_file_llm_preds_text)

    # pull out the answers
    # TODO: Take out coordinates & add to saved information. Should be available in final file.
    get_numeric_predictions_from_llm_text(output_file_llm_preds_text, output_file_llm_preds_num)

if __name__ == "__main__":
    main()
