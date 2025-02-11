
model_id = "meta-llama/Meta-Llama-3.1-8B" # "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

# eButterfly experiment
coordinates_csv = "/data/cher/geollm-bias/presence-absence/data/eButterfly.csv"
species_col = 'scientificName'

prompt_type = "basic" # expert, in-context, with_temporal, with_other_species

# outputs
coordinate_prompt_jsonl = "/data/cher/geollm-bias/presence-absence/output/prompts/eButterfly.jsonl"
llm_sim_output_file = "/data/cher/geollm-bias/presence-absence/output/llm-sim/eButterfly.csv"
output_file_llm_preds = f"/data/cher/geollm-bias/presence-absence/output/expert-incontext-{model_id.split('/')[-1]}.jsonl"


