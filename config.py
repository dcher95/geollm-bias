
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct" # meta-llama/Meta-Llama-3.1-8B # "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
max_tokens = 500

# eButterfly experiment
coordinates_csv = "/data/cher/geollm-bias/data/eButterfly.csv"
species_col = 'scientificName'

prompt_type = "basic" # expert, incontext, incontext_expert, temporal_expert, TODO: with_other_species

# outputs
coordinate_prompt_jsonl = "/data/cher/geollm-bias/output/prompts/eButterfly.jsonl"
llm_sim_output_file = f"/data/cher/geollm-bias/output/eButterfly/cos-sim-{prompt_type}-{model_id.split('/')[-1]}.csv"
output_file_llm_preds = f"/data/cher/geollm-bias/output/eButterfly/preds-{prompt_type}-{model_id.split('/')[-1]}.csv"


