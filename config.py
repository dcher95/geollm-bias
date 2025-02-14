
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct" 
# meta-llama/Meta-Llama-3.1-8B # "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" # Qwen/Qwen2.5-14B-Instruct-1M
max_tokens = 500

# eButterfly experiment
dataset = "eButterfly"
coordinates_csv = "/data/cher/geollm-bias/data/eButterfly.csv"
species_col = 'scientificName'

# eBird experiment
# coordinates_csv = "/data/cher/geollm-bias/data/eBird/sampled_checklists.csv"
# species_col = 'scientificName' # NEED TO UPDATE!

prompt_types = ["only_coords", "basic", "temporal"] # TODO: with_other_species

# outputs
coordinate_prompt_jsonl = f"/data/cher/geollm-bias/output/prompts/{dataset}.jsonl"
llm_sim_output_file = f"/data/cher/geollm-bias/output/{dataset}/cos-sim/{model_id.split('/')[-1]}.csv"
output_path_llm_preds = f"/data/cher/geollm-bias/output/{dataset}/preds/"

### LLM parameters default -- default (max_tokens = 500)

### LLM parameters 1 - More creative
# do_sample=True 
# temperature=0.7
# top_p = 1.0
# top_k = 0
# repitition_penalty = 1.05
# min_p = 0.05
# diversity_penalty = 0.3

### LLM parameters 2 - More strict
# do_sample=True 
# temperature=0.4
# top_p = 1.0
# top_k = 0
# repitition_penalty = 1.1
# min_p = 0.03
# diversity_penalty = 0.2

### LLM parameters 3 -- Nucleus Sampling
# do_sample=True 
# temperature=0.4
# top_p = 0.4
# top_k = 0
# repitition_penalty = 1.1
# min_p = 0.03
# diversity_penalty = 0.2
