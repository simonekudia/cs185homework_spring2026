# Commands for Part 2 Experiments GOPO and Reward Model Ensemble

External LLM Judge Win Rates recorded in eval_log.txt for some of the experiment runs.

### GRPO with GOPO style advantages
```
uv run modal run --detach scripts/modal_train.py::rm_grpo_train_remote -- \
  --algo grpo \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
  --train_split train_gen \
  --eval_split test_gen \
  --reward_model_name Qwen/Qwen2.5-1.5B-Instruct \
  --reward_adapter_path /vol/runs/wildchat_min4_judged_5k_reward_model_v1/checkpoints/step_000445/adapter \
  --output_dir /vol/runs/wildchat_min4_judged_5k_grpo_gopo_rm445_v1 \
  --steps 25 \
  --batch_size 16 \
  --group_size 4 \
  --min_new_tokens 32 \
  --max_new_tokens 256 \
  --temperature 0.8 \
  --top_p 0.95 \
  --lr 1e-5 \
  --grad_accum_steps 2 \
  --ppo_epochs 2 \
  --minibatch_size 8 \
  --clip_eps 0.2 \
  --kl_coef 0.01 \
  --max_prompt_tokens 700 \
  --max_response_tokens 512 \
  --eval_limit 32 \
  --eval_interval 25 \
  --save_interval 25 \
  --wandb_enabled \
  --wandb_project llm-rl-final-project \
  --wandb_name wildchat_min4_judged_5k_grpo_gopo_rm445_v1 \
  --advantage_type rank
```
### GSPO with GOPO style advantages

```
uv run modal run --detach scripts/modal_train.py::rm_grpo_train_remote -- \
  --algo gspo \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
  --train_split train_gen \
  --eval_split test_gen \
  --reward_model_name Qwen/Qwen2.5-1.5B-Instruct \
  --reward_adapter_path /vol/runs/wildchat_min4_judged_5k_reward_model_v1/checkpoints/step_000445/adapter \
  --output_dir /vol/runs/wildchat_min4_judged_5k_gspo_gopo_rm445_v1 \
  --steps 25 \
  --batch_size 16 \
  --group_size 4 \
  --min_new_tokens 32 \
  --max_new_tokens 256 \
  --temperature 0.8 \
  --top_p 0.95 \
  --lr 1e-5 \
  --grad_accum_steps 2 \
  --ppo_epochs 2 \
  --minibatch_size 8 \
  --clip_eps 0.2 \
  --kl_coef 0.01 \
  --max_prompt_tokens 700 \
  --max_response_tokens 512 \
  --eval_limit 32 \
  --eval_interval 25 \
  --save_interval 25 \
  --wandb_enabled \
  --wandb_project llm-rl-final-project \
  --wandb_name wildchat_min4_judged_5k_gspo_gopo_rm445_v1 \
  --advantage_type rank
```

### GRPO with Reward Model Ensemble checkpoints 000100 and 000445

```
uv run modal run --detach scripts/modal_train.py::rm_grpo_train_remote -- \
  --algo grpo \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
  --train_split train_gen \
  --eval_split test_gen \
  --reward_model_name Qwen/Qwen2.5-1.5B-Instruct \
  --reward_adapter_path /vol/runs/wildchat_min4_judged_5k_reward_model_v1/checkpoints/step_000445/adapter \
  --reward_adapter_paths /vol/runs/wildchat_min4_judged_5k_reward_model_v1/checkpoints/step_000100/adapter \
  --ensemble_aggregation min \
  --output_dir /vol/runs/wildchat_min4_judged_5k_grpo_rm100_rm445_ensemble_v1 \
  --steps 25 \
  --batch_size 16 \
  --group_size 4 \
  --min_new_tokens 32 \
  --max_new_tokens 256 \
  --temperature 0.8 \
  --top_p 0.95 \
  --lr 1e-5 \
  --grad_accum_steps 2 \
  --ppo_epochs 2 \
  --minibatch_size 8 \
  --clip_eps 0.2 \
  --kl_coef 0.01 \
  --max_prompt_tokens 700 \
  --max_response_tokens 512 \
  --eval_limit 32 \
  --eval_interval 25 \
  --save_interval 25 \
  --wandb_enabled \
  --wandb_project llm-rl-final-project \
  --wandb_name wildchat_min4_judged_5k_grpo_rm100_rm445_ensemble_v1
```

### GSPO with Reward Model Ensemble checkpoints 000100 and 000445, aggregation : min
```
uv run modal run --detach scripts/modal_train.py::rm_grpo_train_remote -- \
  --algo gspo \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
  --train_split train_gen \
  --eval_split test_gen \
  --reward_model_name Qwen/Qwen2.5-1.5B-Instruct \
  --reward_adapter_path /vol/runs/wildchat_min4_judged_5k_reward_model_v1/checkpoints/step_000445/adapter \
  --reward_adapter_paths /vol/runs/wildchat_min4_judged_5k_reward_model_v1/checkpoints/step_000100/adapter \
  --ensemble_aggregation min \
  --output_dir /vol/runs/wildchat_min4_judged_5k_gspo_rm100_rm445_ensemble_min_v1 \
  --steps 25 \
  --batch_size 16 \
  --group_size 4 \
  --min_new_tokens 32 \
  --max_new_tokens 256 \
  --temperature 0.8 \
  --top_p 0.95 \
  --lr 1e-5 \
  --grad_accum_steps 2 \
  --ppo_epochs 2 \
  --minibatch_size 8 \
  --clip_eps 0.2 \
  --kl_coef 0.01 \
  --max_prompt_tokens 700 \
  --max_response_tokens 512 \
  --eval_limit 32 \
  --eval_interval 25 \
  --save_interval 25 \
  --wandb_enabled \
  --wandb_project llm-rl-final-project \
  --wandb_name wildchat_min4_judged_5k_gspo_rm100_rm445_ensemble_min_v1
```

### GSPO and GOPO with Reward Model Ensemble checkpoints 000100 and 000445, aggregation : min
```
uv run modal run --detach scripts/modal_train.py::rm_grpo_train_remote -- \
  --algo gspo \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
  --train_split train_gen \
  --eval_split test_gen \
  --reward_model_name Qwen/Qwen2.5-1.5B-Instruct \
  --reward_adapter_path /vol/runs/wildchat_min4_judged_5k_reward_model_v1/checkpoints/step_000445/adapter \
  --reward_adapter_paths /vol/runs/wildchat_min4_judged_5k_reward_model_v1/checkpoints/step_000100/adapter \
  --ensemble_aggregation min \
  --output_dir /vol/runs/wildchat_min4_judged_5k_gspo_gopo_rm100_rm445_ensemble_min_v1 \
  --steps 25 \
  --batch_size 16 \
  --group_size 4 \
  --min_new_tokens 32 \
  --max_new_tokens 256 \
  --temperature 0.8 \
  --top_p 0.95 \
  --lr 1e-5 \
  --grad_accum_steps 2 \
  --ppo_epochs 2 \
  --minibatch_size 8 \
  --clip_eps 0.2 \
  --kl_coef 0.01 \
  --max_prompt_tokens 700 \
  --max_response_tokens 512 \
  --eval_limit 32 \
  --eval_interval 25 \
  --save_interval 25 \
  --wandb_enabled \
  --wandb_project llm-rl-final-project \
  --wandb_name wildchat_min4_judged_5k_gspo_gopo_rm100_rm445_ensemble_min_v1 \
  --advantage_type rank
```

### GSPO with Reward Model Ensemble checkpoints 000100 and 000445, aggregation : mean
```
uv run modal run --detach scripts/modal_train.py::rm_grpo_train_remote -- \
  --algo gspo \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
  --train_split train_gen \
  --eval_split test_gen \
  --reward_model_name Qwen/Qwen2.5-1.5B-Instruct \
  --reward_adapter_path /vol/runs/wildchat_min4_judged_5k_reward_model_v1/checkpoints/step_000445/adapter \
  --reward_adapter_paths /vol/runs/wildchat_min4_judged_5k_reward_model_v1/checkpoints/step_000100/adapter \
  --ensemble_aggregation mean \
  --output_dir /vol/runs/wildchat_min4_judged_5k_gspo_rm100_rm445_ensemble_mean_v1 \
  --steps 25 \
  --batch_size 16 \
  --group_size 4 \
  --min_new_tokens 32 \
  --max_new_tokens 256 \
  --temperature 0.8 \
  --top_p 0.95 \
  --lr 1e-5 \
  --grad_accum_steps 2 \
  --ppo_epochs 2 \
  --minibatch_size 8 \
  --clip_eps 0.2 \
  --kl_coef 0.01 \
  --max_prompt_tokens 700 \
  --max_response_tokens 512 \
  --eval_limit 32 \
  --eval_interval 25 \
  --save_interval 25 \
  --wandb_enabled \
  --wandb_project llm-rl-final-project \
  --wandb_name wildchat_min4_judged_5k_gspo_rm100_rm445_ensemble_mean_v1
```

### GSPO and GOPO with Reward Model Ensemble checkpoints 000100 and 000445, aggregation : mean
```
uv run modal run --detach scripts/modal_train.py::rm_grpo_train_remote -- \
  --algo gspo \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
  --train_split train_gen \
  --eval_split test_gen \
  --reward_model_name Qwen/Qwen2.5-1.5B-Instruct \
  --reward_adapter_path /vol/runs/wildchat_min4_judged_5k_reward_model_v1/checkpoints/step_000445/adapter \
  --reward_adapter_paths /vol/runs/wildchat_min4_judged_5k_reward_model_v1/checkpoints/step_000100/adapter \
  --ensemble_aggregation mean \
  --output_dir /vol/runs/wildchat_min4_judged_5k_gspo_gopo_rm100_rm445_ensemble_mean_v1 \
  --steps 25 \
  --batch_size 16 \
  --group_size 4 \
  --min_new_tokens 32 \
  --max_new_tokens 256 \
  --temperature 0.8 \
  --top_p 0.95 \
  --lr 1e-5 \
  --grad_accum_steps 2 \
  --ppo_epochs 2 \
  --minibatch_size 8 \
  --clip_eps 0.2 \
  --kl_coef 0.01 \
  --max_prompt_tokens 700 \
  --max_response_tokens 512 \
  --eval_limit 32 \
  --eval_interval 25 \
  --save_interval 25 \
  --wandb_enabled \
  --wandb_project llm-rl-final-project \
  --wandb_name wildchat_min4_judged_5k_gspo_gopo_rm100_rm445_ensemble_mean_v1 \
  --advantage_type rank
```

### GSPO with Reward Model Ensemble checkpoints 000100 and 000445, aggregation : pessimistic
```
uv run modal run --detach scripts/modal_train.py::rm_grpo_train_remote -- \
  --algo gspo \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
  --train_split train_gen \
  --eval_split test_gen \
  --reward_model_name Qwen/Qwen2.5-1.5B-Instruct \
  --reward_adapter_path /vol/runs/wildchat_min4_judged_5k_reward_model_v1/checkpoints/step_000445/adapter \
  --reward_adapter_paths /vol/runs/wildchat_min4_judged_5k_reward_model_v1/checkpoints/step_000100/adapter \
  --ensemble_aggregation pessimistic \
  --output_dir /vol/runs/wildchat_min4_judged_5k_gspo_rm100_rm445_ensemble_pess_v1 \
  --steps 25 \
  --batch_size 16 \
  --group_size 4 \
  --min_new_tokens 32 \
  --max_new_tokens 256 \
  --temperature 0.8 \
  --top_p 0.95 \
  --lr 1e-5 \
  --grad_accum_steps 2 \
  --ppo_epochs 2 \
  --minibatch_size 8 \
  --clip_eps 0.2 \
  --kl_coef 0.01 \
  --max_prompt_tokens 700 \
  --max_response_tokens 512 \
  --eval_limit 32 \
  --eval_interval 25 \
  --save_interval 25 \
  --wandb_enabled \
  --wandb_project llm-rl-final-project \
  --wandb_name wildchat_min4_judged_5k_gspo_rm100_rm445_ensemble_pess_v1
```

### GSPO and GOPO with Reward Model Ensemble checkpoints 000100 and 000445, aggregation : pessimistic
```
uv run modal run --detach scripts/modal_train.py::rm_grpo_train_remote -- \
  --algo gspo \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
  --train_split train_gen \
  --eval_split test_gen \
  --reward_model_name Qwen/Qwen2.5-1.5B-Instruct \
  --reward_adapter_path /vol/runs/wildchat_min4_judged_5k_reward_model_v1/checkpoints/step_000445/adapter \
  --reward_adapter_paths /vol/runs/wildchat_min4_judged_5k_reward_model_v1/checkpoints/step_000100/adapter \
  --ensemble_aggregation pessimistic \
  --output_dir /vol/runs/wildchat_min4_judged_5k_gspo_gopo_rm100_rm445_ensemble_pess_v1 \
  --steps 25 \
  --batch_size 16 \
  --group_size 4 \
  --min_new_tokens 32 \
  --max_new_tokens 256 \
  --temperature 0.8 \
  --top_p 0.95 \
  --lr 1e-5 \
  --grad_accum_steps 2 \
  --ppo_epochs 2 \
  --minibatch_size 8 \
  --clip_eps 0.2 \
  --kl_coef 0.01 \
  --max_prompt_tokens 700 \
  --max_response_tokens 512 \
  --eval_limit 32 \
  --eval_interval 25 \
  --save_interval 25 \
  --wandb_enabled \
  --wandb_project llm-rl-final-project \
  --wandb_name wildchat_min4_judged_5k_gspo_gopo_rm100_rm445_ensemble_pess_v1 \
  --advantage_type rank
```

### Build Policy Submission to submit to external LLM Judge

Replace path with specific run, using step_000025 for all part2 experiments.

```
uv run modal run --detach scripts/modal_train.py::build_policy_submission_remote -- \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_path /vol/runs/<path>/checkpoints/step_000025/adapter \
  --prompts_jsonl /root/project/public_eval/public_test_gen_prompts_128.jsonl \
  --output_jsonl /vol/submissions/<name>.jsonl \
  --max_prompt_tokens 700 \
  --max_new_tokens 256 \
  --temperature 0.0 \
  --top_p 1.0
```

Run Directory names from Modal to replace path with:

```
wildchat_min4_judged_5k_grpo_gopo_rm445_v1	

wildchat_min4_judged_5k_gspo_gopo_rm100_rm445_ensemble_mean_v1	

wildchat_min4_judged_5k_gspo_gopo_rm100_rm445_ensemble_min_v1	

wildchat_min4_judged_5k_gspo_gopo_rm100_rm445_ensemble_pess_v1	

wildchat_min4_judged_5k_gspo_gopo_rm445_v1	

wildchat_min4_judged_5k_gspo_rm100_rm445_ensemble_mean_v1	

wildchat_min4_judged_5k_gspo_rm100_rm445_ensemble_min_v1	

wildchat_min4_judged_5k_gspo_rm100_rm445_ensemble_pess_v1	
```

### Get Part 2 Experiment Submissions From Modal
```
mkdir -p llm_rl_final_proj_experiment_runs

uv run modal volume get llm-rl-final-project-volume /submissions/<name>.jsonl \
  llm_rl_final_proj_experiment_runs/

```

### Run Single Eval to external LLM Judge for Part 2 Online Threshold
```
echo "=== $(date) ===" >> eval_log.txt && uv run python student_autograder/eval_single.py llm_rl_final_proj_public_submission/policy_generations/<run>.jsonl 0.70 | tee -a eval_log.txt

echo "=== $(date) ===" >> eval_log.txt && uv run python student_autograder/eval_single.py llm_rl_final_proj_experiment_runs/policy_generations/<run>.jsonl 0.70 | tee -a eval_log.txt

```

### Get run data to plot for part 2: 
```
modal volume get llm-rl-final-project-volume runs/wildchat_min4_judged_5k_grpo_rm445_v1/metrics.jsonl part2_runs/grpo_rm445_metrics.jsonl

modal volume get llm-rl-final-project-volume runs/wildchat_min4_judged_5k_grpo_gopo_rm445_v1/metrics.jsonl part2_runs/grpo_gopo_rm445_metrics.jsonl

modal volume get llm-rl-final-project-volume runs/wildchat_min4_judged_5k_gspo_rm445_v1/metrics.jsonl part2_runs/gspo_rm_445_metrics.jsonl

modal volume get llm-rl-final-project-volume runs/wildchat_min4_judged_5k_gspo_gopo_rm445_v1/metrics.jsonl part2_runs/gspo_gopo_rm_445_metrics.jsonl

modal volume get llm-rl-final-project-volume runs/wildchat_min4_judged_5k_gspo_rm100_rm445_ensemble_mean_v1/metrics.jsonl part2_runs/gspo_rm100_rm445_ensemble_mean_metrics.jsonl

modal volume get llm-rl-final-project-volume runs/wildchat_min4_judged_5k_gspo_rm100_rm445_ensemble_min_v1/metrics.jsonl part2_runs/gspo_rm100_rm445_ensemble_min_metrics.jsonl

modal volume get llm-rl-final-project-volume runs/wildchat_min4_judged_5k_gspo_rm100_rm445_ensemble_pess_v1/metrics.jsonl part2_runs/gspo_rm100_rm445_ensemble_pess_metrics.jsonl

modal volume get llm-rl-final-project-volume runs/wildchat_min4_judged_5k_gspo_gopo_rm100_rm445_ensemble_mean_v1/metrics.jsonl part2_runs/gspo_gopo_rm100_rm445_ensemble_mean_metrics.jsonl

modal volume get llm-rl-final-project-volume runs/wildchat_min4_judged_5k_gspo_gopo_rm100_rm445_ensemble_min_v1/metrics.jsonl part2_runs/gspo_gopo_rm100_rm445_ensemble_min_metrics.jsonl

modal volume get llm-rl-final-project-volume runs/wildchat_min4_judged_5k_gspo_gopo_rm100_rm445_ensemble_pess_v1/metrics.jsonl part2_runs/gspo_gopo_rm100_rm445_ensemble_pess_metrics.jsonl
```

### Run plot scripts
```
uv run python scripts/part1_plots.py
uv run python scripts/part2_plots.py
```