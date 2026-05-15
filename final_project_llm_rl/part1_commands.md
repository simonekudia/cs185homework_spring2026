# Part 1 Commands

### Reward Model
```
uv run modal run scripts/modal_train.py::reward_model_train_remote -- \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
  --train_split train_prefs \
  --eval_split test_prefs \
  --output_dir /vol/runs/wildchat_min4_judged_5k_reward_model_v1 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --grad_accum_steps 4 \
  --lr 3e-5 \
  --num_train_epochs 3 \
  --max_prompt_tokens 700 \
  --max_response_tokens 512 \
  --eval_interval 25 \
  --save_interval 50 \
  --wandb_enabled \
  --wandb_project llm-rl-final-project \
  --wandb_name wildchat_min4_judged_5k_reward_model_v1
```

### Offline Preference Training

DPO

```
uv run modal run scripts/modal_train.py::train_remote -- \
  --algo dpo \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
  --train_split train_prefs \
  --eval_split test_prefs \
  --generation_split test_gen \
  --output_dir /vol/runs/wildchat_min4_judged_5k_dpo_beta01_v1 \
  --beta 0.1 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --grad_accum_steps 4 \
  --lr 5e-5 \
  --num_train_epochs 3 \
  --max_prompt_tokens 700 \
  --max_response_tokens 512 \
  --generation_eval_limit 32 \
  --generation_eval_max_new_tokens 256 \
  --generation_eval_every 100 \
  --eval_interval 100 \
  --save_interval 100 \
  --wandb_enabled \
  --wandb_project llm-rl-final-project \
  --wandb_name wildchat_min4_judged_5k_dpo_beta01_v1
```

IPO

```
uv run modal run scripts/modal_train.py::train_remote -- \
  --algo ipo \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
  --train_split train_prefs \
  --eval_split test_prefs \
  --generation_split test_gen \
  --output_dir /vol/runs/wildchat_min4_judged_5k_ipo_v1 \
  --beta 0.1 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --grad_accum_steps 4 \
  --lr 5e-5 \
  --num_train_epochs 3 \
  --max_prompt_tokens 700 \
  --max_response_tokens 512 \
  --generation_eval_limit 32 \
  --generation_eval_max_new_tokens 256 \
  --generation_eval_every 100 \
  --eval_interval 100 \
  --save_interval 100 \
  --wandb_enabled \
  --wandb_project llm-rl-final-project \
  --wandb_name wildchat_min4_judged_5k_ipo_v1
```

AOT

```
uv run modal run scripts/modal_train.py::train_remote -- \
  --algo aot \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
  --train_split train_prefs \
  --eval_split test_prefs \
  --generation_split test_gen \
  --output_dir /vol/runs/wildchat_min4_judged_5k_aot_beta02_v1 \
  --beta 0.2 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --grad_accum_steps 4 \
  --lr 5e-5 \
  --num_train_epochs 3 \
  --max_prompt_tokens 700 \
  --max_response_tokens 512 \
  --generation_eval_limit 32 \
  --generation_eval_max_new_tokens 256 \
  --generation_eval_every 50 \
  --eval_interval 50 \
  --save_interval 50 \
  --wandb_enabled \
  --wandb_project llm-rl-final-project \
  --wandb_name wildchat_min4_judged_5k_aot_beta02_v1
```

### Online RLHF Training

GRPO

```
uv run modal run scripts/modal_train.py::rm_grpo_train_remote -- \
  --algo grpo \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
  --train_split train_gen \
  --eval_split test_gen \
  --reward_model_name Qwen/Qwen2.5-1.5B-Instruct \
  --reward_adapter_path /vol/runs/wildchat_min4_judged_5k_reward_model_v1/checkpoints/step_000445/adapter \
  --output_dir /vol/runs/wildchat_min4_judged_5k_grpo_rm445_v1 \
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
  --wandb_name wildchat_min4_judged_5k_grpo_rm445_v1
```
DrGRPO
```
uv run modal run scripts/modal_train.py::rm_grpo_train_remote -- \
  --algo dr_grpo \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
  --train_split train_gen \
  --eval_split test_gen \
  --reward_model_name Qwen/Qwen2.5-1.5B-Instruct \
  --reward_adapter_path /vol/runs/wildchat_min4_judged_5k_reward_model_v1/checkpoints/step_000445/adapter \
  --output_dir /vol/runs/wildchat_min4_judged_5k_drgrpo_rm445_v1 \
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
  --wandb_name wildchat_min4_judged_5k_drgrpo_rm445_v1
```
GSPO
```
uv run modal run scripts/modal_train.py::rm_grpo_train_remote -- \
  --algo gspo \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
  --train_split train_gen \
  --eval_split test_gen \
  --reward_model_name Qwen/Qwen2.5-1.5B-Instruct \
  --reward_adapter_path /vol/runs/wildchat_min4_judged_5k_reward_model_v1/checkpoints/step_000445/adapter \
  --output_dir /vol/runs/wildchat_min4_judged_5k_gspo_rm445_v1 \
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
  --wandb_name wildchat_min4_judged_5k_gspo_rm445_v1
```

### Part 1 Checkpoint Paths used for Submission

| File | Adapter path |
| --- | --- |
| `policy_generations/dpo.jsonl` | `/vol/runs/wildchat_min4_judged_5k_dpo_beta01_v1/checkpoints/step_000100/adapter` |
| `policy_generations/ipo.jsonl` | `/vol/runs/wildchat_min4_judged_5k_ipo_v1/checkpoints/step_000300/adapter` |
| `policy_generations/aot.jsonl` | `/vol/runs/wildchat_min4_judged_5k_aot_beta02_v1/checkpoints/step_000550/adapter` |
| `policy_generations/grpo.jsonl` | `/vol/runs/wildchat_min4_judged_5k_grpo_rm445_v1/checkpoints/step_000025/adapter` |
| `policy_generations/drgrpo.jsonl` | `/vol/runs/wildchat_min4_judged_5k_drgrpo_rm445_v1/checkpoints/step_000025/adapter` |
| `policy_generations/gspo.jsonl` | `/vol/runs/wildchat_min4_judged_5k_gspo_rm445_v1/checkpoints/step_000025/adapter` |
| `reward_model/public_test_pref_scores.jsonl` | `/vol/runs/wildchat_min4_judged_5k_reward_model_v1/checkpoints/step_000445/adapter` |

### Part 1 Policy Submission

Replace adapter path and name with corresponding pair from above table
```
uv run modal run scripts/modal_train.py::build_policy_submission_remote -- \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_path /vol/runs/<run>/checkpoints/<step>/adapter \
  --prompts_jsonl /root/project/public_eval/public_test_gen_prompts_128.jsonl \
  --output_jsonl /vol/submissions/<name>.jsonl \
  --max_prompt_tokens 700 \
  --max_new_tokens 256 \
  --temperature 0.0 \
  --top_p 1.0
```

### Part 1 Reward Model Submission
```
uv run modal run scripts/modal_train.py::build_reward_model_submission_remote -- \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_path /vol/runs/wildchat_min4_judged_5k_reward_model_v1/checkpoints/step_000445/adapter \
  --prefs_jsonl /root/project/public_eval/public_test_prefs_256.jsonl \
  --output_jsonl /vol/submissions/public_test_pref_scores.jsonl \
  --max_prompt_tokens 700 \
  --max_response_tokens 512
```

### Part 2 Submissions

```
uv run modal run scripts/modal_train.py::build_policy_submission_remote -- \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_path /vol/runs/wildchat_min4_judged_5k_dpo_beta01_v1/checkpoints/step_000100/adapter \
  --prompts_jsonl /root/project/public_eval/public_test_gen_prompts_128.jsonl \
  --output_jsonl /vol/submissions/offline_best.jsonl \
  --max_prompt_tokens 700 \
  --max_new_tokens 256 \
  --temperature 0.0 \
  --top_p 1.0

uv run modal run scripts/modal_train.py::build_policy_submission_remote -- \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_path /vol/runs/wildchat_min4_judged_5k_gspo_gopo_rm100_rm445_ensemble_mean_v1/checkpoints/step_000025/adapter \
  --prompts_jsonl /root/project/public_eval/public_test_gen_prompts_128.jsonl \
  --output_jsonl /vol/submissions/online_best.jsonl \
  --max_prompt_tokens 700 \
  --max_new_tokens 256 \
  --temperature 0.0 \
  --top_p 1.0
```

Followed ReadMe to download remote files and create submission zip
