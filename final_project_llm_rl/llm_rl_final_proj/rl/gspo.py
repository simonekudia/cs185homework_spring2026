from __future__ import annotations

from typing import Dict

import torch

from llm_rl_final_proj.models.load import PolicyModel
from llm_rl_final_proj.rl.base import RLAlgorithm
from llm_rl_final_proj.rollout.rollout_buffer import RolloutBatch
from llm_rl_final_proj.models.logprobs import compute_per_token_logprobs, approx_kl_from_logprobs
from llm_rl_final_proj.rollout.rollout_buffer import iter_minibatches
        


class GSPO(RLAlgorithm):
    """Sequence-level clipped surrogate using geometric-mean likelihood ratios."""

    name = "gspo"

    def update(
        self,
        model: PolicyModel,
        optimizer: torch.optim.Optimizer,
        rollout: RolloutBatch,
        grad_accum_steps: int = 1,
    ) -> Dict[str, float]:
        # TODO(student): implement GSPO.
        # The main change relative to GRPO is that you should aggregate token log-ratios into
        # one sequence-level ratio before applying PPO-style clipping.
        cfg = self.cfg
        metrics_accum: Dict[str, float] = {}
        total_updates = 0
        for epoch in range(cfg.ppo_epochs):
            gen = torch.Generator()
            gen.manual_seed(self._next_update_seed())
            for mb in iter_minibatches(rollout, cfg.minibatch_size, shuffle=True, generator=gen):
                new_logprobs = compute_per_token_logprobs(
                    model,
                    input_ids=mb.input_ids,
                    attention_mask=mb.attention_mask,
                    enable_grad=True,
                )
                mask = mb.completion_mask
                adv = mb.advantages.clamp(-cfg.adv_clip, cfg.adv_clip)
                log_ratio = new_logprobs - mb.old_logprobs
                ratio = log_ratio.exp()
                token_counts = mask.sum(dim=1).clamp_min(1) 
                seq_log_ratio = (log_ratio * mask).sum(dim=1) / token_counts
                seq_ratio = seq_log_ratio.exp()
               
                unclipped = seq_ratio * adv
                clipped = seq_ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv
                surrogate = torch.min(unclipped, clipped) 
                
                policy_loss = -surrogate.mean()
                kl = approx_kl_from_logprobs(new_logprobs, mb.ref_logprobs, mask)
                loss = policy_loss + cfg.kl_coef * kl
                (loss / grad_accum_steps).backward()
                step_metrics = {
                    "train/policy_loss": policy_loss.item(),
                    "train/kl": kl.item(),
                    "train/loss": loss.item(),
                    "train/ratio_mean": ((ratio * mask).sum() / mask.sum().clamp_min(1)).item(),
                }
                for k, v in step_metrics.items():
                    metrics_accum[k] = metrics_accum.get(k, 0.0) + v
                total_updates += 1
                if total_updates % grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        cfg.max_grad_norm,
                    )
                    optimizer.step()
                    optimizer.zero_grad()
        if total_updates % grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                cfg.max_grad_norm,
            )
            optimizer.step()
            optimizer.zero_grad()
        return {k: v / max(1, total_updates) for k, v in metrics_accum.items()}

