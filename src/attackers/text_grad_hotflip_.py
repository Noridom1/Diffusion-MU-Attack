"""Baseline-timestep text attack using discrete HotFlip coordinate updates."""

from __future__ import annotations

from typing import Any

import torch

from .hotflip_utils import make_candidate_ids, replacement_scores, top_replacements
from .text_grad_ import TextGrad


class TextGradHotFlip(TextGrad):
    """Keep the baseline timestep protocol, but optimize actual token IDs."""

    def __init__(
        self,
        candidate_topk: int = 8,
        candidate_batch_size: int = 4,
        min_improvement: float = 0.0,
        init_strategy: str = 'random_allowed',
        forbid_special_tokens: bool = True,
        early_stop: bool = True,
        max_timesteps: int = 50,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.candidate_topk = int(candidate_topk)
        self.candidate_batch_size = int(candidate_batch_size)
        self.min_improvement = float(min_improvement)
        self.init_strategy = init_strategy
        self.forbid_special_tokens = bool(forbid_special_tokens)
        self.early_stop = bool(early_stop)
        self.max_timesteps = int(max_timesteps)
        if self.candidate_topk <= 0 or self.candidate_batch_size <= 0:
            raise ValueError('candidate_topk and candidate_batch_size must be positive')
        if self.max_timesteps <= 0:
            raise ValueError('max_timesteps must be positive')
        if self.init_strategy != 'random_allowed':
            raise ValueError("only init_strategy='random_allowed' is currently supported")

    @staticmethod
    def _decode_prompt(task, input_ids):
        ids = input_ids[0][1:].tolist()
        ids = [token_id for token_id in ids if token_id != task.tokenizer.eos_token_id]
        return task.tokenizer.decode(ids)

    @staticmethod
    def _allowed_mask(task, k, device, forbid_special_tokens=True):
        vocab_size = task.all_embeddings.shape[1]
        mask = torch.ones((k, vocab_size), dtype=torch.bool, device=device)
        if forbid_special_tokens and task.tokenizer.all_special_ids:
            special_ids = [token_id for token_id in task.tokenizer.all_special_ids
                           if 0 <= token_id < vocab_size]
            if special_ids:
                mask[:, special_ids] = False
        if forbid_special_tokens and not task.tokenizer.all_special_ids and task.tokenizer.eos_token_id is not None:
            if 0 <= task.tokenizer.eos_token_id < vocab_size:
                mask[:, task.tokenizer.eos_token_id] = False
        return mask

    @staticmethod
    def _construct_candidate_full_ids(attacker, candidate_adv_ids, sot_id, eot_id, mid_id):
        full_ids = [
            attacker.construct_id(candidate.unsqueeze(0), sot_id, eot_id, mid_id)
            for candidate in candidate_adv_ids
        ]
        return torch.cat(full_ids, dim=0)

    def _initial_ids(self, task, allowed_mask):
        vocab_size = allowed_mask.shape[1]
        allowed = allowed_mask[0].nonzero(as_tuple=False).flatten()
        if allowed.numel() == 0:
            raise RuntimeError('no allowed vocabulary tokens available for HotFlip')
        random_indices = torch.randint(allowed.numel(), (self.k,), device=task.device)
        return allowed[random_indices].unsqueeze(0)

    def _candidate_eval(self, task, x0, t, candidate_adv_ids, sot_id, eot_id,
                        mid_id, noise):
        candidate_full_ids = self._construct_candidate_full_ids(
            self, candidate_adv_ids, sot_id, eot_id, mid_id
        )
        losses = []
        for start in range(0, candidate_full_ids.shape[0], self.candidate_batch_size):
            end = start + self.candidate_batch_size
            ids_chunk = candidate_full_ids[start:end]
            embeds_chunk = task.id2embedding(ids_chunk)
            losses.append(task.get_loss(
                x0=x0,
                t=t,
                input_ids=ids_chunk,
                input_embeddings=embeds_chunk,
                noise=noise,
                reduction='none',
            ).detach())
        return candidate_full_ids, torch.cat(losses, dim=0)

    def run(self, task, logger):
        image, prompt, seed, guidance = task.dataset[self.attack_idx]
        if seed is None:
            seed = self.eval_seed
        task.tokenizer.pad_token = task.tokenizer.eos_token

        original_ids = task.str2id(prompt)
        original_embeddings = task.id2embedding(original_ids)
        original_len = (original_ids == 49407).nonzero(as_tuple=True)[1][0] - 1
        sot_id, mid_id, eot_id = self.split_id(original_ids, original_len)
        self.split_embd(original_embeddings, original_len)

        results = task.eval(original_ids, prompt, seed=seed, guidance_scale=guidance)
        results['prompt'] = prompt
        logger.save_img('orig', results.pop('image'))
        logger.log(results)
        if results.get('success'):
            return 0
        if self.universal or not self.sequential:
            raise NotImplementedError('HotFlip currently requires sequential, non-universal attacks')

        x0 = task.img2latent(image)
        allowed_mask = self._allowed_mask(
            task, self.k, task.device, self.forbid_special_tokens
        )
        adv_ids = self._initial_ids(task, allowed_mask)
        timestep_values = task.sampled_t[:self.max_timesteps]

        for outer_step, t in enumerate(timestep_values):
            total_loss = 0.0
            inner_history = []
            accepted_flips = 0

            for inner_step in range(self.iteration):
                noise = torch.randn((1, 4, 64, 64), device=task.device)
                current_adv_embeds = task.all_embeddings.squeeze(0)[adv_ids]
                current_adv_embeds = current_adv_embeds.detach().requires_grad_(True)
                current_full_ids = self.construct_id(adv_ids, sot_id, eot_id, mid_id)
                current_full_embeds = self.construct_embd(current_adv_embeds)
                gradient_loss = task.get_loss(
                    x0=x0,
                    t=t,
                    input_ids=current_full_ids,
                    input_embeddings=current_full_embeds,
                    noise=noise,
                )
                gradient = torch.autograd.grad(gradient_loss, current_adv_embeds)[0].squeeze(0)
                scores = replacement_scores(gradient, task.all_embeddings.squeeze(0))
                positions, token_ids, predicted_scores = top_replacements(
                    scores, adv_ids.squeeze(0), allowed_mask, self.candidate_topk
                )
                candidate_adv_ids = make_candidate_ids(
                    adv_ids.squeeze(0), positions, token_ids
                )
                all_adv_ids = torch.cat([adv_ids, candidate_adv_ids], dim=0)
                _, exact_losses = self._candidate_eval(
                    task, x0, t, all_adv_ids, sot_id, eot_id, mid_id, noise
                )
                best_index = int(torch.argmin(exact_losses).item())
                current_exact = float(exact_losses[0].item())
                best_exact = float(exact_losses[best_index].item())
                accepted = best_index != 0 and best_exact < current_exact - self.min_improvement
                if accepted:
                    adv_ids = all_adv_ids[best_index:best_index + 1].detach()
                    accepted_flips += 1

                total_loss += float(gradient_loss.detach().item())
                inner_history.append({
                    'inner_step': inner_step,
                    'gradient_loss': float(gradient_loss.detach().item()),
                    'current_exact_loss': current_exact,
                    'best_exact_loss': best_exact,
                    'accepted': bool(accepted),
                    'position': int(positions[best_index - 1].item()) if accepted else None,
                    'new_token_id': int(token_ids[best_index - 1].item()) if accepted else None,
                    'predicted_score': float(predicted_scores[best_index - 1].item()) if accepted else None,
                })

            final_ids = self.construct_id(adv_ids, sot_id, eot_id, mid_id)
            final_prompt = self._decode_prompt(task, final_ids)
            results = task.eval(final_ids, final_prompt, seed, guidance_scale=guidance)
            results.update({
                'prompt': final_prompt,
                'loss': total_loss,
                'loss_mean': total_loss / max(self.iteration, 1),
                'record_type': 'evaluation',
                'outer_step': outer_step,
                'timestep': int(t),
                'accepted_flips': accepted_flips,
                'adv_token_ids': adv_ids[0].tolist(),
                'inner_history': inner_history,
            })
            logger.save_img(f'{t}', results.pop('image'))
            logger.log(results)
            if self.early_stop and results.get('success'):
                break
        return 0


def get(**kwargs):
    return TextGradHotFlip(**kwargs)
