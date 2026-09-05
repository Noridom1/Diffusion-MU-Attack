"""Text-gradient attack with stratified timestep expectation over transforms.

The original :mod:`text_grad_` attacker is intentionally left unchanged.  This
variant performs one optimizer update using three timestep losses (low, middle,
and high regions of the task's sampled diffusion timesteps), then evaluates the
discrete prompt every ``eval_interval`` updates.  The three losses share the
same straight-through sampled prefix for an update, so the update optimizes a
single prompt against a small, deterministic EOT batch.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .text_grad_ import STERandSelect, TextGrad
from .text_grad_eot_utils import make_timestep_schedule, evaluation_steps


class TextGradEOT(TextGrad):
    """TextGrad variant that averages gradients across three timesteps."""

    def __init__(
        self,
        timesteps_per_update: int = 3,
        eval_interval: int = 10,
        early_stop: bool = True,
        timestep_seed: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.timesteps_per_update = int(timesteps_per_update)
        self.eval_interval = int(eval_interval)
        self.early_stop = bool(early_stop)
        self.timestep_seed = int(timestep_seed)

    @staticmethod
    def _eval_preserving_rng(task, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Evaluate without changing the RNG stream used by optimization.

        ``ClassifierTask.sampling`` calls ``torch.manual_seed(seed)``.  Without
        restoring the states, every evaluation would silently change the random
        noise used by subsequent ``get_loss`` calls, making an EOT run depend on
        the evaluation frequency.
        """

        cpu_state = torch.random.get_rng_state()
        cuda_states = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        try:
            return task.eval(*args, **kwargs)
        finally:
            torch.random.set_rng_state(cpu_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)

    @staticmethod
    def _log_evaluation(
        logger,
        image_name: str,
        results: dict[str, Any],
        prompt: str,
        *,
        optimization_step: int | None = None,
        loss_window: list[dict[str, Any]] | None = None,
    ) -> None:
        """Save one evaluation image and a JSON-serializable log record."""

        results = dict(results)
        results["prompt"] = prompt
        if optimization_step is not None:
            window = list(loss_window or [])
            results["record_type"] = "evaluation"
            results["optimization_step"] = int(optimization_step)
            results["loss_window"] = window
            if window:
                results["loss"] = float(window[-1]["mean_loss"])
                results["loss_window_mean"] = float(
                    np.mean([entry["mean_loss"] for entry in window])
                )
        image = results.pop("image", None)
        if image is not None:
            logger.save_img(image_name, image)
        logger.log(results)

    def _decode_projected_prompt(
        self,
        task,
        projected_ids: torch.Tensor,
        sot_id: torch.Tensor,
        eot_id: torch.Tensor,
        mid_id: torch.Tensor,
    ) -> tuple[torch.Tensor, str]:
        new_id = self.construct_id(projected_ids, sot_id, eot_id, mid_id)
        id_list = new_id[0][1:].tolist()
        id_list = [token_id for token_id in id_list if token_id != task.tokenizer.eos_token_id]
        prompt = task.tokenizer.decode(id_list)
        return new_id, prompt

    def run(self, task, logger):
        image, prompt, seed, guidance = task.dataset[self.attack_idx]

        if seed is None:
            seed = self.eval_seed

        task.tokenizer.pad_token = task.tokenizer.eos_token

        visualize_prompt_id = task.str2id(prompt)
        visualize_embedding = task.id2embedding(visualize_prompt_id)
        visualize_orig_prompt_len = (
            (visualize_prompt_id == 49407).nonzero(as_tuple=True)[1][0] - 1
        )

        self.init_adv(task, visualize_orig_prompt_len.item())
        self.init_opt()

        visualize_sot_id, visualize_mid_id, visualize_eot_id = self.split_id(
            visualize_prompt_id, visualize_orig_prompt_len
        )

        results = self._eval_preserving_rng(
            task,
            visualize_prompt_id,
            prompt,
            seed=seed,
            guidance_scale=guidance,
        )
        self._log_evaluation(logger, "orig", results, prompt)
        if results.get("success") is not None and results["success"]:
            return 0

        if self.universal:
            raise NotImplementedError("TextGradEOT does not support universal attacks")
        if not self.sequential:
            raise NotImplementedError("TextGradEOT requires sequential optimization")

        x0 = task.img2latent(image)
        input_ids = task.str2id(prompt)
        orig_prompt_len = (input_ids == 49407).nonzero(as_tuple=True)[1][0] - 1
        input_embeddings = task.id2embedding(input_ids)
        self.split_embd(input_embeddings, orig_prompt_len)

        schedule = make_timestep_schedule(
            task.sampled_t,
            self.iteration,
            timesteps_per_update=self.timesteps_per_update,
            seed=self.timestep_seed,
        )
        checkpoints = set(evaluation_steps(self.iteration, self.eval_interval))
        loss_window: list[dict[str, Any]] = []

        for step, timesteps in enumerate(schedule, start=1):
            self.optimizer.zero_grad()
            # One STE sample is shared by all three timestep losses in this
            # update.  This makes their average gradient target one prompt.
            adv_one_hot = STERandSelect.apply(self.adv_embedding)
            tmp_embeds = adv_one_hot @ task.all_embeddings
            adv_input_embeddings = self.construct_embd(tmp_embeds)

            component_losses: list[float] = []
            losses: list[torch.Tensor] = []
            for t in timesteps:
                input_arguments = {
                    "x0": x0,
                    "t": t,
                    "input_ids": input_ids,
                    "input_embeddings": adv_input_embeddings,
                    "orig_input_ids": visualize_prompt_id,
                    "orig_input_embeddings": visualize_embedding,
                    "seed": seed,
                    "guidance_scale": guidance,
                }
                loss = task.get_loss(**input_arguments)
                losses.append(loss)
                component_losses.append(float(loss.detach().item()))

            mean_loss_tensor = torch.stack(losses).mean()
            self.adv_embedding.grad = torch.autograd.grad(
                mean_loss_tensor, [self.adv_embedding]
            )[0]
            self.optimizer.step()
            self.adv_embedding.data = self.projection(self.adv_embedding).data

            loss_window.append(
                {
                    "optimization_step": step,
                    "timesteps": [int(t) for t in timesteps],
                    "component_losses": component_losses,
                    "mean_loss": float(mean_loss_tensor.detach().item()),
                }
            )

            if step not in checkpoints:
                continue

            _, projected_ids = self.argmax_project(
                self.adv_embedding, task.all_embeddings, task.tokenizer
            )
            new_visualize_id, new_visualize_prompt = self._decode_projected_prompt(
                task,
                projected_ids,
                visualize_sot_id,
                visualize_eot_id,
                visualize_mid_id,
            )
            eval_results = self._eval_preserving_rng(
                task,
                new_visualize_id,
                new_visualize_prompt,
                seed,
                guidance_scale=guidance,
            )
            self._log_evaluation(
                logger,
                f"step_{step:04d}",
                eval_results,
                new_visualize_prompt,
                optimization_step=step,
                loss_window=loss_window,
            )
            loss_window = []

            if self.early_stop and eval_results.get("success"):
                break

        return 0


def get(**kwargs):
    return TextGradEOT(**kwargs)
