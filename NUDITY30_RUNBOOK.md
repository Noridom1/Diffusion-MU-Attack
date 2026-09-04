# ESD nudity: reproducible 30-prompt run on two GPUs

This workflow keeps the upstream attack settings, but evaluates a frozen random
subset rather than all 142 source rows. The default experiment uses 30 prompts,
subset seed `2024`, attack seed `0`, and physical GPUs `0` and `1`.

## What the wrapper fixes

- It samples only prompts that pass the repo's CLIP `<=60` token filter, then
  freezes their source rows and original case numbers in a JSON manifest.
- It renumbers the subset to `0..29`, so `attack_idx` has an unambiguous meaning.
- It prepares reference images in two separate staging directories. The upstream
  generator is not safe for two processes writing the same dataset directory.
- It runs even indices on one GPU and odd indices on the other. Each command sees
  its assigned physical GPU as logical `cuda:0`, which matches the hard-coded task.
- It records the Git commit, explicit Conda package list, pip freeze, checkpoint
  SHA-256, per-prompt stdout, and a `.done` completion marker.
- It validates exactly 30 baseline and attack results before reporting rates.

The subset is a fast course-project baseline, not a statistically equivalent
reproduction of the paper's 142-prompt headline number. Keep this same manifest
for every later ablation and report confidence intervals or per-prompt paired
differences when comparing methods.

## Run

Prerequisites are a recent NVIDIA driver, two visible GPUs, Git, and Conda. The
environment itself uses the upstream CUDA 11.8/PyTorch 2.0.1 stack. From the repo
root:

```bash
bash scripts/nudity30_linux.sh setup
bash scripts/nudity30_linux.sh prepare
bash scripts/nudity30_linux.sh baseline
bash scripts/nudity30_linux.sh attack
bash scripts/nudity30_linux.sh evaluate
```

`setup` downloads the Hugging Face SD v1.4 components and the authors' Google
Drive “Others” bundle concurrently. It extracts the exact ESD checkpoint expected
by the checked-in nudity configs. If Hugging Face requires authentication, run
`huggingface-cli login` inside the environment and repeat `setup`.

The single-command form is available, but the attack can take many hours:

```bash
bash scripts/nudity30_linux.sh all
```

To use other physical GPU IDs or seeds, provide environment variables on every
invocation so all paths remain consistent:

```bash
GPU0=2 GPU1=3 SAMPLE_SEED=2024 RUN_SEED=0 \
  bash scripts/nudity30_linux.sh prepare
```

The main artifacts are under:

- `files/manifests/nudity_n30_sample2024_run0/`
- `files/dataset/nudity_n30_sample2024_run0/`
- `files/results/nudity_n30_sample2024_run0/no_attack/`
- `files/results/nudity_n30_sample2024_run0/unlearndiff/`

Reruns skip directories with `.done`. If a process was interrupted after writing
a log, the launcher moves that directory to a timestamped `.incomplete.*` backup
before retrying; it does not silently append a second experiment to the old log.

## Parallelism and expected resource use

There are exactly two attack workers, one per GPU. Each worker processes 15
indices sequentially. This avoids placing two model copies on a 16 GiB card.
Preparation also runs two generators concurrently, 15 prompts per GPU. Model and
checkpoint downloads run concurrently during setup.

The upstream executable initializes a new Python process and reloads the models
for every prompt; this wrapper deliberately preserves that behavior for a clean
baseline. Static even/odd sharding can leave one GPU idle near the end because
successful attacks stop early. A persistent-task worker or dynamic queue is a
later performance optimization and should be measured separately from the first
reproduction.

## Interpretation

`evaluate` reports:

- **Pre-ASR:** baseline prompts whose original generation is detected as nudity.
- **Post-ASR:** prompts whose final attack log is successful, including original
  prompts that were already successful.
- **New successes:** attacked successes restricted to baseline failures.

Because `n=30`, one prompt changes either rate by 3.33 percentage points. Do not
expect the result to exactly match the paper's full-set ESD value.
