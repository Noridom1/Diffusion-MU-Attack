# ESD nudity: reproducible stress20 run on two GPUs

This workflow keeps the upstream attack settings, but evaluates a frozen subset
rather than all 142 source rows. The default experiment uses a frozen
20-prompt stress subset, attack seed `0`, and physical GPUs `0` and `1`.

## What the wrapper fixes

- It selects the source IDs listed in
  `prompts/nudity_stress20_case_numbers.txt`. These are sexual-category prompts
  with `hard == 0` and original `nudity_percentage <= 20`; the CLIP `<=60`
  token constraint is checked before selection.
- It freezes source rows and original case numbers in a JSON manifest, then
  renumbers the subset to `0..19`, so `attack_idx` has an unambiguous meaning.
- It prepares reference images in two separate staging directories. The upstream
  generator is not safe for two processes writing the same dataset directory.
- It runs even indices on one GPU and odd indices on the other. Each command sees
  its assigned physical GPU as logical `cuda:0`, which matches the hard-coded task.
- It records the Git commit, explicit Conda package list, pip freeze, checkpoint
  SHA-256, per-prompt stdout, and a `.done` completion marker.
- It validates exactly 20 baseline and attack results before reporting rates.

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

The launcher name is retained for compatibility with the earlier n30 workflow.
To reproduce the old seeded-random behavior, explicitly clear the case list on
every invocation:

```bash
CASE_LIST= N_PROMPTS=30 SUBSET_TAG=sample30_seed2024 \
  bash scripts/nudity30_linux.sh select
```

`setup` downloads the Hugging Face SD v1.4 components and the exact ESD checkpoint
expected by the checked-in nudity configs concurrently. The checkpoint helper
uses HTTP byte ranges, so it does **not** need to download or unpack the authors'
25.5 GB Google Drive “Others” ZIP. If Hugging Face requires authentication, run
`huggingface-cli login` inside the environment and repeat `setup`.

If `esd_fmn_models.zip` has already been downloaded, do not unpack the whole
archive. Once the ZIP is complete, stream only the needed member (this requires
about 3.5 GiB of additional disk space):

```bash
unzip -p files/downloads/esd_fmn_models.zip \
  files/pretrained/SD-1-4/ESD_ckpt/Nudity-ESDx1-UNET-SD.pt \
  > files/pretrained/SD-1-4/ESD_ckpt/Nudity-ESDx1-UNET-SD.pt.part \
  && mv files/pretrained/SD-1-4/ESD_ckpt/Nudity-ESDx1-UNET-SD.pt.part \
        files/pretrained/SD-1-4/ESD_ckpt/Nudity-ESDx1-UNET-SD.pt
```

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

- `files/manifests/nudity_stress20_run0/`
- `files/dataset/nudity_stress20_run0/`
- `files/results/nudity_stress20_run0/no_attack/`
- `files/results/nudity_stress20_run0/unlearndiff/`

Reruns skip directories with `.done`. If a process was interrupted after writing
a log, the launcher moves that directory to a timestamped `.incomplete.*` backup
before retrying; it does not silently append a second experiment to the old log.

## Parallelism and expected resource use

There are exactly two attack workers, one per GPU. Each worker processes 10
indices sequentially. This avoids placing two model copies on a 16 GiB card.
Preparation also runs two generators concurrently, 10 prompts per GPU. Model and
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

Because `n=20`, one prompt changes either rate by 5 percentage points. Do not
expect the stress-set result to match the paper's full-set ESD value; report it
as a separate robustness/stress evaluation rather than as the headline ASR.
