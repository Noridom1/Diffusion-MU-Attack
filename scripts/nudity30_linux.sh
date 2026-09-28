#!/usr/bin/env bash
# Reproducible ESD/nudity setup and two-GPU launcher.  The default is the
# deterministic stress20 case-list subset; set CASE_LIST= to use random mode.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

ENV_NAME="${ENV_NAME:-ldm-nudity30}"
GPU0="${GPU0:-0}"
# Leave GPU1 empty on single-GPU hosts such as Google Colab.
GPU1="${GPU1-1}"
N_PROMPTS="${N_PROMPTS:-20}"
SAMPLE_SEED="${SAMPLE_SEED:-2024}"
RUN_SEED="${RUN_SEED:-0}"
SUBSET_TAG="${SUBSET_TAG:-stress20}"
# Set CASE_LIST= (an explicitly empty value) to restore seeded-random mode.
CASE_LIST="${CASE_LIST-prompts/nudity_stress20_case_numbers.txt}"
if [[ -n "$CASE_LIST" ]]; then
    EXPERIMENT="${EXPERIMENT:-nudity_${SUBSET_TAG}_run${RUN_SEED}}"
else
    EXPERIMENT="${EXPERIMENT:-nudity_n${N_PROMPTS}_sample${SAMPLE_SEED}_run${RUN_SEED}}"
fi

CACHE_DIR="${CACHE_DIR:-.cache}"
CHECKPOINT="${CHECKPOINT:-files/pretrained/SD-1-4/ESD_ckpt/Nudity-ESDx1-UNET-SD.pt}"
MANIFEST_DIR="files/manifests/$EXPERIMENT"
SUBSET_CSV="$MANIFEST_DIR/prompts.csv"
SUBSET_MANIFEST="$MANIFEST_DIR/manifest.json"
DATASET_DIR="files/dataset/$EXPERIMENT"
WORK_DIR="files/work/$EXPERIMENT"
RESULT_ROOT="files/results/$EXPERIMENT"
BASELINE_ROOT="$RESULT_ROOT/no_attack"
ATTACK_ROOT="$RESULT_ROOT/unlearndiff"
ORIGINAL_ROOT="$RESULT_ROOT/original_sd"
ORIGINAL_WORK="$WORK_DIR/original_sd"

BASELINE_CONFIG="configs/nudity/no_attack_esd_nudity_classifier.json"
ATTACK_CONFIG="configs/nudity/text_grad_esd_nudity_classifier.json"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

usage() {
    cat <<EOF
Usage: bash scripts/nudity30_linux.sh COMMAND

Commands:
  setup      Create the pinned Conda environment; download SD v1.4 and ESD checkpoint
  select     Freeze the configured case-list (default) or seeded random subset
  prepare    Select prompts and generate half the target images per GPU in parallel
  preflight  Verify GPUs, cached artifacts, subset, and prepared dataset
  baseline   Run no-attack evaluation, split across two GPUs
  attack     Run UnlearnDiff, split across two GPUs
  evaluate   Validate all result folders and print Pre-ASR/Post-ASR
  original   Generate the matched, unedited SD v1.4 arm at 50 steps
  report     Score all three arms, mask copies, and package report artifacts
  missing-figures  original -> report (does not rerun ESD or the attack)
  all        setup -> prepare -> preflight -> baseline -> attack -> evaluate

Environment overrides:
  ENV_NAME=$ENV_NAME  GPU0=$GPU0  GPU1=$GPU1 (empty GPU1 = single-GPU mode)
  N_PROMPTS=$N_PROMPTS  SAMPLE_SEED=$SAMPLE_SEED  RUN_SEED=$RUN_SEED
  SUBSET_TAG=$SUBSET_TAG  CASE_LIST=$CASE_LIST
  CACHE_DIR=$CACHE_DIR  CHECKPOINT=$CHECKPOINT
EOF
}

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "ERROR: required command not found: $1" >&2
        exit 1
    fi
}

conda_python() {
    conda run --no-capture-output -n "$ENV_NAME" python "$@"
}

wait_for_pair() {
    local pid0="$1"
    local pid1="$2"
    local status0=0
    local status1=0
    wait "$pid0" || status0=$?
    wait "$pid1" || status1=$?
    if (( status0 != 0 || status1 != 0 )); then
        echo "ERROR: parallel jobs failed (statuses: $status0, $status1)" >&2
        exit 1
    fi
}

setup_env() {
    require_command conda
    require_command git
    require_command nvidia-smi
    echo "Repository: $REPO_ROOT"
    git rev-parse HEAD
    nvidia-smi

    if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
        echo "Updating dedicated Conda environment: $ENV_NAME"
        conda env update -n "$ENV_NAME" \
            --file environments/nudity30-x86_64.yaml \
            --prune
    else
        conda env create -n "$ENV_NAME" --file environments/nudity30-x86_64.yaml
    fi

    mkdir -p "$MANIFEST_DIR"
    echo "Downloading SD v1.4 and the ESD bundle in parallel..."
    conda_python scripts/nudity30_tools.py download-model \
        --cache-dir "$CACHE_DIR" &
    local model_pid=$!
    conda_python scripts/nudity30_tools.py download-checkpoint \
        --target "$CHECKPOINT" \
        --download-dir files/downloads &
    local checkpoint_pid=$!
    wait_for_pair "$model_pid" "$checkpoint_pid"

    conda list -n "$ENV_NAME" --explicit > "$MANIFEST_DIR/conda-explicit.txt"
    conda run -n "$ENV_NAME" python -m pip freeze > "$MANIFEST_DIR/pip-freeze.txt"
    git rev-parse HEAD > "$MANIFEST_DIR/repo-commit.txt"
    local -a verify_args=(--cache-dir "$CACHE_DIR" --checkpoint "$CHECKPOINT")
    [[ -n "$GPU1" ]] && verify_args+=(--require-two-gpus)
    conda_python scripts/nudity30_tools.py verify "${verify_args[@]}"
}

select_prompts() {
    mkdir -p "$MANIFEST_DIR"
    local -a selection_args=(
        --source prompts/nudity.csv \
        --output "$SUBSET_CSV" \
        --manifest "$SUBSET_MANIFEST" \
        --cache-dir "$CACHE_DIR" \
        --count "$N_PROMPTS" \
        --seed "$SAMPLE_SEED"
    )
    if [[ -n "$CASE_LIST" ]]; then
        selection_args+=(--case-list "$CASE_LIST")
    fi
    conda_python scripts/nudity30_tools.py select "${selection_args[@]}"
}

prepare_images() {
    select_prompts
    mkdir -p "$WORK_DIR"
    echo "Generating target images on physical GPUs $GPU0 and $GPU1..."
    CUDA_VISIBLE_DEVICES="$GPU0" conda_python src/execs/generate_dataset.py \
        --prompts_path "${SUBSET_CSV%.csv}.shard0.csv" \
        --concept shard0 \
        --save_path "$WORK_DIR" \
        --device cuda:0 \
        --num_samples 1 \
        --ddim_steps 25 \
        --cache_dir "$CACHE_DIR" &
    local prepare0_pid=$!
    if [[ -n "$GPU1" ]]; then
        CUDA_VISIBLE_DEVICES="$GPU1" conda_python src/execs/generate_dataset.py \
            --prompts_path "${SUBSET_CSV%.csv}.shard1.csv" \
            --concept shard1 --save_path "$WORK_DIR" --device cuda:0 \
            --num_samples 1 --ddim_steps 25 --cache_dir "$CACHE_DIR" &
        wait_for_pair "$prepare0_pid" "$!"
    else
        wait "$prepare0_pid"
        CUDA_VISIBLE_DEVICES="$GPU0" conda_python src/execs/generate_dataset.py \
            --prompts_path "${SUBSET_CSV%.csv}.shard1.csv" \
            --concept shard1 --save_path "$WORK_DIR" --device cuda:0 \
            --num_samples 1 --ddim_steps 25 --cache_dir "$CACHE_DIR"
    fi

    mkdir -p "$DATASET_DIR/imgs"
    cp "$WORK_DIR/shard0/imgs/"*.png "$DATASET_DIR/imgs/"
    cp "$WORK_DIR/shard1/imgs/"*.png "$DATASET_DIR/imgs/"
    cp "$SUBSET_CSV" "$DATASET_DIR/prompts.csv"
    printf '[]\n' > "$DATASET_DIR/ignore.json"
    preflight
}

preflight() {
    local -a verify_args=(
        --cache-dir "$CACHE_DIR"
        --checkpoint "$CHECKPOINT"
        --subset "$SUBSET_CSV"
        --dataset "$DATASET_DIR"
        --expected-count "$N_PROMPTS"
    )
    [[ -n "$GPU1" ]] && verify_args+=(--require-two-gpus)
    if [[ -n "$CASE_LIST" ]]; then
        verify_args+=(--case-list "$CASE_LIST")
    fi
    CUDA_VISIBLE_DEVICES="${GPU0}${GPU1:+,$GPU1}" conda_python scripts/nudity30_tools.py verify \
        "${verify_args[@]}"
}

archive_incomplete_run() {
    local run_dir="$1"
    if [[ -f "$run_dir/log.json" && ! -f "$run_dir/.done" ]]; then
        local archived="${run_dir}.incomplete.$(date -u +%Y%m%dT%H%M%SZ)"
        echo "Archiving incomplete run: $run_dir -> $archived"
        mv "$run_dir" "$archived"
    fi
}

run_one() {
    local mode="$1"
    local gpu="$2"
    local idx="$3"
    local config
    local output_root
    local run_dir
    local -a extra_args

    if [[ "$mode" == "baseline" ]]; then
        config="$BASELINE_CONFIG"
        output_root="$BASELINE_ROOT"
        extra_args=(--attacker.no_attack.dataset_path "$DATASET_DIR")
    else
        config="$ATTACK_CONFIG"
        output_root="$ATTACK_ROOT"
        extra_args=()
    fi
    run_dir="$output_root/attack_idx_$idx"
    if [[ -f "$run_dir/.done" ]]; then
        echo "[$mode][GPU $gpu] skip completed index $idx"
        return
    fi
    archive_incomplete_run "$run_dir"
    mkdir -p "$run_dir"

    echo "[$mode][GPU $gpu] starting index $idx"
    CUDA_VISIBLE_DEVICES="$gpu" conda_python src/execs/attack.py \
        --config-file "$config" \
        --overall.seed "$RUN_SEED" \
        --task.target_ckpt "$CHECKPOINT" \
        --task.cache_path "$CACHE_DIR" \
        --task.dataset_path "$DATASET_DIR" \
        --attacker.attack_idx "$idx" \
        --logger.name "attack_idx_$idx" \
        --logger.json.root "$output_root" \
        "${extra_args[@]}" 2>&1 | tee "$run_dir/process.log"
    touch "$run_dir/.done"
    echo "[$mode][GPU $gpu] completed index $idx"
}

run_worker() {
    local mode="$1"
    local gpu="$2"
    local first="$3"
    local idx
    for ((idx=first; idx<N_PROMPTS; idx+=2)); do
        run_one "$mode" "$gpu" "$idx"
    done
}

run_experiment() {
    local mode="$1"
    preflight
    if [[ -z "$GPU1" ]]; then
        echo "Running $mode serially on GPU $GPU0"
        local idx
        for ((idx=0; idx<N_PROMPTS; idx++)); do
            run_one "$mode" "$GPU0" "$idx"
        done
        return
    fi
    echo "Running $mode with alternating static shards on GPUs $GPU0 and $GPU1"
    run_worker "$mode" "$GPU0" 0 &
    local worker0_pid=$!
    run_worker "$mode" "$GPU1" 1 &
    local worker1_pid=$!
    wait_for_pair "$worker0_pid" "$worker1_pid"
}

evaluate_results() {
    conda_python scripts/nudity30_tools.py summarize \
        --baseline-root "$BASELINE_ROOT" \
        --attack-root "$ATTACK_ROOT" \
        --expected-count "$N_PROMPTS"
}

generate_original() {
    select_prompts
    local existing=0
    if [[ -d "$ORIGINAL_ROOT/imgs" ]]; then
        existing="$(find "$ORIGINAL_ROOT/imgs" -maxdepth 1 -type f -name '*_0.png' | wc -l)"
    fi
    if [[ -f "$ORIGINAL_ROOT/.done" && "$existing" -eq "$N_PROMPTS" ]]; then
        echo "Matched original-SD arm already complete: $ORIGINAL_ROOT"
        return
    fi

    mkdir -p "$ORIGINAL_WORK"
    echo "Generating matched original SD v1.4 images at 50 steps..."
    CUDA_VISIBLE_DEVICES="$GPU0" conda_python src/execs/generate_dataset.py \
        --prompts_path "${SUBSET_CSV%.csv}.shard0.csv" \
        --concept shard0 \
        --save_path "$ORIGINAL_WORK" \
        --device cuda:0 \
        --num_samples 1 \
        --ddim_steps 50 \
        --cache_dir "$CACHE_DIR" &
    local original0_pid=$!
    if [[ -n "$GPU1" ]]; then
        CUDA_VISIBLE_DEVICES="$GPU1" conda_python src/execs/generate_dataset.py \
            --prompts_path "${SUBSET_CSV%.csv}.shard1.csv" \
            --concept shard1 --save_path "$ORIGINAL_WORK" --device cuda:0 \
            --num_samples 1 --ddim_steps 50 --cache_dir "$CACHE_DIR" &
        wait_for_pair "$original0_pid" "$!"
    else
        wait "$original0_pid"
        CUDA_VISIBLE_DEVICES="$GPU0" conda_python src/execs/generate_dataset.py \
            --prompts_path "${SUBSET_CSV%.csv}.shard1.csv" \
            --concept shard1 --save_path "$ORIGINAL_WORK" --device cuda:0 \
            --num_samples 1 --ddim_steps 50 --cache_dir "$CACHE_DIR"
    fi

    mkdir -p "$ORIGINAL_ROOT/imgs"
    cp "$ORIGINAL_WORK/shard0/imgs/"*.png "$ORIGINAL_ROOT/imgs/"
    cp "$ORIGINAL_WORK/shard1/imgs/"*.png "$ORIGINAL_ROOT/imgs/"
    cp "$SUBSET_CSV" "$ORIGINAL_ROOT/prompts.csv"
    existing="$(find "$ORIGINAL_ROOT/imgs" -maxdepth 1 -type f -name '*_0.png' | wc -l)"
    if [[ "$existing" -ne "$N_PROMPTS" ]]; then
        echo "ERROR: original-SD arm has $existing images, expected $N_PROMPTS" >&2
        exit 1
    fi
    touch "$ORIGINAL_ROOT/.done"
}

prepare_report_artifacts() {
    conda_python -m report.prepare_existing_figures \
        --original-root "$ORIGINAL_ROOT" \
        --baseline-root "$BASELINE_ROOT" \
        --attack-root "$ATTACK_ROOT" \
        --figure-dir report/figures \
        --expected-count "$N_PROMPTS"

    local bundle="$RESULT_ROOT/report_artifacts.tar.gz"
    tar -czf "$bundle" \
        report/figures/generated \
        report/figures/erasure_pairs.tex \
        report/figures/attack_triplets.tex \
        report/figures/detection_rates.tex \
        report/figures/report_metrics.json \
        "$ORIGINAL_ROOT/prompts.csv"
    sha256sum "$bundle" > "$bundle.sha256"
    echo "Report bundle ready: $bundle"
}

command="${1:-}"
case "$command" in
    setup) setup_env ;;
    select) select_prompts ;;
    prepare) prepare_images ;;
    preflight) preflight ;;
    baseline) run_experiment baseline ;;
    attack) run_experiment attack ;;
    evaluate) evaluate_results ;;
    original) generate_original ;;
    report) prepare_report_artifacts ;;
    missing-figures)
        generate_original
        prepare_report_artifacts
        ;;
    all)
        setup_env
        prepare_images
        preflight
        run_experiment baseline
        run_experiment attack
        evaluate_results
        ;;
    -h|--help|help) usage ;;
    *) usage; exit 2 ;;
esac
