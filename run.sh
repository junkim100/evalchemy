#!/bin/bash

set -euo pipefail

# =============================================================================
# HELPERS
# =============================================================================
SCRIPT_NAME="$(basename "$0")"

print_presets() {
  cat <<'EOF'
Available run_name presets:
  minimal : AIME24, BigCodeBench, MATH500, WMT24, LogicKor, GPQADiamond, zeroeval, AutoLogi, LongBenchv2, global_mmlu_ko, hrm8k, cmmlu, TauBench, HarmBench
  full     : All minimal tasks plus: ArenaHard, WritingBench, longbench, LongBenchv2, babilong, zebralogic, LiveCodeBenchv5, japanese_leaderboard, HLE, WMT19, mmlu_redux, gpqa_diamond_zeroshot, IFEval, AIME25, gsm8k_symbolic, ruler, m_ifeval_ja, mgsm_direct_en/ja/zh, mmlu_prox_ko, include_base_44_korean, kobalt, csatqa, click, kmmlu, ceval-valid, agieval_cn
  full_minus_minimal : Tasks in 'full' excluding those in 'minimal'

EOF
}

print_help() {
  cat <<EOF
Usage:
  $SCRIPT_NAME --model_path <path> --run_name <minimal|full|full_minus_minimal|broken1> [--debug] [--debug_limit N]

Required:
  --model_path <path>     Path to the model directory (used for both pretrained= and tokenizer=)
  --run_name <name>       One of the presets below (controls which benchmarks run)

$(print_presets)

Optional:
  --debug                 Enable debug mode (propagates to eval.eval --debug and model_args limit=)
  --debug_limit N         Cap number of samples per task when --debug is enabled (default: 64)
  -h, --help              Show this help and exit
  --list-presets          Show only the preset names and tasks

Examples:
  $SCRIPT_NAME --model_path /mnt/models/Qwen3-30B-A3B --run_name minimal
  $SCRIPT_NAME --model_path /mnt/models/Llama-3-70B --run_name full --debug --debug_limit 32
  $SCRIPT_NAME --model_path /mnt/models/Llama-3-70B --run_name full_minus_minimal

Output path pattern:
  LOG_DIR/{model_name}/{run_name[_debug]}
EOF
}

die() { echo "Error: $*" >&2; echo; print_help >&2; exit 2; }

# Quick one-shot helpers
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  print_help
  exit 0
fi
if [[ "${1:-}" == "--list-presets" ]]; then
  print_presets
  exit 0
fi

# =============================================================================
# CLI FLAGS
#   --model_path <path>                    (required: path to model dir)
#   --run_name <minimal|full|full_minus_minimal|broken1>       (required: controls which TASKS to run)
#   --debug                                (optional: enables debug mode)
#   --debug_limit <N>                      (optional: #samples in debug; default 64)
# =============================================================================

MODEL_PATH=""
RUN_NAME=""
DEBUG="false"
DEBUG_LIMIT=64

# No-arg call? Show help.
if [[ $# -eq 0 ]]; then
  print_help
  exit 0
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_path)
      [[ $# -ge 2 ]] || die "--model_path requires a value"
      MODEL_PATH="$2"; shift 2;;
    --run_name)
      [[ $# -ge 2 ]] || die "--run_name requires a value"
      RUN_NAME="$2"; shift 2;;
    --debug)
      DEBUG="true"; shift 1;;
    --debug_limit)
      [[ $# -ge 2 ]] || die "--debug_limit requires an integer value"
      DEBUG_LIMIT="$2"
      # Validate integer
      [[ "$DEBUG_LIMIT" =~ ^[0-9]+$ ]] || die "--debug_limit must be a non-negative integer"
      shift 2;;
    -h|--help)
      print_help; exit 0;;
    --list-presets)
      print_presets; exit 0;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
done

[[ -n "$MODEL_PATH" ]] || die "--model_path is required."
[[ -n "$RUN_NAME"   ]] || die "--run_name is required (e.g., minimal, full, full_minus_minimal, broken1)."

# =============================================================================
# BASE CONFIG
# =============================================================================

BASE_MODEL_ARGS="pretrained=${MODEL_PATH},tokenizer=${MODEL_PATH},tensor_parallel_size=8,dtype=bfloat16,gpu_memory_utilization=0.5,max_model_len=32768,trust_remote_code=True,disable_log_stats=True"

# =============================================================================
# TASK PRESETS BY RUN_NAME (Option A: fail fast on unknown)
# =============================================================================
declare -a TASKS
case "$RUN_NAME" in
  minimal) TASKS=("BigCodeBench" "MATH500" "GPQADiamond" "WMT24" "LogicKor" "zebralogic" "LongBenchv2" "global_mmlu_ko" "hrm8k" "cmmlu" "TauBench" "HarmBench") ;;
  full)    TASKS=("AIME24" "AIME25" "BigCodeBench" "MATH500" "GPQADiamond" "WMT19" "WMT24" "LogicKor" "zeroeval" "AutoLogi" "LongBenchv2" "global_mmlu_ko" "hrm8k" "cmmlu" "TauBench" "HarmBench" "ArenaHard" "WritingBench" "longbench" "babilong" "zebralogic" "LiveCodeBenchv5" "japanese_leaderboard" "HLE" "mmlu_redux" "gpqa_diamond_zeroshot" "IFEval" "gsm8k_symbolic" "ruler" "m_ifeval_ja" "mgsm_direct_en" "mgsm_direct_ja" "mgsm_direct_zh" "mmlu_prox_ko" "include_base_44_korean" "kobalt" "csatqa" "click" "kmmlu" "ceval-valid" "agieval_cn") ;;
  full_minus_minimal) TASKS=("AIME24" "AIME25" "WMT19" "zeroeval" "ArenaHard" "WritingBench" "longbench" "babilong" "zebralogic" "LiveCodeBenchv5" "japanese_leaderboard" "HLE" "mmlu_redux" "gpqa_diamond_zeroshot" "IFEval" "gsm8k_symbolic" "ruler" "m_ifeval_ja" "mgsm_direct_en" "mgsm_direct_ja" "mgsm_direct_zh" "mmlu_prox_ko" "include_base_44_korean" "kobalt" "csatqa" "click" "kmmlu" "ceval-valid" "agieval_cn") ;;
  *)
    die "unknown --run_name '$RUN_NAME'. Valid options: minimal, full, full_minus_minimal, broken1"
    ;;
esac

# Convert tasks array to comma-separated string
TASKS_STRING=$(IFS=','; echo "${TASKS[*]}")

echo "=== WBL Evaluation ==="
echo "Timestamp: $(date)"
echo "Run name: ${RUN_NAME}"
echo "Debug: ${DEBUG} (limit=${DEBUG_LIMIT})"
echo "Tasks to evaluate: ${TASKS_STRING}"

# =============================================================================
# ENVIRONMENT
# =============================================================================
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export MPLBACKEND=Agg
export PYTHONWARNINGS="ignore:findfont"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

# =============================================================================
# DIRECTORIES
# =============================================================================
LOG_DIR="${LOG_DIR:-/mnt/nlpai-storage/eval_team/logs}"
MODEL_NAME="$(basename "$MODEL_PATH")"

OUT_RUN_NAME="$RUN_NAME"
if [[ "$DEBUG" == "true" ]]; then
  OUT_RUN_NAME="${RUN_NAME}_debug"
fi

RUN_TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
OUTPUT_DIR="${LOG_DIR}/${MODEL_NAME}/${OUT_RUN_NAME}_${RUN_TIMESTAMP}"
if ! mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"; then
  die "Failed to create log/output directories. Check permissions: ${LOG_DIR}"
fi

echo "Log directory: ${LOG_DIR}"
echo "Output directory: ${OUTPUT_DIR}"

# =============================================================================
# CLEANUP PRE-EXISTING PROCESSES
# =============================================================================
echo "Cleaning up existing processes..."
pkill -f 'python.*eval.eval' 2>/dev/null || true
pkill -f 'VllmWorkerProcess' 2>/dev/null || true
ray stop --force 2>/dev/null || true
echo "Did not find any active Ray processes."

# =============================================================================
# VERIFY MODEL
# =============================================================================
if [ ! -d "$MODEL_PATH" ]; then
  echo "Error: Model not found at $MODEL_PATH"
  exit 1
fi
echo "Model found at: $MODEL_PATH"

# =============================================================================
# BUILD MODEL_ARGS AND EVAL FLAGS
# =============================================================================
MODEL_ARGS="$BASE_MODEL_ARGS"

# Debug flag and limit for eval.eval
EVAL_DEBUG_FLAG=()
if [[ "$DEBUG" == "true" ]]; then
  EVAL_DEBUG_FLAG+=(--debug)
  EVAL_DEBUG_FLAG+=(--limit "${DEBUG_LIMIT}")
fi

# =============================================================================
# START EVALUATION
# =============================================================================
echo "Starting evaluation..."
echo "Tasks: ${TASKS_STRING}"

# Create a timestamp marker to locate "newer" temp outputs later
START_MARKER_FILE="/tmp/eval_start_$$.stamp"
date +"%s" > "$START_MARKER_FILE"

cd /app
set +e  # allow pipeline to capture exit code
python -m eval.eval \
  --model vllm \
  --tasks "${TASKS_STRING}" \
  --model_args "${MODEL_ARGS}" \
  --batch_size 1 \
  --output_path "${OUTPUT_DIR}" \
  --apply_chat_template \
  --verbosity INFO \
  "${EVAL_DEBUG_FLAG[@]}" 2>&1 | tee "${OUTPUT_DIR}/evaluation_${RUN_TIMESTAMP}.log"
EVAL_EXIT_CODE=${PIPESTATUS[0]}
set -e

echo "[eval] Evaluation completed with exit code: $EVAL_EXIT_CODE"

# =============================================================================
# CLEANUP RUNTIME
# =============================================================================
pkill -f 'VllmWorkerProcess' 2>/dev/null || true
ray stop --force 2>/dev/null || true

# =============================================================================
# CHECK RESULTS + RESCUE FROM TEMP
# =============================================================================
echo "Checking results in: $OUTPUT_DIR"
if [ -d "$OUTPUT_DIR" ]; then
  RESULT_FILES=$(find "$OUTPUT_DIR" -name "*.json" -o -name "*.jsonl" | wc -l | tr -d ' ')
  echo "Found $RESULT_FILES result files"

  if [ "$RESULT_FILES" -gt 0 ]; then
    echo "=== EVALUATION SUCCESSFUL ==="
    echo "Results saved to: $OUTPUT_DIR"
    ls -la "$OUTPUT_DIR"
    echo "Sample result files:"
    find "$OUTPUT_DIR" -name "*.json" -o -name "*.jsonl" | head -5
  else
    echo "=== CHECKING TEMP DIRECTORIES FOR RESULTS ==="
    # Use the start marker to find newly written temp files
    TEMP_RESULTS=$(find /tmp -name "generated_*.jsonl" -newer "$START_MARKER_FILE" 2>/dev/null | head -10)
    if [ -n "$TEMP_RESULTS" ]; then
      echo "Found results in temp directories, copying to permanent location..."
      LATEST_TEMP_DIR=$(find /tmp -name "generated_*.jsonl" -newer "$START_MARKER_FILE" -exec dirname {} \; 2>/dev/null | sort -u | head -1)
      if [ -n "$LATEST_TEMP_DIR" ]; then
        RESULTS_DIR="${LOG_DIR}/${MODEL_NAME}/${OUT_RUN_NAME}_temp"
        mkdir -p "$(dirname "$RESULTS_DIR")"
        cp -r "$LATEST_TEMP_DIR" "$RESULTS_DIR"
        echo "=== EVALUATION SUCCESSFUL ==="
        echo "Results copied to: $RESULTS_DIR"
        ls -la "$RESULTS_DIR"
        echo "Result file summary:"
        if ls "$RESULTS_DIR"/generated_*.jsonl 1> /dev/null 2>&1; then
          wc -l "$RESULTS_DIR"/generated_*.jsonl
        else
          echo "Generated files: $(find "$RESULTS_DIR" -name "*.json*" | wc -l | tr -d ' ')"
        fi
      else
        echo "=== EVALUATION FAILED - NO RESULTS ==="
        exit 1
      fi
    else
      echo "=== EVALUATION FAILED - NO RESULTS ==="
      exit 1
    fi
  fi
else
  echo "[eval] Output directory not found"
  exit 1
fi

# =============================================================================
# FINAL CLEANUP
# =============================================================================
echo "[cleanup] Starting cleanup of vLLM / Ray / eval processes…"
pkill -f 'python.*eval' 2>/dev/null || true
pkill -f 'VllmWorkerProcess' 2>/dev/null || true
pkill -f 'ray::' 2>/dev/null || true
ray stop --force 2>/dev/null || true
echo "Did not find any active Ray processes."

python3 - <<'PY'
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print('[cleanup] GPU memory cleared')
else:
    print('[cleanup] No CUDA available')
PY

# =============================================================================
# CONSOLIDATE SUMMARY
# =============================================================================
echo "[eval] Consolidating results..."
if [ -d "$OUTPUT_DIR" ]; then
  TOTAL_FILES=$(find "$OUTPUT_DIR" -name "*.json" -o -name "*.jsonl" | wc -l | tr -d ' ')
  echo "[eval] Total result files generated: $TOTAL_FILES"
  if [ "$TOTAL_FILES" -gt 0 ]; then
    echo "[eval] ✓ Evaluation completed successfully for tasks: ${TASKS_STRING}"
    echo "[eval] Results location: $OUTPUT_DIR"
  else
    echo "[eval] ✗ No results generated"
  fi
else
  echo "[eval] ✗ Output directory not found"
fi

echo "[eval] Evaluation script completed"
