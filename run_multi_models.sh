#!/bin/bash

# =============================================================================
# WBL Multi-Model Evaluation Script
# =============================================================================
# This script runs evaluations on multiple models using the existing run.sh script.
# It loops through a predefined list of models and runs the specified evaluation preset.
#
# Usage:
#   ./run_multi_models.sh [OPTIONS]
#
# Options:
#   --run_name PRESET     Evaluation preset (minimal, full1, full2) [default: minimal]
#   --debug               Enable debug mode with limited samples
#   --debug_limit N       Number of samples in debug mode [default: 64]
#   --parallel            Run models in parallel (experimental)
#   --skip_existing       Skip models that already have results
#   --help                Show this help message
#
# Examples:
#   ./run_multi_models.sh --run_name minimal --debug
#   ./run_multi_models.sh --run_name full1 --skip_existing
#   ./run_multi_models.sh --run_name minimal --parallel
# =============================================================================

set -euo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================

# List of models to evaluate
MODELS=(
    "/mnt/nlpai-storage/models/Qwen3-30B-A3B"
    "/mnt/nlpai-storage/models/gpt-oss-120b"
    "/mnt/nlpai-storage/Qwen3-Next-80B-A3B-Thinking"
    "/mnt/nlpai-storage/models/Qwen3-235B-A22B-Thinking-2507"
    "/mnt/nlpai-storage/models/DeepSeek-V3.1"
)

# Default values
RUN_NAME="minimal"
DEBUG="false"
DEBUG_LIMIT="64"
PARALLEL="false"
SKIP_EXISTING="false"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/run.sh"

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

show_help() {
    head -n 30 "$0" | grep "^#" | sed 's/^# //' | sed 's/^#//'
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --run_name)
            RUN_NAME="$2"
            shift 2
            ;;
        --debug)
            DEBUG="true"
            shift
            ;;
        --debug_limit)
            DEBUG_LIMIT="$2"
            shift 2
            ;;
        --parallel)
            PARALLEL="true"
            shift
            ;;
        --skip_existing)
            SKIP_EXISTING="true"
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

# =============================================================================
# VALIDATION
# =============================================================================

# Check if run.sh exists
if [[ ! -f "$RUN_SCRIPT" ]]; then
    echo "ERROR: run.sh not found at $RUN_SCRIPT"
    exit 1
fi

# Make run.sh executable
chmod +x "$RUN_SCRIPT"

# Validate run_name
case "$RUN_NAME" in
    minimal|full1|full2|broken1|broken2) ;;
    *)
        echo "ERROR: Invalid run_name '$RUN_NAME'. Must be one of: minimal, full1, full2, broken1, broken2"
        exit 1
        ;;
esac

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

# Extract model name from path for logging
get_model_name() {
    local model_path="$1"
    basename "$model_path"
}

# Check if results already exist for a model
has_existing_results() {
    local model_path="$1"
    local model_name
    model_name=$(get_model_name "$model_path")

    # Check for results in the expected output directory
    local output_dir="logs/${model_name}/${RUN_NAME}"
    if [[ "$DEBUG" == "true" ]]; then
        output_dir="${output_dir}_debug"
    fi

    if [[ -d "$output_dir" ]] && [[ -n "$(find "$output_dir" -name "*.json" -type f 2>/dev/null)" ]]; then
        return 0  # Results exist
    else
        return 1  # No results
    fi
}

# Run evaluation for a single model
run_single_model() {
    local model_path="$1"
    local model_name
    model_name=$(get_model_name "$model_path")

    echo "========================================"
    echo "Starting evaluation for: $model_name"
    echo "Model path: $model_path"
    echo "Run name: $RUN_NAME"
    echo "Debug: $DEBUG"
    if [[ "$DEBUG" == "true" ]]; then
        echo "Debug limit: $DEBUG_LIMIT"
    fi
    echo "Timestamp: $(date)"
    echo "========================================"

    # Check if model path exists
    if [[ ! -d "$model_path" ]]; then
        echo "WARNING: Model path does not exist: $model_path"
        echo "Skipping $model_name"
        return 1
    fi

    # Skip if results already exist
    if [[ "$SKIP_EXISTING" == "true" ]] && has_existing_results "$model_path"; then
        echo "Results already exist for $model_name, skipping..."
        return 0
    fi

    # Build command arguments
    local cmd_args=(
        "--model_path" "$model_path"
        "--run_name" "$RUN_NAME"
    )

    if [[ "$DEBUG" == "true" ]]; then
        cmd_args+=("--debug" "--debug_limit" "$DEBUG_LIMIT")
    fi

    # Run the evaluation
    echo "Running: $RUN_SCRIPT ${cmd_args[*]}"
    if "$RUN_SCRIPT" "${cmd_args[@]}"; then
        echo "✅ Successfully completed evaluation for $model_name"
        return 0
    else
        echo "❌ Failed evaluation for $model_name"
        return 1
    fi
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

echo "============================================="
echo "WBL Multi-Model Evaluation"
echo "============================================="
echo "Run name: $RUN_NAME"
echo "Debug mode: $DEBUG"
if [[ "$DEBUG" == "true" ]]; then
    echo "Debug limit: $DEBUG_LIMIT"
fi
echo "Parallel execution: $PARALLEL"
echo "Skip existing: $SKIP_EXISTING"
echo "Number of models: ${#MODELS[@]}"
echo "Models to evaluate:"
for model in "${MODELS[@]}"; do
    echo "  - $(get_model_name "$model")"
done
echo "============================================="

# Track results
declare -a SUCCESSFUL_MODELS=()
declare -a FAILED_MODELS=()
declare -a SKIPPED_MODELS=()

START_TIME=$(date +%s)

if [[ "$PARALLEL" == "true" ]]; then
    echo "Running models in parallel..."

    # Array to store background process PIDs
    declare -a PIDS=()

    # Start all models in background
    for model_path in "${MODELS[@]}"; do
        model_name=$(get_model_name "$model_path")

        # Check if we should skip
        if [[ "$SKIP_EXISTING" == "true" ]] && has_existing_results "$model_path"; then
            echo "Results already exist for $model_name, skipping..."
            SKIPPED_MODELS+=("$model_name")
            continue
        fi

        # Start in background and capture PID
        (
            if run_single_model "$model_path"; then
                echo "PARALLEL_SUCCESS:$model_name" >> /tmp/multi_model_results.$$
            else
                echo "PARALLEL_FAILED:$model_name" >> /tmp/multi_model_results.$$
            fi
        ) &

        PIDS+=($!)
        echo "Started $model_name in background (PID: $!)"
    done

    # Wait for all background processes
    echo "Waiting for all models to complete..."
    for pid in "${PIDS[@]}"; do
        wait "$pid"
    done

    # Process results
    if [[ -f "/tmp/multi_model_results.$$" ]]; then
        while IFS= read -r line; do
            if [[ "$line" == PARALLEL_SUCCESS:* ]]; then
                SUCCESSFUL_MODELS+=("${line#PARALLEL_SUCCESS:}")
            elif [[ "$line" == PARALLEL_FAILED:* ]]; then
                FAILED_MODELS+=("${line#PARALLEL_FAILED:}")
            fi
        done < "/tmp/multi_model_results.$$"
        rm -f "/tmp/multi_model_results.$$"
    fi

else
    echo "Running models sequentially..."

    # Run models one by one
    for model_path in "${MODELS[@]}"; do
        model_name=$(get_model_name "$model_path")

        # Check if we should skip
        if [[ "$SKIP_EXISTING" == "true" ]] && has_existing_results "$model_path"; then
            echo "Results already exist for $model_name, skipping..."
            SKIPPED_MODELS+=("$model_name")
            continue
        fi

        if run_single_model "$model_path"; then
            SUCCESSFUL_MODELS+=("$model_name")
        else
            FAILED_MODELS+=("$model_name")
        fi

        echo ""  # Add spacing between models
    done
fi

# =============================================================================
# SUMMARY
# =============================================================================

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "============================================="
echo "Multi-Model Evaluation Summary"
echo "============================================="
echo "Total runtime: ${DURATION}s ($(date -d@${DURATION} -u +%H:%M:%S))"
echo ""
echo "Successful models (${#SUCCESSFUL_MODELS[@]}):"
for model in "${SUCCESSFUL_MODELS[@]}"; do
    echo "  ✅ $model"
done
echo ""
echo "Failed models (${#FAILED_MODELS[@]}):"
for model in "${FAILED_MODELS[@]}"; do
    echo "  ❌ $model"
done
echo ""
echo "Skipped models (${#SKIPPED_MODELS[@]}):"
for model in "${SKIPPED_MODELS[@]}"; do
    echo "  ⏭️  $model"
done
echo "============================================="

# Exit with error code if any models failed
if [[ ${#FAILED_MODELS[@]} -gt 0 ]]; then
    exit 1
else
    exit 0
fi
