#!/usr/bin/env bash
# =============================================================================
# run_dream_7b_general.sh — evaluate DREAM-7B (base + 8 LoRA) on the full
# MMLU / BBH / HumanEval / IFEval suite.
#
# HOW TO USE:  edit the SETTINGS block below, then just run:
#                  ./run_dream_7b_general.sh
#              (optional run.py flags still work, e.g. ./run_dream_7b_general.sh -r latest)
# =============================================================================
set -eo pipefail

# ========================= SETTINGS — EDIT THESE =============================
BASE_MODEL_PATH="/media/chaoyu/BCF8E25947FDB178/扩散大语言模型实验/Dream-7b"
LORA_PATH="/media/chaoyu/BCF8E25947FDB178/打包实验/Trace_Results/outputs_dream_7b_500/lora_7b"
#   LORA_PATH must contain 8 adapter subfolders 0/ .. 7/.
#   Set LORA_PATH="" to evaluate the base model only.
BATCH_SIZE=16              # inference batch size (H200 can go to 32-64)
CONDA_ENV="opencompass"    # conda env created by install.sh
# =============================================================================

EVAL_SCRIPT="eval_dream_7b_general.py"
cd "$(dirname "${BASH_SOURCE[0]}")"

# --- validate paths ----------------------------------------------------------
if [ ! -d "$BASE_MODEL_PATH" ]; then
    echo "ERROR: BASE_MODEL_PATH does not exist:" >&2
    echo "  '$BASE_MODEL_PATH'" >&2
    echo "Fix it in the SETTINGS block at the top of this script." >&2
    exit 1
fi
if [ -n "$LORA_PATH" ] && [ ! -d "$LORA_PATH" ]; then
    echo "ERROR: LORA_PATH does not exist:" >&2
    echo "  '$LORA_PATH'" >&2
    echo "Fix it in the SETTINGS block, or set LORA_PATH=\"\" for base-only." >&2
    exit 1
fi

# --- activate the conda env --------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: 'conda' not found on PATH." >&2
    exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# --- summary -----------------------------------------------------------------
echo "=================================================================="
echo " eval script : $EVAL_SCRIPT"
echo " base model  : $BASE_MODEL_PATH"
if [ -n "$LORA_PATH" ]; then
    echo " lora path   : $LORA_PATH"
    echo "               (base model + 8 LoRA variants)"
else
    echo " lora path   : (none — base model only)"
fi
echo " batch size  : $BATCH_SIZE"
echo " conda env   : $CONDA_ENV"
echo "=================================================================="

# --- run ---------------------------------------------------------------------
export BASE_MODEL_PATH
export EVAL_BATCH_SIZE="$BATCH_SIZE"
if [ -n "$LORA_PATH" ]; then
    export LORA_PATH
fi
exec python run.py "$EVAL_SCRIPT" "$@"
