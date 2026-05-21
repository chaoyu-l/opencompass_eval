#!/usr/bin/env bash
# =============================================================================
# install.sh — one-shot environment setup for opencompass_eval.
#
# Creates the conda env, installs every dependency, and fetches the nltk data
# IFEval needs. Run this once; afterwards just run the four run_*.sh scripts.
#
# Usage:   ./install.sh
# Override the env name:   CONDA_ENV=myenv ./install.sh
# =============================================================================
set -eo pipefail

CONDA_ENV="${CONDA_ENV:-opencompass}"
PYTHON_VERSION="3.10"

# run from the repo root (this script lives there)
cd "$(dirname "${BASH_SOURCE[0]}")"

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: 'conda' not found on PATH. Install Miniconda/Anaconda first." >&2
    exit 1
fi
if [ ! -f requirements_trace.txt ]; then
    echo "ERROR: requirements_trace.txt not found — run this from the repo root." >&2
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

# --- 1/4 · conda env (skip creation if it already exists) --------------------
if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
    echo "[1/4] conda env '$CONDA_ENV' already exists — reusing it."
else
    echo "[1/4] creating conda env '$CONDA_ENV' (python $PYTHON_VERSION) ..."
    conda create -n "$CONDA_ENV" python="$PYTHON_VERSION" -y
fi
conda activate "$CONDA_ENV"

# --- 2/4 · python dependencies -----------------------------------------------
echo "[2/4] installing requirements_trace.txt ..."
pip install -r requirements_trace.txt

# --- 3/4 · this package (editable) -------------------------------------------
echo "[3/4] installing the opencompass package (editable) ..."
pip install -e .
# mmengine resolves `opencompass.configs.*` (the lazy imports in the eval
# scripts) by locating the package dir inside site-packages. A PEP 660 editable
# install does not create it there, so symlink it — otherwise config loading
# fails with "opencompass/configs/.../<name>.py not found".
SITE_PKGS="$(python -c 'import site; print(site.getsitepackages()[0])')"
ln -sfn "$(pwd)/opencompass" "$SITE_PKGS/opencompass"
echo "      linked $SITE_PKGS/opencompass -> $(pwd)/opencompass"

# --- 4/4 · nltk data for IFEval ----------------------------------------------
echo "[4/4] downloading nltk data (punkt) for IFEval ..."
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

echo ""
echo "=================================================================="
echo " Done — conda env '$CONDA_ENV' is ready."
echo " Next: edit the SETTINGS block (model paths) at the top of each"
echo " run_*.sh script, then run it — no arguments needed:"
echo "   ./run_dream_7b_general.sh"
echo "   ./run_llada_8b_general.sh"
echo "   ./run_llama3_8b_general.sh"
echo "   ./run_qwen25_7b_general.sh"
echo "=================================================================="
