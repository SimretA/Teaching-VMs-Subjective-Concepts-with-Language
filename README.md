# Teaching VMs Subjective Concepts with Language

## Setup

### 1. Create Virtual Environment

```bash
python -m venv .venv
```

Activate the virtual environment:

- Windows: `.venv\Scripts\activate`
- Linux/Mac: `source .venv/bin/activate`

### 2. Install Dependencies with uv

Install [uv](https://docs.astral.sh/uv/) if you haven't already:

```bash
pip install uv
```

Sync dependencies based on your hardware:

**For CPU-only:**

```bash
uv sync --extra cpu
```

**For CUDA 12.8:**

```bash
uv sync --extra cu128
```

Note: The `cpu` and `cu128` extras are mutually exclusive (you can only use one).

### 3. Download Datasets

Create a `.env` file in the project root and add your Kaggle API token:

```env
KAGGLE_API_TOKEN=your_token_here
```

Run the download script:

```bash
# Download all datasets
python scripts/load_datasets.py

# Download specific dataset
python scripts/load_datasets.py --dataset celeba
python scripts/load_datasets.py --dataset flickr
```

Datasets will be saved to the `data/` directory.
