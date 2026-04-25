# GraOmicSynergy

## Setup

Use Python `3.12.3` and install the project dependencies from `requirements.txt`.

```bash
python3.12 --version
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Resources

+ `process_split_data.ipynb`: builds the split files in `data/split_data/`
+ `create_data.ipynb`: builds the processed `.pkl` files used for training
+ `utils.py`: shared dataset and training helpers
+ `pipeline/`: split and feature-loading helpers
+ `models/gin_add_at_rmse.ipynb`: regression training
+ `models/gin_add_atnew_rmse.ipynb`: regression variant
+ `models/gin_add_at_cls.ipynb`: classification training

Gene expression data can be downloaded from:
https://drive.google.com/file/d/1orMkX_4xC6N8fah98nUFJSuZueHOadX4/view?usp=sharing

## Workflow

### 1. Process and split the data

Run `process_split_data.ipynb` first.

This creates the split files under `data/split_data/`.

### 2. Build the training data
Run `create_data.ipynb` after the split files exist.

This notebook reads the split files, loads the drug graphs and omics features, and writes the processed `.pkl` files used by the training notebooks.

### 3. Train the models

Use the notebooks in `models/` depending on the task you want to run:

+ Regression: `models/gin_add_at_rmse.ipynb`
+ Regression variant: `models/gin_add_atnew_rmse.ipynb`
+ Classification: `models/gin_add_at_cls.ipynb`

These notebooks read the processed `.pkl` files from `data/split_data/.../processed/` and write logs, metrics, and plots under `model/save_model/...`.

## Notes

+ `process_split_data.ipynb` and `create_data.ipynb` are setup notebooks.
+ Once the `.pkl` files exist, you can rerun the model notebooks without rebuilding the data.
