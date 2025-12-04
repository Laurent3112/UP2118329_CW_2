# UP2118329_CW_2
Here is my coursework 2

# UP2118329 – Coursework 2: Star–Galaxy–QSO Classification

This repository contains my solution to **Coursework 2** for the Data Science / Machine Learning module.

I use a real astronomical dataset of objects observed by the Sloan Digital Sky Survey (SDSS) and model the problem of classifying each object as a **STAR**, **GALAXY**, or **QSO (quasar)**.

The work is organised into three main parts:

- **Q1 – Traditional machine learning model (beginner tutorial)**  
- **Q2 – Neural network model (beginner tutorial)**  
- **Q3 – Research-style investigation** of how using different amounts of training data affects neural network performance.

---

## 1. Dataset

**File:** `data/star_classification.csv`  

This dataset contains photometric and related features for a large number of astronomical objects, together with a class label.

- **Target column**
  - `class`: one of `STAR`, `GALAXY`, or `QSO`.

- **Example feature columns** (may vary slightly depending on the exact file version):
  - `u, g, r, i, z`: magnitude values in the SDSS photometric bands  
  - `redshift`: estimated redshift of the object  
  - `alpha`, `delta`: sky coordinates  
  - Survey metadata such as `run_ID`, `cam_col`, `field_ID`, `plate`, `fiber_ID`, etc.

The goal is to learn from these features to predict the `class` label for each object.

The dataset is stored locally in the `data/` directory and is loaded by helper functions in `py/functions.py`.

---

## 2. Project structure

```text
UP2118329_CW2/
├─ README.md               # This file
├─ dependencies.txt        # List of Python packages and versions
├─ data/
│  └─ star_classification.csv
├─ py/
│  ├─ __init__.py          # Optional convenience file
│  └─ functions.py         # Shared helper functions for loading, preprocessing, etc.
├─ Q1_folder/
│  ├─ Q1.ipynb             # Tutorial: traditional ML baseline
│  └─ README.md
├─ Q2_folder/
│  ├─ Q2.ipynb             # Tutorial: neural network model
│  └─ README.md
└─ Q3_folder/
   ├─ Q3.ipynb             # Research-style study: effect of training data size
   └─ README.md
