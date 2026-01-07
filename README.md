# UP2118329_CW_2
Here is my coursework 2

# UP2118329 – Coursework 2: Star–Galaxy–QSO Classification

This repository contains my submission for Coursework 2 for the Data Science / Machine Learning module.

he coursework is based on a real astronomical dataset from the Sloan Digital Sky Survey (SDSS) and addresses the supervised classification problem of identifying whether an observed object is a **STAR**, **GALAXY**, or **QSO (quasar)** from its measured features.

The work is organised into three main parts:

- **Q1 – Traditional machine learning model (beginner tutorial)**
 A tutorial-style notebook implementing a standard multiclass classifier and establishing a performance baseline.
- **Q2 – Neural network model (beginner tutorial)**
A PyTorch-based neural network trained on the same dataset and preprocessing pipeline to enable direct comparison with Q1.
- **Q3 – Research-style investigation**
An experimental study of how neural network performance varies with the size of the training dataset.

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

The aim is to learn from these features to predict the `class` label for each object.

The dataset is stored locally in the `data/` directory and is loaded by helper functions in `py/functions.py`.

---

## 2. Project structure

```text
UP2118329_CW2/
├─ README.md               # Main project overview (this file)
├─ dependencies.txt        # Python dependencies
├─ data/
│  └─ star_classification.csv
├─ py/
│  ├─ __init__.py
│  └─ functions.py         # Shared helper functions (loading, preprocessing, splitting)
├─ Q1_folder/
│  ├─ Q1.ipynb             # Traditional machine learning baseline
│  └─ README.md
├─ Q2_folder/
│  ├─ Q2.ipynb             # Neural network classifier (PyTorch)
│  └─ README.md
└─ Q3_folder/
   ├─ Q3.ipynb             # Effect of training data size on performance
   └─ README.md

3. Summary of methodology

A consistent preprocessing pipeline is used throughout the coursework, including stratified data splitting, feature standardisation, and label encoding.

Q1 establishes a traditional machine learning baseline.

Q2 trains a feed-forward neural network using PyTorch.

Q3 reuses the neural network architecture to investigate how classification accuracy depends on training dataset size.

This design allows controlled and fair comparison between approaches.

4. How to run the coursework

Install the required packages:

pip install -r dependencies.txt


Ensure data/star_classification.csv is present in the data/ directory.

Open and run the notebooks:

Q1_folder/Q1.ipynb

Q2_folder/Q2.ipynb

Q3_folder/Q3.ipynb

Each notebook can be run independently.

5. Reproducibility

Fixed random seeds are used where appropriate.

The same dataset splits and pre-processing pipeline are reused across questions.

All results and figures are generated within the notebooks.



