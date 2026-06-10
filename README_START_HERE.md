# SentinelFlow V3 Final Project Package

**SentinelFlow: Deep Learning-Based Network Intrusion Detection Using Fast Fourier Transform-Enhanced Traffic Profiling**

**Created by Capstone Group 3: Altonaga, Sarceda, Sunga, Torres**

---

## Short Problem Statement

Existing Intrusion Detection Systems (IDS) often rely on time-domain traffic features such as packet counts, byte counts, flow duration, and throughput. These features may miss burst patterns or repeated traffic behavior found in attacks such as Distributed Denial-of-Service (DDoS), brute-force attacks, botnet activity, and port scanning. SentinelFlow demonstrates how Fast Fourier Transform (FFT)-enhanced traffic profiling can be added to a deep learning-based intrusion detection workflow.

---

## What This Package Contains

This package contains three main parts:

1. **Dockerized Jupyter Notebook Pipeline**  
   Used for database preparation, data cleaning, FFT feature extraction, model training, expanded metrics, confusion matrices, and dashboard output.

2. **Django Prototype App**  
   Used for upload, dataset validation, traffic profiling, model status checking, dashboard viewing, and report presentation.

3. **HTML/CSS/JS Presentation Web App**  
   Used for presenting the project concept, methodology, FFT explanation, binary and multiclass classification, metrics, and final project summary.

---

## Folder Structure

```text
sentinelflow_project/
│
├── docker-compose.yml
├── Dockerfile
├── Dockerfile.jupyter
├── Dockerfile.django
├── requirements.txt
├── README_START_HERE.md
├── RUN_ORDER.txt
│
├── notebooks_v3_metrics_pipeline/
│   ├── 00_sentinelflow_environment_check.ipynb
│   ├── 01_sentinelflow_database_preparation.ipynb
│   ├── 02_sentinelflow_data_cleaning_eda.ipynb
│   ├── 03_sentinelflow_fft_feature_extraction.ipynb
│   ├── 04_sentinelflow_deep_learning_models.ipynb
│   └── 05_sentinelflow_results_and_discussion.ipynb
│
├── sentinelflow_web/
│   └── Django prototype app files
│
├── presentation_web_app/
│   ├── index.html
│   ├── css/
│   ├── js/
│   └── assets/
│
├── src/
│   └── sentinelflow_utils.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── database/
│
├── models/
├── outputs/
└── reports/
```

---

# Part 1: Starting the Dockerized Environment

## Step 1: Extract the project folder

Extract the SentinelFlow ZIP file into a clean folder.

Recommended example:

```text
E:\ML Exercises hub\Final Project\sentinelflow_project
```

Do not merge this folder with older SentinelFlow folders. Use a fresh folder to avoid path conflicts.

## Step 2: Open PowerShell in the project folder

Open the folder that contains:

```text
docker-compose.yml
Dockerfile.jupyter
Dockerfile.django
requirements.txt
```

Your PowerShell path should look similar to:

```powershell
PS E:\ML Exercises hub\Final Project\sentinelflow_project>
```

## Step 3: Start Docker Desktop

Make sure Docker Desktop is open and running before using the commands.

## Step 4: Build and run the containers

Run:

```powershell
docker compose down --remove-orphans
docker compose build --no-cache
docker compose up
```

Wait until the terminal shows that Jupyter and Django are running.

---

# Part 2: Opening JupyterLab

Open this link in your browser:

```text
http://127.0.0.1:8899/lab
```

or:

```text
http://localhost:8899/lab
```

This setup uses port `8899` and should not ask for a token or password.

## Verify the environment

Inside JupyterLab, open:

```text
notebooks_v3_metrics_pipeline/00_sentinelflow_environment_check.ipynb
```

Then run:

```text
Kernel > Restart Kernel and Clear Outputs
Run > Run All Cells
```

The notebook should show:

```text
PROJECT_ROOT: /workspace
```

If it shows a Windows path like `E:\ML Exercises hub\...`, the notebook is not running inside Docker.

---

# Part 3: Running the Notebook Pipeline

Run the notebooks in this exact order:

```text
00_sentinelflow_environment_check.ipynb
01_sentinelflow_database_preparation.ipynb
02_sentinelflow_data_cleaning_eda.ipynb
03_sentinelflow_fft_feature_extraction.ipynb
04_sentinelflow_deep_learning_models.ipynb
05_sentinelflow_results_and_discussion.ipynb
```

For each notebook:

```text
Kernel > Restart Kernel and Clear Outputs
Run > Run All Cells
```

## Notebook 00: Environment Check

Confirms Python, Docker pathing, PyTorch, and required libraries.

## Notebook 01: Database Preparation

Prepares the dataset for faster processing. It may create:

```text
data/processed/
data/database/
outputs/
reports/
```

## Notebook 02: Data Cleaning and EDA

Checks missing values, infinite values, class distribution, attack distribution, and basic NetFlow patterns.

## Notebook 03: FFT Feature Extraction

Converts traffic into time-windowed signals and extracts FFT-based features such as spectral energy, dominant frequency, entropy, burstiness, and periodicity.

## Notebook 04: Deep Learning Models

Trains and compares binary and multiclass models.

Expected model types may include:

```text
MLP
CNN-1D
LSTM
GRU
Transformer Encoder
```

## Notebook 05: Results and Discussion

Generates the final metrics and dashboard outputs.

Expected outputs include:

```text
outputs/03_model_results.csv
outputs/metrics/04_model_results_expanded.csv
reports/sentinelflow_v3_results_dashboard.html
reports/confusion_matrices/
reports/classification_reports/
```

---

# Part 4: Using the Full Dataset

Place the full NetFlow dataset inside:

```text
data/raw/
```

Example:

```text
data/raw/NF-UQ-NIDS-v2.csv
```

Then rerun notebooks `01` to `05`.

The bundled sample data is only for checking if the pipeline works. Use the full dataset for the actual final project output.

---

# Part 5: Starting the Django Prototype App

The Django app runs inside the same Docker Compose setup.

Open this link in your browser:

```text
http://127.0.0.1:8000
```

or:

```text
http://localhost:8000
```

## What the Django app does

The Django app supports:

```text
Dataset upload
Data validation
NetFlow cleaning summary
FFT traffic profiling summary
Binary and multiclass dashboard viewing
Notebook model status checking
Report viewing
```

## Important note about models

The Django app is a prototype demonstration interface. Model training should still happen inside the notebooks.

The app checks for expected model files inside:

```text
models/
```

Expected files may include:

```text
binary_model.pt
multiclass_model.pt
binary_scaler.pkl
multiclass_scaler.pkl
binary_feature_list.json
multiclass_feature_list.json
binary_label_encoder.pkl
multiclass_label_encoder.pkl
```

If these files are missing, the app will still run and show upload profiling, dataset validation, notebook metrics, and available reports. Run the notebooks first if model files are needed.

## Using the upload page

1. Open the Django app at `http://127.0.0.1:8000`.
2. Click **Upload**.
3. Choose a `.csv`, `.tsv`, `.txt`, `.parquet`, or `.pq` NetFlow file.
4. Click **Analyze Upload**.
5. Review row count, class distribution, signal windows, baseline segments, and FFT-enhanced segments.

## Using the dashboard page

1. Run the notebook pipeline first.
2. Open the Django app.
3. Click **Dashboard**.
4. Review available notebook metrics and upload analysis.

## Using the model status page

1. Open **Models**.
2. Check whether binary and multiclass model files exist in the `models/` folder.
3. If files are missing, rerun the model training notebook.

## Using the reports page

1. Run Notebook 05.
2. Open **Reports** in the Django app.
3. Review available generated HTML reports.

---

# Part 6: Opening the HTML/CSS/JS Presentation Web App

The presentation web app is separate from Django. It is for consultation, defense, or final presentation.

## Option A: Open directly

Open:

```text
presentation_web_app/index.html
```

You can double-click the file.

## Option B: Open using a local server

From the project folder, run:

```powershell
cd presentation_web_app
python -m http.server 5500
```

Then open:

```text
http://127.0.0.1:5500
```

## What the presentation app includes

```text
Project overview
Problem statement
NetFlow explanation
FFT traffic profiling explanation
Binary classification explanation
Multiclass classification explanation
Metrics explanation
Confusion matrix explanation
SentinelFlow architecture
Final project summary
```

The presentation app uses a cybersecurity-themed dark interface with packet-flow style visuals and an interactive FFT mini-animation.

---

# Part 7: Common Error Checks

## Error: Jupyter asks for token

Use:

```text
http://127.0.0.1:8899/lab
```

Do not use port `8888`.

If the browser still asks for a token, open an incognito/private window and use the same link.

## Error: Notebook shows Windows path

If the notebook shows:

```text
E:\ML Exercises hub\...
```

instead of:

```text
/workspace
```

then it is not running inside Docker. Go back to:

```text
http://127.0.0.1:8899/lab
```

and run the notebooks there.

## Error: PyTorch unavailable

Run the environment check notebook first.

The output should confirm that PyTorch is installed and working.

## Error: Dataset not found

Make sure the dataset is inside:

```text
data/raw/
```

Then rerun the database preparation notebook.

## Error: Perfect 1.000 model results

This may mean the sample dataset is too small or too easy.

Use the full dataset for more meaningful final project results.

## Error: Django cannot find model files

Run the notebook training pipeline first and make sure the model files exist inside:

```text
models/
```

The app will still run even if model files are missing, but prediction-specific model loading will require exported model files.

## Error: HTML dashboard does not update

Rerun:

```text
05_sentinelflow_results_and_discussion.ipynb
```

Then reopen:

```text
reports/sentinelflow_v3_results_dashboard.html
```

---

# Acronym Guide

**IDS** means **Intrusion Detection System**.  
**FFT** means **Fast Fourier Transform**.  
**DDoS** means **Distributed Denial-of-Service**.  
**DoS** means **Denial-of-Service**.  
**MLP** means **Multilayer Perceptron**.  
**CNN** means **Convolutional Neural Network**.  
**CNN-1D** means **One-Dimensional Convolutional Neural Network**.  
**LSTM** means **Long Short-Term Memory**.  
**GRU** means **Gated Recurrent Unit**.  
**ROC-AUC** means **Receiver Operating Characteristic Area Under the Curve**.  
**PR-AUC** means **Precision-Recall Area Under the Curve**.  
**NetFlow** means summarized network flow records.
