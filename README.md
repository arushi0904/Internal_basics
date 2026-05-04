# 🚀 PipelineIQ: Build Time Prediction using MLOps

## 📌 Overview

PipelineIQ is a machine learning project designed to **predict CI/CD build times** based on repository characteristics such as code size, dependencies, and test cases.

This project demonstrates a complete **MLOps pipeline**, including:

* Model training
* Experiment tracking
* Validation
* Deployment simulation
* Monitoring

---

## 🧠 Problem Statement

In modern CI/CD systems, build times can vary significantly depending on project complexity.
This project aims to **predict build time (in minutes)** to help:

* Optimize pipelines
* Reduce delays
* Improve developer productivity

---

## 🏗️ Project Structure

```
MLOPS_Lab_CIE/
│── data/
│   └── training_data.csv
│── src/
│   └── train.py
│── models/
│   ├── SVR.pkl
│   └── RandomForest.pkl
│── results/
│   ├── step1_s1.json
│   ├── step2_s2.json
│   ├── step3_s3.json
│   ├── step4_s4.json
│   └── step5_s5.json
│── mlruns/
│── README.md
```

---

## ⚙️ Technologies Used

* Python 🐍
* Pandas & NumPy
* Scikit-learn
* MLflow (Experiment Tracking)
* Joblib (Model Saving)

---

## 📊 Dataset

The dataset includes features such as:

* `lines_of_code`
* `num_dependencies`
* `test_cases`
* `avg_file_size_kb`

Target variable:

* `build_time_min`

---

## 🚀 How to Run Locally

### 1️⃣ Clone the repository

```
git clone <your-repo-link>
cd MLOPS_Lab_CIE
```

### 2️⃣ Create virtual environment

```
python -m venv .venv
.venv\Scripts\activate
```

### 3️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 4️⃣ Run training script

```
python src/train.py
```

---

## 📈 MLflow Tracking

To visualize experiments:

```
mlflow ui
```

Open in browser:
👉 http://127.0.0.1:5000

---

## 🔄 MLOps Pipeline Steps

### ✅ Step 1: Model Training

* Trained SVR and Random Forest models
* Evaluated using MAE and RMSE

### ✅ Step 2: Validation

* Checked against performance thresholds

### ✅ Step 3: Model Registration

* Best model registered with versioning

### ✅ Step 4: Deployment

* Simulated local API endpoint

### ✅ Step 5: Monitoring

* Tracked latency, drift, and error rate

---

## 🏆 Results Summary

* **Best Model:** Random Forest
* **MAE:** ~0.96
* **RMSE:** ~1.38

The model shows strong performance for predicting build times.

---

## 📌 Future Improvements

* Hyperparameter tuning
* Real-time deployment (FastAPI)
* CI/CD integration
* Drift detection with alerts

---

## 👩‍💻 Author

Arushi

---

## 📜 License

This project is for educational purposes.
