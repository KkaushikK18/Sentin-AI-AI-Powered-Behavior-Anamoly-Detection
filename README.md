# AI-Powered Behavioral Anomaly Detection for Cybersecurity

Welcome to **SentinAI**, an advanced, sequence-aware cybersecurity anomaly detection platform. 

Traditional cybersecurity systems rely heavily on predefined rules and signature-based detection. These systems fail against Zero-day attacks, slow data exfiltration, and credential compromise. 

SentinAI solves this by learning the **normal behavioral baseline** of every entity (User, Service Account, IoT Device, etc.) and detecting complex temporal deviations using Deep Learning (LSTMs) and Explainable AI (SHAP).

## Key Features
- **Synthetic Enterprise Data Generator**: Simulates realistic logs representing years of user interactions across diverse entities and attack vectors (Extreme Class Imbalance <2%).
- **Advanced Feature Engineering**: Extracts temporal windows, session statistics, geo-velocities (Impossible Travel), and behavioral shifts.
- **Multi-Task PyTorch LSTM**: Evaluates the *sequence* of events to output both an Anomaly Risk Score (0-100) and an Attack Type Classification.
- **Explainable AI**: Integrates SHAP values to explain *why* an event was flagged.
- **Premium Hackathon Dashboard**: A glassmorphism, dark-themed Streamlit UI for threat hunting and entity investigation.

## Project Architecture
```text
.
├── backend/            # Future API extensions
├── config/             # YAML configurations
├── dashboard/          # Streamlit UI (app.py)
├── data/               # Generators and synthetic datasets
├── models/             # PyTorch LSTMs and Sklearn Baselines
├── reports/            # Technical reports, slides, and evaluation metrics
├── saved_models/       # Serialized .pt and .pkl models
├── tests/              # Pytest suites
├── training/           # Sequence model PyTorch training loops
└── utils/              # Helper functions
```

## Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Generate Data & Features
```bash
# Generates 30 days of simulated logs (~300k+ events)
python data/data_generator.py

# Extracts temporal/behavioral features and sequence IDs
python data/feature_engineering.py
```

### 3. Train Models
```bash
# Trains Baseline Unsupervised Models (Isolation Forest, OCSVM)
python models/baseline_models.py

# Trains the PyTorch Multi-Task LSTM
python -m training.train_sequence_model
```

### 4. Launch the Dashboard
```bash
streamlit run dashboard/app.py
```

## Evaluation Metrics
Our models were heavily evaluated for severe class imbalance. While traditional baselines (Isolation Forest) struggled with temporal anomalies (PR AUC ~0.06), our Sequence-Aware LSTM significantly improved detection capabilities by understanding the context of historical actions.

## Built With
- **Python 3.10+**
- **PyTorch** (Deep Learning)
- **Scikit-Learn** (Baselines & Preprocessing)
- **Streamlit** & **Plotly** (Dashboard)
- **SHAP** (Explainability)
- **Faker** & **Pandas** (Data Synthesis)

---
*Developed for AI/ML Hackathon.*
