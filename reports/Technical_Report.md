# SentinAI - Technical Report

## 1. Executive Summary
Traditional cybersecurity systems rely on signature-based detection, failing against zero-day threats and insider drift. SentinAI proposes a behavioral-based anomaly detection platform powered by Sequence-Aware Deep Learning (GRU/LSTM). 

## 2. Data Engineering & Synthesis
Due to the confidentiality of enterprise logs, we engineered a synthetic generator using Python `Faker` to simulate 30 days of telemetry across Users, Admins, and Devices. 
- **Normal Profile:** Base behavior constrained by working hours, IP geolocations, and typical resources.
- **Injected Anomalies:** Brute Force, Impossible Travel, Credential Stuffing, Lateral Movement, Device Spoofing, Low & Slow Exfiltration, and Insider Drift.
- **Dataset Size:** >320,000 logs containing extreme class imbalance (<3% anomalies).

## 3. Feature Engineering
We extracted temporal, behavioral, and statistical features to capture Concept Drift:
- Rolling window counts (failed logins, unique IPs)
- Time since last login
- Device and Geo mismatch indicators

## 4. Modeling Architecture
The system employs a dual-layered modeling approach:
### 4.1 Baseline Behavior Models
- **Isolation Forest & One-Class SVM:** Provided a baseline for unsupervised outlier detection. IF achieved a PR AUC of 0.06, struggling with the sequential nature of attacks.
### 4.2 Sequence-Aware Deep Learning Models
- We implemented **PyTorch LSTM, GRU, and Transformer Encoders**.
- Models used a dual-head architecture: Head 1 for Binary Anomaly Scoring (BCE Loss) and Head 2 for Attack Classification (Cross-Entropy).
- We dynamically weighted the loss functions to force the network to optimize for minority classes.
- **Results:** The GRU model achieved the highest performance (ROC AUC: 0.82, PR AUC: 0.81).

## 5. Explainability (SHAP)
We integrated `shap.DeepExplainer` to interpret the neural network's decisions, translating black-box predictions into actionable feature importance metrics (e.g., "Impossible Travel" had a 35% impact on the decision).

## 6. Generative AI Response
We implemented the Gemini 1.5 Flash API to automatically parse raw telemetry and SHAP values into an Executive Incident Report, drastically reducing Triage time for SOC analysts.

## 7. Conclusion
SentinAI successfully demonstrates that sequence-aware deep learning, combined with automated explainability, drastically outperforms legacy signature-based detection in identifying novel, low-and-slow cyber attacks.
