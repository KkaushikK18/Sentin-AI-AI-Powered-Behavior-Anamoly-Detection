# SentinAI: AI-Powered Behavioral Anomaly Detection
## Pitch Deck / Presentation Slides

---

### Slide 1: The Problem
**Signature-Based Security is Failing**
- Current SIEMs rely on static rules.
- They completely miss Zero-Day attacks, Insider Threats, and Low-and-Slow Exfiltration.
- When they do alert, SOC analysts are blinded by "Alert Fatigue" (thousands of false positives).

---

### Slide 2: The Solution - SentinAI
**Learning Normal to Detect the Abnormal**
- Instead of memorizing malware signatures, SentinAI uses Deep Learning to memorize the *baseline behavior* of every User, Server, and Device.
- We utilize Sequence-Aware AI to look at the *context* of events over time, not just in isolation.

---

### Slide 3: Data & Feature Engineering
**Synthesizing the Enterprise**
- Built a robust data generator to simulate 30 days of telemetry across 320,000+ events.
- Injected 7 advanced Persistent Threat (APT) vectors (e.g., Credential Stuffing, Impossible Travel).
- Extracted temporal features (rolling windows) to handle *Concept Drift* (remote work shifts).

---

### Slide 4: The AI Architecture
**Dual-Headed PyTorch Sequence Models**
- We trained and evaluated Isolation Forests, LSTMs, GRUs, and Transformers.
- **The Winner:** Gated Recurrent Unit (GRU). 
- It simultaneously outputs a 0-100 Risk Score AND a Multi-Class Attack Classification.
- **Metrics:** Achieved an **0.81 PR AUC**, mathematically proving its ability to find the 2% of anomalies in a sea of normal traffic without false positives.

---

### Slide 5: Radical Transparency
**Explainable AI & GenAI**
- Security teams don't trust black boxes.
- We implemented **SHAP (DeepExplainer)** to mathematically prove *why* the neural network flagged an event.
- **GenAI Integration:** We hooked up Gemini 1.5 to automatically read those SHAP values and instantly write an Executive Incident Report and SOAR playbook.

---

### Slide 6: Live Demo
**The Security Operations Center (SOC) Dashboard**
- Show the Streamlit UI.
- Show the Live Event hunting.
- Click "Generate AI Report" on a flagged entity.
