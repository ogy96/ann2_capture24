# CAPTURE-24 — Human Activity Recognition Benchmarks and Temporal Models

This repository contains experiments for **Human Activity Recognition (HAR)** on the  
**CAPTURE-24** wearable accelerometer dataset, following the **official benchmark protocol** and extending it with **temporal deep learning models**.

The work compares:
- classical machine-learning baselines,
- convolutional neural networks (CNN),
- CNN with **Hidden Markov Model (HMM)** temporal smoothing,
- and an **end-to-end CNN–LSTM** architecture.

All experiments follow the **Walmsley et al. (2020) four-class activity taxonomy**.

---

## Repository Structure

```text
ann2_capture24/
│
├── capture24/                 # Core project code
│   ├── prepared_data/         # ❌ Not tracked (raw windowed data)
│   ├── outputs/               # ❌ Not tracked (model outputs, checkpoints)
│   │
│   ├── train_deep_models.py   # Official CNN / CNN+HMM training (Lightning)
│   ├── train_cnn_lstm_simple.py
│   ├── train_cnn_lstm_tuned.py
│   │
│   ├── deep_models.py         # CNN backbone (ResNet-style)
│   ├── hmm.py                 # HMM smoothing + Viterbi decoding
│   ├── augmentation.py        # Data augmentation
│   ├── utils.py               # Metrics, helpers, decoding
│   ├── rf.py / xgb.py         # Classical ML baselines
│   ├── prepare_data.py        # Dataset preprocessing
│   │
│   └── conf/                  # Hydra configuration files
│
├── notebooks/
│   ├── 01_data_sanity_check.ipynb
│   └── conf_matrices.ipynb
│
└── .gitignore
