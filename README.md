# ST-InducedTrans: Cellular Traffic Forecasting & Autonomous RAN Optimization

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

> **A linear-complexity Spatio-Temporal Bottleneck Transformer ($\mathcal{O}(NK)$) with Uncertainty Quantification for proactive 5G/6G Radio Access Network (RAN) slicing, dynamic PRB allocation, and green energy sleep-mode optimization.**

---

## 📌 Executive Summary

Modern 5G-Advanced and emerging 6G cellular networks experience extreme, non-linear traffic fluctuations across concurrent service slices: **Enhanced Mobile Broadband (eMBB / Internet)**, **Voice Calls**, and **SMS**. 

Legacy cellular network management relies on **reactive heuristics** (allocating resources based on past hours or static margins), which leads to:
1. **Severe Resource Over-Provisioning:** Significant compute and radio-frequency (RF) power wasted during low-traffic hours.
2. **Under-Provisioning & SLA Violations:** Heavy Physical Resource Block (PRB) starvation, buffer bloat, and dropped connections during sudden spikes.

**ST-InducedTrans** solves this challenge by shifting network operations from reactive troubleshooting to **proactive, closed-loop autonomous orchestration**. By combining **linear-complexity cross-attention** with **Gaussian Negative Log-Likelihood (NLL) uncertainty quantification**, the system provides calibrated forward-looking telemetry to prevent network congestion before it occurs.

---

## 🌟 Key Technical Innovations

1. **Spatial Bottleneck Transformer (SBT) with Linear Complexity $\mathcal{O}(NK)$:**
   * Standard multi-head self-attention scales quadratically ($\mathcal{O}(N^2)$), causing memory exhaustion on urban base station grids ($N=10,000$ cells in Milan scale).
   * SBT introduces $K$ learnable inducing points ($K \ll N$) acting as a latent spatial bottleneck:
     * **Compression:** Latent points query the spatial grid via Cross-Attention ($\mathcal{O}(KND)$).
     * **Correlation Retrieval:** Grid cells query the compressed latents ($\mathcal{O}(NKD)$).
   * **Total complexity is strictly linear $\mathcal{O}(NK)$**, dropping per-sample inference latency to **~2.6 ms**.

2. **Dual-Branch Temporal Attention & Explicit Calendar Clues:**
   * **Closeness Branch:** Captures immediate short-term momentum over consecutive preceding hours ($t-2, t-1, t$).
   * **Periodicity Branch:** Models daily circadian rhythms across preceding days ($t - 24, t - 48, t - 72\text{h}$).
   * **Calendar Encodings:** Injects cyclical continuous time features ($\sin/\cos$ hour of day, day of week, and holiday indicators) to account for regional workweek dynamics.

3. **Risk-Aware Uncertainty Quantification (UQ):**
   * Employs a dual-head projection yielding predictive mean $\mu$ and log-variance $\log \sigma^2$.
   * Optimized with a multi-task loss combining **Gaussian Negative Log-Likelihood (NLL)** and **Mean Squared Error (MSE)**.
   * Delivers rigorous $\pm 2\sigma$ confidence intervals with **100% empirical coverage** on unseen test evaluations.

4. **Closed-Loop Digital Twin Simulation:**
   * Replaces reactive allocation ($t-1$) with proactive $\mu + 1\sigma$ PRB allocation, slashing buffer overflow SLA violations from **59.64% down to 6.28%**.
   * Implements an **Uncertainty-Gated Green Sleep Mode**, keeping micro-cells awake unless $\mu + 2\sigma < \text{Threshold}_{\text{sleep}}$, guaranteeing **0.00% SLA violations**.

---

## 📊 Visual Highlights & Experimental Results

### 1. Spatial Topology & Multi-Service Dynamics
Realistic base station spatial clustering (modeled after Cairo/Giza regional telemetry and OpenCelliD) paired with multi-service temporal traffic:
![Spatial Topology & Multi-Service Traffic](assets/topology_and_traffic.png)

---

### 2. City-Wide Spatial Heatmaps & Prediction Error
Side-by-side comparison of Ground Truth, ST-InducedTrans predicted mean, ConvLSTM baseline, and absolute residual errors during peak evening hours:
![City Heatmaps](assets/spatial_heatmaps.png)

---

### 3. Uncertainty Quantification (UQ) 24-Hour Confidence Envelopes
Consecutive 24-hour test horizon demonstrating true load bounded within the predicted $\pm 2\sigma$ uncertainty envelope (95% theoretical, 100% empirical coverage):
![UQ Confidence Envelopes](assets/uq_confidence_envelope.png)

---

### 4. Attention Interpretability: Discovered Urban Clusters
Projecting spatial attention weights from the $K=16$ learned inducing points back onto the geographic grid reveals functional urban archetypes (commercial centers, transit hubs, and residential zones):
![Inducing Attention Clusters](assets/inducing_attention_clusters.png)

---

### 5. Downstream Digital Twin & Green Energy Sleep-Mode
Simulating proactive vs. reactive PRB allocation and uncertainty-gated micro-cell sleep mode:
![Digital Twin Simulation](assets/digital_twin_simulation.png)

---

## 📈 Benchmark Comparison

Evaluated on unseen chronological test sets using physical traffic scales:

| Model | Complexity Per Layer | Internet $R^2$ | Voice $R^2$ | SMS $R^2$ | Mean $R^2$ | Composite NRMSE | MAE | Latency (ms/sample) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Daily-Period Avg (DPA)** | $\mathcal{O}(1)$ | 0.9495 | 0.8609 | 0.7822 | 0.8642 | 0.0280 | 1.69 | **0.020 ms** |
| **Cell-wise LSTM** | Recurrent $\mathcal{O}(T)$ | 0.8775 | 0.8638 | 0.8096 | 0.8503 | 0.0316 | 2.32 | 0.381 ms |
| **ConvLSTM2D** | Recurrent + 2D CNN | 0.9212 | 0.8846 | 0.8219 | 0.8759 | 0.0279 | 2.09 | 2.640 ms |
| **ST-InducedTrans (Ours)** | **Linear $\mathcal{O}(NK)$** | **0.9107** | **0.9456** | **0.8505** | **0.9023** | **0.0238** | **1.83** | **2.627 ms** |

### Operational RAN Digital Twin Metrics (First 72 Test Hours $\times$ 400 Cells)
* **Reactive PRB Allocation SLA Violations:** `59.64%` (severe packet buffer overflow during demand ramps)
* **Proactive ST-InducedTrans Allocation SLA Violations:** `6.28%` (**~89.5% relative drop in violations**)
* **Reactive Sleep Mode SLA Violations:** `3.02%` (premature sleep causes dropped calls)
* **UQ-Gated Sleep Mode SLA Violations:** `0.00%` (**perfect safety guarantee**)

---

## 📁 Repository Structure

```text
├── assets/                                 # Visualizations, error maps, and benchmark curves
│   ├── topology_and_traffic.png
│   ├── training_curves.png
│   ├── benchmark_comparison.png
│   ├── spatial_heatmaps.png
│   ├── uq_confidence_envelope.png
│   ├── inducing_attention_clusters.png
│   └── digital_twin_simulation.png
├── docs/                                   # Detailed system specifications and mathematics
│   ├── problem_formulation.md              # Mathematical formulation and domain equations
│   └── implementation_blueprint.md         # Step-by-step architecture blueprint
├── milan_data/                             # Telecom Italia Milan CDR benchmark cache
│   └── milan_hourly_cache_dall.npz
├── cellular_load_forecasting_st_inducedtrans.ipynb  # Master Jupyter / Google Colab notebook
├── requirements.txt                        # Python dependencies
├── .gitignore                              # Git exclusion rules
├── LICENSE                                 # MIT License
└── README.md                               # Project documentation
```

---

## 🚀 Quickstart & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/cellular-load-forecasting-st-inducedtrans.git
cd cellular-load-forecasting-st-inducedtrans
```

### 2. Set Up Python Environment
```bash
# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run in Jupyter or Google Colab
Launch the master notebook:
```bash
jupyter notebook cellular_load_forecasting_st_inducedtrans.ipynb
```
* Or upload `cellular_load_forecasting_st_inducedtrans.ipynb` to [Google Colab](https://colab.research.google.com/) with GPU acceleration enabled (`Runtime > Change runtime type > T4 GPU`).

---

## ⚙️ Configuration & Datasets

All hyperparameters and dataset settings are centralized in a strongly-typed `Config` dataclass inside the notebook:

```python
@dataclass
class Config:
    # Spatial Topology
    grid_h: int = 20                 # 20x20 prototyping (400 cells); 100x100 for Milan
    grid_w: int = 20
    num_services: int = 3            # Internet (eMBB), Voice, SMS

    # Temporal Windows
    c_closeness: int = 3             # Immediate preceding hours [t-2, t-1, t]
    p_period: int = 3                # Preceding daily periods [t-72h, t-48h, t-24h]
    target_horizon: int = 3          # Future forecast horizon [t+1, t+2, t+3]

    # ST-InducedTrans Hyperparameters
    k_inducing: int = 16             # Number of learnable inducing points (K << N)
    d_model: int = 64                # Representation dimension
    n_heads: int = 4                 # Attention heads
    
    # Dataset Mode
    dataset_mode: str = "cairo"      # "cairo" (synthetic topology) | "milan" (real CDR)
```

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

## 🤝 Contributing & Feedback
Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/3bdelhadYy/cellular-load-forecasting-st-inducedtrans/issues).
