# SDXL Adaptive Comic Generation – Group 14

This repository contains the code and measurement files used in our project  
**“An Adaptive Comic Generation Service System Based on SDXL.”**

The project studies latency–quality trade-offs in SDXL generation on a T4 GPU,
and evaluates online scheduling strategies under continuous job arrivals.

---

## 📁 Repository Structure

```

measurement/        # Real SDXL measurements (Fast vs High), sample images, notebooks
simulation/         # M/G/k simulator, SJF scheduler, scaling experiments
README.md           # This file

```

---

## 🔍 Summary of Components

### **1. Measurement (A part)**
- Real inference latency measured on SDXL (Fast & High modes)  
- 50 runs per configuration on Google Colab T4  
- Complex prompt dataset used to evaluate quality differences  
- JSON files contain empirical latency distributions used by the simulator  
- Notebooks show the full measurement and data collection process  

### **2. Simulation & Scheduling (B part)**
- M/G/k queueing model with empirical service times  
- FCFS vs SJF scheduling  
- Multi-server scaling (1/2/4 GPUs)  
- Evaluation of P99 latency, throughput and GPU cost  

---

## 📊 Streamlit Dashboard

Interactive visualization of latency, scaling and cost simulation results:  
**https://sdxl-dashboard-aue98vkh9ub3947vuvqspl.streamlit.app/**

---

## 👥 Authors
Group 14 — MSGAI Course  
- Tech A – Measurement & Engineering  
- Tech B – Simulation  
- Tech C – Report Writing  

---

## 📦 Notes
- Only representative samples of measurement images are included.  
- Full datasets are stored in our shared project drive (available on request).  
```

