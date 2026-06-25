# CSCN8020 Assignment 2 — Q-Learning on the Taxi Gymnasium Environment

**Student:** Prajesh Bhatt 
**Course:** CSCN8020 — Reinforcement Learning Programming  
**Institution:** Conestoga College  

---

## Project Summary

This project implements Q-Learning on the Taxi-v4 Gymnasium environment. A taxi agent operates on a 5×5 grid and must learn — purely through environmental interaction and a sparse reward signal — to navigate to a passenger, pick them up, and deliver them to the correct destination. The notebook trains agents across multiple hyperparameter configurations, compares convergence behaviour using rolling average plots, and evaluates final policy quality via greedy rollouts. An object-oriented architecture cleanly separates environment handling, agent logic, training loop, metric collection, plotting, and experiment orchestration.

---

## Repository Structure

```
CSCN8020_Assignment2/
├── CSCN8020_Assignment2.ipynb   # Full solution notebook
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .gitignore                   # Excludes venv, cache, checkpoints
├── logs/
│   └── qlearning_experiments.log  # Generated log file (after running notebook)
└── plots/
    └── *.png                    # Generated plots (after running notebook)
```

---

## Q-Learning Implementation

The agent learns a Q-table Q(s,a) — 500 states × 6 actions = 3,000 entries — initialized to zero. At each step, it selects actions via ε-greedy exploration and updates Q-values using the Bellman TD update:

```
Q(S,A) ← Q(S,A) + α [ R + γ max_a Q(S',a) − Q(S,A) ]
```

Experiments test α ∈ {0.1, 0.01, 0.001, 0.2} and ε ∈ {0.1, 0.2, 0.3} against a baseline of α=0.1, ε=0.1, γ=0.9.

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/<USRID>/CSCN8020_Assignment2.git
cd CSCN8020_Assignment2

# 2. Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate       # Linux/macOS
venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter and run the notebook top-to-bottom
jupyter notebook CSCN8020_Assignment2.ipynb
```


*Reference: Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction (2nd ed.). MIT Press.*
