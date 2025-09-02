> **Disclaimer:**  
> This project was developed as part of a **University of Liverpool** assignment.  
> It is for **educational purposes only** and must not be copied, reused, or distributed without permission, in accordance with the University's academic integrity policies.

# Multi-Armed Bandit (ε-Greedy) — NumPy & Matplotlib

A Python script that simulates the **k-armed bandit** problem with an **ε-greedy** agent, compares different exploration rates, and visualises learning via **average reward** and **% optimal action** over time.

---

## What it does
- **Problem setup**
  - Creates a bandit with **k arms** whose true values are drawn from `N(0, 1)`; rewards are sampled as `N(q*, 1)` for the chosen arm.
- **Agent**
  - Uses **ε-greedy** action selection (explore with probability ε, otherwise exploit the current best estimate).
  - Updates action values by the **sample-average** rule:  
    `Q(n+1) = Q(n) + (R − Q(n)) / N`
- **Experiments**
  - Runs multiple **independent runs** and averages results for several ε values (default `ε ∈ {0, 0.01, 0.1}`) across a number of **time steps** (default `plays = 1000`).
- **Visualisation**
  - Plots:
    - **Average reward vs. steps** for each ε
    - **% optimal action vs. steps** for each ε

---

## Files
- `multi-armed bandit.py` — main script defining the `MultiArmedBandit` class, experiment loop, and plotting utilities; executes an experiment when run.

---

## How to Run
1. **Clone or download the project**
   ```bash
   git clone https://github.com/your-username/mab-epsilon-greedy.git
   cd mab-epsilon-greedy
