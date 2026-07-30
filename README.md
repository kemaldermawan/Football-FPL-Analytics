# Professional Football Analytics and FPL Decision Engine Hub

## System Overview

This repository houses an end-to-end, high-performance local analytics platform engineered specifically for the Premier League and UEFA Champions League. Designed with computational efficiency in mind, the system executes spatial tactical analysis, machine learning match forecasting, and deterministic mathematical optimization for Fantasy Premier League (FPL) directly on local hardware. The architecture is optimized to run seamlessly on standard environments, ensuring rapid execution without relying on heavy cloud databases.

## 1. Data Engineering and ETL Pipeline

The foundational layer of this system relies on an automated Extract, Transform, and Load (ETL) pipeline to guarantee that tactical models and optimization algorithms utilize the most recent match data.

* Data Extraction
The pipeline interfaces with public statistical repositories (FBref and Understat) using the `soccerdata` wrapper to scrape raw event data, including X and Y coordinates for shots and passes. Simultaneously, it queries the official FPL REST API endpoints to fetch live player valuations, injury statuses, and dynamic gameweek data.

* Data Transformation
Raw JSON and CSV outputs undergo rigorous normalization via `pandas`. Statistical outputs are adjusted to per-90-minute metrics to eliminate play-time bias. Expected Goals (xG) and Expected Assists (xA) variables are cleaned and merged with FPL pricing matrices to create unified player data frames.

* Local Serialization (Load)
To maximize memory efficiency and rapid disk read times, all processed data is serialized into Apache Parquet (`.parquet`) format using `pyarrow`. This approach compresses the database footprint to under 100MB, allowing lightweight code editors to handle the repository smoothly without memory overflow.

## 2. Tactical Spatial Analysis Methodologies

The system moves beyond basic aggregated statistics by quantifying the spatial value of on-ball actions.

* Expected Threat (xT) Grid Modeling
The pitch is segmented into a 16x12 matrix. The model assigns a probability value to each zone representing the likelihood of scoring from that specific area. By analyzing passes and ball carries that move the ball from lower-value zones to higher-value zones, the system calculates a player's progressive tactical contribution prior to a shot being taken.

* Pass Network and Centrality Mapping
Utilizing `mplsoccer` and graph theory, the platform constructs passing networks. Node size is dictated by the total volume of successful passes, while edge thickness represents the frequency of passing combinations between specific player pairs. This visualizes a team's build-up structure and identifies primary playmakers.

* K-Means Statistical Clustering
Unsupervised machine learning (`scikit-learn`) is deployed to group players based on their multi-dimensional statistical profiles. This allows for advanced scouting and similarity searches, identifying lesser-known players who output identical underlying metrics to premium assets.

## 3. Predictive Modeling and Stochastic Simulation

Match forecasting relies on rigorous mathematical probability rather than historical head-to-head biases.

* Poisson Regression Parameters
Based on the foundational framework established by Dixon and Coles (1997), the model calculates Attack Strength and Defense Vulnerability for every team using rolling averages of xG and xGA. These parameters feed into a Poisson distribution to determine the exact probability of specific scorelines.

* Monte Carlo Match Simulations
To account for variance and unpredictability in football, the system executes 10,000 stochastic simulations for each upcoming fixture. This outputs stable, aggregated probabilities for Home Win, Draw, and Away Win, which are then rendered into interactive heatmap matrices.

## 4. FPL Optimization via Operations Research

The Fantasy Premier League decision engine replaces emotional bias with deterministic Operations Research methodologies.

* Mixed-Integer Linear Programming (MILP)
The core of the decision engine utilizes the `PuLP` library to solve multi-objective optimization problems. The objective function maximizes the total Expected Points (xPts) of a 15-man squad over a rolling 5-gameweek horizon.

* Deterministic Constraints
The MILP solver operates under strict FPL rules acting as algebraic constraints. These include the 100.0m budget limit, a maximum of three players per club, specific positional quotas (e.g., exactly two goalkeepers), and point deduction penalties for exceeding free transfers.

* Automated Output Generation
The engine outputs the mathematically optimal starting XI, captaincy choice, vice-captaincy, bench order, and precise transfer sequences required to navigate upcoming fixture difficulties.

## 5. Local Installation and Setup Guide

1. Clone the repository to your local directory:
```bash
git clone [https://github.com/your-username/football-analytics-hub.git](https://github.com/your-username/football-analytics-hub.git)
cd football-analytics-hub
```

2. Establish an isolated virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Execute the ETL data synchronization script:
```bash
python update_engine.py
```

5. Launch the Streamlit dashboard:
```bash
streamlit run app.py
```

The tactical dashboard will initialize and bind to `http://localhost:8501`.

