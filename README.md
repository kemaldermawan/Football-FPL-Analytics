# Professional Football Analytics and FPL Decision Engine Hub

## System Overview

This repository houses an end-to-end, high-performance local analytics platform engineered specifically for the Premier League and UEFA Champions League. Designed with computational efficiency in mind, the system executes spatial tactical analysis, machine learning match forecasting, and deterministic mathematical optimization for Fantasy Premier League directly on local hardware. The architecture is optimized to run seamlessly on standard environments, ensuring rapid execution without relying on heavy cloud databases.

## 1. Data Engineering and ETL Pipeline

The foundational layer of this system relies on an automated Extract, Transform, and Load pipeline to guarantee that tactical models and optimization algorithms utilize the most recent match data.

* Data Extraction
The pipeline interfaces with public statistical repositories using wrappers to scrape raw event data, including coordinates for shots and passes. Simultaneously, it queries the official FPL REST API endpoints to fetch live player valuations, injury statuses, and dynamic gameweek data.

* Data Transformation
Raw outputs undergo rigorous normalization via pandas. Statistical outputs are adjusted to per-90-minute metrics to eliminate play-time bias. Expected Goals and Expected Assists variables are cleaned and merged with FPL pricing matrices to create unified player data frames.

* Local Serialization
To maximize memory efficiency and rapid disk read times, all processed data is serialized into Apache Parquet format using pyarrow. This approach compresses the database footprint to under 100 megabytes, allowing lightweight code editors to handle the repository smoothly without memory overflow.

## 2. FPL Decision Engine Core Modules

The Fantasy Premier League decision engine replaces emotional bias with deterministic Operations Research and advanced statistical modeling.

* Market Analysis and VORP
Calculates Value Over Replacement Player metrics per position using baseline costs and performance projections, filtering out small-sample outliers through customizable minutes thresholds.

* Advanced Fixture Matrix (Custom FDR)
A logarithmic Dixon-Coles difficulty matrix mapping opponent strength versus team vulnerability. Features dynamic modulators for home advantage, European club congestion, and key player absences across attack and defense modules.

* Custom Expected Points Engine
Deterministic xPts projections utilizing underlying expected goal metrics modulated by dual-dimension Dixon-Coles matrices, player availability probabilities, and expected minutes factors.

* MILP Squad Optimizer
Executes Mixed-Integer Linear Programming via PuLP to generate the mathematically optimal 15-man squad under strict budget constraints, positional quotas, per-club limits, and user-defined vetos.

* Multi-Horizon Dynamic Planner
Extends the single-week solver into a rolling dynamic program selecting a squad for every gameweek in a five to eight week horizon simultaneously while tracking transfer penalties and banked free transfers.

* Stochastic Chip Evaluator
Synchronizes active FPL teams via ID and evaluates chip activation thresholds such as Wildcard or transfer penalty hits by comparing current squad projections against global mathematical optima.

* Live Standings 2026/2027
A full-height real-time league table monitoring actual club performance, goal differences, and current form to guide tactical decisions.

## 3. Tactical Football Analyst Modules

The platform includes a secondary suite dedicated to spatial deconstruction, machine learning scouting, and tactical matrix generation.

* Pro Analytics and Lineups
Processes multi-league metrics to generate predicted starting lineups and identify key playmakers through progressive passing and expected assist indexes.

* K-Means Statistical Clustering
Unsupervised machine learning groups players based on multi-dimensional statistical profiles to identify hidden tactical alternatives and transfer targets.

* Spatial Pass Network and xT Grid
Constructs passing networks using graph theory while evaluating pitch zones through an Expected Threat grid model that values pitch locations by proximity to goal and centrality.

* Defensive Flank Vulnerability Matrix
Splits conceded shots into left, central, and right thirds of the pitch to flag which flank a defense leaks the most expected goals against.

* Match Simulator
Combines Dixon-Coles Poisson regression with 10,000 Monte Carlo stochastic simulations to generate stable outcome probabilities and exact score matrices.

## 4. Future Roadmap and Advanced Theoretical Modules

* Quantitative Enhancements
Integration of Expected Bonus Points models, portfolio theory for squad risk correlation, dynamic Elo ratings, and gradient boosting machine learning models for point predictions.

* Theoretical Frontiers
Exploration of quantum annealing for combinatorial squad optimization, computer vision biomechanics, and multi-agent reinforcement learning.

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
