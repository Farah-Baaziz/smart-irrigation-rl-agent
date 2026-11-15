# Smart Irrigation Agent: Deep Learning Forecasting + Reinforcement Learning

An agent that decides **each day how much water to irrigate** (0, 5, 10, 15 or 20 mm) during a maize growing season, combining a deep-learning soil-moisture forecaster with a reinforcement-learning decision policy, evaluated against simple, well-tuned baselines.

**Authors:** Farah Baaziz and Ben Haj Bel Kacem Marwen

---

## Problem

Agriculture uses about 70% of global freshwater withdrawals (FAO). Over-irrigation wastes water; under-irrigation causes water stress and yield loss. Fixed irrigation schedules ignore weather and soil state.

**Research question:** How can a daily irrigation dose be chosen from soil state and weather so that water is saved without sacrificing the crop?

## Approach

The system has three blocks:

| Block | Question | Method | Tools |
|---|---|---|---|
| 1. Forecasting | Soil moisture in 1 to 7 days? | Supervised multi-horizon regression on 14-day windows | PyTorch: LSTM, GRU, Transformer |
| 2. Decision | How much to irrigate today? | Reinforcement learning (PPO, compared with DQN) | Stable-Baselines3, gym-DSSAT |
| 3. Explanation | Why this decision? | Template-based text (optional LLM) | Python, Anthropic API |

The forecaster is not a decision-maker. The explanation block only restates numbers produced by the agent.

### Two agent versions

- **Agent v1 (PPO):** sees soil moisture at three depths, rainfall, Tmax, Tmin, radiation, crop day and last irrigation.
- **Agent v2 (PPO + forecast):** same inputs plus the 7 forecast values from the LSTM/GRU/Transformer. This measures the actual contribution of deep learning to the decision.

### Baselines

- No irrigation
- Fixed calendar
- Soil-moisture threshold (tuned on the validation set)
- Random Forest and XGBoost for forecasting (persistence as the simplest reference)

## Data

- **Simulated seasons** from gym-DSSAT (train / validation / test seeds kept disjoint). Generated files are in `data/processed/`.
- **ERA5-Land** daily reanalysis at a point in central Tunisia (1981 to 2023), downloaded from the [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/). Raw files are not included in this repository because of their size; see *Reproducing the data* below.
- **SMAP** soil-moisture cross-check (optional).

## Evaluation

All methods are compared on the same weather seasons, with confidence intervals. Agents are also tested under stress conditions: drought, heatwave and noisy sensors (`src/stress_tests.py`).

### Forecasting results (RMSE by horizon)

| Model | J+1 | J+3 | J+7 |
|---|---|---|---|
| Persistence | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD |
| XGBoost | TBD | TBD | TBD |
| LSTM | TBD | TBD | TBD |
| GRU | TBD | TBD | TBD |
| Transformer | TBD | TBD | TBD |

### Decision results (validation / test)

| Policy | Yield | Water (mm) | Water-stress days | Reward |
|---|---|---|---|---|
| No irrigation | TBD | TBD | TBD | TBD |
| Fixed calendar | TBD | TBD | TBD | TBD |
| Threshold | TBD | TBD | TBD | TBD |
| PPO v1 | TBD | TBD | TBD | TBD |
| PPO v2 (with forecast) | TBD | TBD | TBD | TBD |
| DQN | TBD | TBD | TBD | TBD |

> Results are reported honestly, including when the agent does not beat the best baseline.

## Installation

Requirements: Python 3.8+ (the Docker image uses 3.8), Docker, and a CDS account for ERA5.

```bash
git clone git@github.com:Farah-Baaziz/smart-irrigation-rl-agent.git
cd smart-irrigation-rl-agent

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements_colab.txt
```

The gym-DSSAT simulator runs in Docker:

```bash
docker start dssat               # or create it from the image described in the setup guide
docker exec -it dssat /bin/bash
```

To test the whole Python pipeline without DSSAT, use the included toy simulator:

```bash
USE_FAKE_DSSAT=1 python -m tests.smoke_test
```

## Usage

```bash
# Generate simulated datasets (train / val / test)
python -m src.generate_data

# Forecasting baselines
python -m src.ml_baselines --source sim
python -m src.ml_baselines --source era5

# Prepare ERA5-Land
python -m src.era5_prepare

# Evaluate baselines on validation
python -m src.evaluate --split val --methods best

# Train PPO / DQN, optionally with the forecaster
python -m src.train_rl --algo ppo --tag ppo_v1
python -m src.train_rl --algo ppo --tag ppo_v2 --forecaster models/forecaster_sim_lstm_s0.pt

# Compare results and run stress tests
python -m src.compare --files results/*.csv --ref threshold
python -m src.stress_tests --rl models/ppo_v1,models/ppo_v2
```

Colab: the LSTM/GRU/Transformer training was run on a GPU in Google Colab (see `requirements_colab.txt`).

## Project structure

```
irrigation_agent/
├── README.md
├── requirements_docker.txt
├── requirements_colab.txt
├── src/          config, env_wrapper, policies, evaluate, train_rl, ml_baselines, era5_prepare, stress_tests, compare, ...
├── scripts/      00_discover_env.py, 01_run_one_season.py, 02_calibrate_soil.py
├── tests/        fake_dssat.py (toy simulator), smoke_test.py
├── data/raw/     ERA5 and SMAP raw files (not versioned)
├── data/processed/   simulated datasets, era5_daily.csv, soil_bounds.json
├── models/       forecaster_*.pt and one folder per RL agent (not versioned)
├── results/      evaluation CSVs, figures (not versioned)
└── logs/         DSSAT and training logs (not versioned)
```

## Reproducing the data

1. Create a Copernicus CDS account and add your API key to `~/.cdsapirc`.
2. Run the ERA5 download step from the setup guide (`logs/era5_download.log` ends with `OK`).
3. Run `python -m src.era5_prepare`.

## Limitations

- The simulator is gym-DSSAT with a single maize scenario at one location in central Tunisia. Results do not transfer to other crops or regions without recalibration.
- The soil bounds are calibrated from simulated data (`scripts/02_calibrate_soil.py`).
- The agent is trained and evaluated on simulated seasons. It has not been tested in the field.
- SMAP cross-checking was optional and may not be included in the final results.

## Possible extensions

- Add a Transformer- or LSTM-based forecast directly into the RL state (v2 already does this).
- Test RecurrentPPO (`sb3-contrib`) for partial observability.
- Replace the template explanations with a language model, keeping the numbers fixed by the agent.

## License

MIT (to be confirmed by the authors).

## Acknowledgements

Simulator: gym-DSSAT (`gym_dssat_pdi`). Weather data: ERA5-Land, Copernicus Climate Change Service. Soil-moisture cross-check: SMAP, NASA.
