# Smart Option Pricer (Binomial + Black–Scholes)

This project implements a **market-data-driven option pricer**:
- **Binomial tree pricer** (European + American style)
- **Black–Scholes** benchmark (European)
- Optional **implied volatility** retrieval from option chains (Yahoo via `yfinance`)
- Optional **IV surface** building + **Dupire local volatility** surface (when selected)
- Tree visualization with optional hover tooltips (matplotlib + mplcursors)

The repository is organized so that a recruiter can run everything with **one click / one command**.



## Project structure

- `run_for_recrutors.py`  
  **Entry point for recrutors.** Creates an isolated virtual environment (`.venv`), installs dependencies, then launches the app.

- `internal_skeleton.py`  
  The actual pricer logic (data fetch, calibration, binomial tree, Black–Scholes, IV surface, Dupire local vol, plotting).

- `requirements.txt`  
  Python dependencies used by the runner.

- `README.md`  
  This file.



## Quick start (recommended)


### Option A “Press Run” in VS Code
1. Open the project folder in VS Code  
2. Open `run_for_recrutors.py`  
3. Click **Run**

The script will:
- Create `.venv` (first run only, unless `--clean`)
- Install dependencies from `requirements.txt`
- Launch the pricer (`internal_skeleton.py`)

### Option B Terminal
From the project folder:

```bash` 
python3 run_for_recrutors.py


## Useful flags

python3 run_for_recrutors.py --clean
python3 run_for_recrutors.py --skip-install

What the program asks

## What the program asks

When running, the pricer will prompt for:
- Ticker (e.g. AAPL, MSFT, NVDA, or European tickers like AIR.PA)
- Option type: call / put
- Time to maturity (days)
- Contract style: American or European
- Strike (ATM by default)
- Volatility source:
  - hist (historical)
  - atm (ATM implied vol from option chain)
  - mkt (implied vol at strike K)
- Binomial steps
- Vol mode:
  - constant
  - term
  - local
  - dupire (build local vol surface from IV surface)

## Output

Output includes:
- Data source used (yahoo, stooq, or manual)
- Auto risk-free rate suggestion (editable)
- European binomial price vs Black–Scholes
- American binomial price + premium
- Optional early-exercise nodes count (put)
