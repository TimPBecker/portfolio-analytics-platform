# 🏛️ Portfolio Analytics Platform (Monorepo)

An institutional-grade portfolio risk analytics, automated data pipeline, and interactive web dashboard platform.

---

## 🏗️ Monorepo Architecture

```
portfolio-analytics-platform/
├── packages/
│   └── portfolio-core/             # 🧠 Shared Core Library (Single Source of Truth)
│       ├── src/portfolio_core/
│       │   ├── config.py           # Configuration, YAML loader & .env credentials
│       │   ├── db.py               # SQLAlchemy connection pooling, 13 automated schemas & queries
│       │   └── analytics/          # Pure financial mathematics
│       │       ├── volatility.py   # Rolling sample, EWMA (λ=0.94), Parkinson, scaling factors
│       │       ├── var.py          # Historical VaR, Vol-Scaled VaR, CVaR, Shapley Attribution
│       │       ├── empirical_cdf.py# Non-parametric empirical CDF percentile & scenario estimation
│       │       ├── backtesting.py  # Kupiec POF, Christoffersen independence & binomial calibration tests
│       │       └── statistics.py   # Returns, KDE, moments, Jarque-Bera normality tests
│       └── tests/                  # Core analytics and database unit tests
│
├── apps/
│   ├── pipeline/                   # ⚙️ Dagster & Dask Ingestion & Materialization Engine
│   │   ├── repo.py                 # Dagster jobs, assets, ops, schedules & backfill definitions
│   │   ├── reporting.py            # Automated multi-panel PDF/chart Telegram reporter
│   │   ├── config.yaml             # Pipeline execution parameters & DB config
│   │   └── tests/                  # Pipeline reporting integration tests
│   │
│   └── dashboard/                  # 📊 Streamlit + Plotly Interactive Risk Dashboard
│       ├── app.py                  # Main dashboard application entrypoint & cached data loader
│       ├── config.yaml             # Dashboard connection & risk modeling parameters
│       └── src/ui/                 # Modular UI tabs
│           ├── tab_portfolio.py    # Holdings allocation, valuation timeline & correlation matrix
│           ├── tab_benchmarks.py   # Centralized benchmarking, shadow portfolio replication & comparison
│           ├── tab_var.py          # Full VaR spectrum (1%-99%), Shapley attribution & scenarios
│           ├── tab_backtesting.py  # VaR model validation, outlier clustering & diagnostic tests
│           ├── tab_returns.py      # Price levels, returns series, histogram & normality
│           ├── tab_volatility.py   # Rolling volatilities & multi-estimator comparison
│           ├── tab_transactions.py # Buy/sell transaction recording with live Yahoo quote lookup
│           └── tab_market_data.py  # Market data coverage overview, custom time series ingestion & refresh
│
├── docker-compose.yml              # Multi-container local/production deployment
├── pyproject.toml                  # Monorepo workspace configuration
└── .env.example                    # Template for database & API credentials
```

---

## 🚀 Quickstart & Installation

### 1. Clone & Set Up Environment

#### Linux / macOS
```bash
git clone <repository_url>
cd portfolio-analytics-platform
python3 -m venv .venv
source .venv/bin/activate

# Install the shared core library in editable mode
pip install -e packages/portfolio-core

# Install pipeline & dashboard dependencies
pip install -r apps/pipeline/requirements.txt
pip install -r apps/dashboard/requirements.txt
```

#### Windows (PowerShell)
```powershell
git clone <repository_url>
cd portfolio-analytics-platform
python -m venv .venv

# If script execution is restricted in PowerShell, run:
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.venv\Scripts\Activate.ps1

# Install the shared core library in editable mode
pip install -e packages/portfolio-core

# Install pipeline & dashboard dependencies
pip install -r apps/pipeline/requirements.txt
pip install -r apps/dashboard/requirements.txt
```

*(For health warnings, prerequisites, and troubleshooting on Windows, see the [Windows Setup & Support (Beta)](#-windows-setup--support-beta) section below).*

### 2. Configure Environment Variables

**Linux / macOS:**
```bash
cp .env.example .env
```

**Windows (PowerShell / Command Prompt):**
```powershell
# PowerShell:
Copy-Item .env.example .env

# Command Prompt (cmd.exe):
copy .env.example .env
```

Edit `.env` to match your selected database backend (see [Database Setup & Configuration](#-database-setup--configuration) below for details).

---

## 🗄️ Database Setup & Configuration

The platform supports two database backends:
1. **SQLite**: Recommended for fast, single-machine local development and lightweight testing with zero server setup.
2. **MariaDB / MySQL**: Recommended for institutional, multi-container, production, and networked environments.

---

### ❓ Do Users Need to Create Database Tables?

> [!IMPORTANT]
> **NO, you do NOT need to create any database tables manually!**
> The platform manages all table creation automatically. You do NOT need to write SQL DDL or execute migration scripts.

When the application runs (launching the Streamlit dashboard, running a Dagster pipeline asset, recording a transaction, or running tests), the platform automatically executes `create_all_tables(engine)` with `CREATE TABLE IF NOT EXISTS` across your database.

If you wish to explicitly pre-initialize or verify all tables from the command line upfront, run:

```bash
# Initialize all 13 tables in the active database:
python -m portfolio_core.db --init-tables
```

Or programmatically in Python:

```python
from portfolio_core.db import create_all_tables
create_all_tables()
```

#### Managed Database Tables (13 Tables)

| Table | Description |
| :--- | :--- |
| `TRANSACTIONS` | Portfolio trade ledger (Buy/Sell order history, dates, and share quantities). |
| `ASSET_PRICES` | Daily historical asset prices (OHLCV, dividends, stock splits, currency). |
| `FX_RATES` | Daily foreign exchange rates to GBP with forward/back-filling. |
| `CASHFLOWS` | Computed dividend payout cashflows per held position. |
| `CASHACCOUNT` | Multi-currency cash account balances and cumulative valuations in GBP. |
| `PORTFOLIO_VALUES` | Daily total portfolio valuation history (Stocks, Cash, and Total Value in GBP). |
| `PORTFOLIO_VAR` | Value-at-Risk calculations across confidence levels (1%-99%) and multiple risk models. |
| `PORTFOLIO_SCENARIO_PNL` | Simulated scenario profit-and-loss distributions across risk models. |
| `PORTFOLIO_RISK_CONTRIBUTIONS` | Game-theoretic Shapley Value risk attributions per asset position. |
| `BENCHMARKS` | Benchmark definitions and constituent asset weights (single or linear combinations). |
| `BENCHMARK_TRANSACTIONS` | Shadow transactions automatically replicating portfolio cash deployments on trade dates. |
| `BENCHMARK_VALUES` | Daily mark-to-market valuations for all configured benchmark portfolios. |
| `PROCESSED_DATES` | Audit ledger tracking all successfully completed trading days and data ingestion dates. |

---

### Option A: SQLite Setup (Fastest & Simplest)

SQLite requires **zero external database server installation**. The database file is created automatically on disk upon first connection.

#### Which Configurations are NOT Needed for SQLite?
When using SQLite, server connection parameters and credentials are **NOT needed and are ignored**:
- `DB_PASSWORD`: **Not needed** (leave empty or omit from `.env`)
- `DB_USER`: **Not needed**
- `DB_HOST`: **Not needed**
- `DB_PORT`: **Not needed**

#### Configuration Instructions:
1. In `apps/dashboard/config.yaml` and `apps/pipeline/config.yaml`:
   ```yaml
   db:
     type: "sqlite"
     sqlite_path: "stocks.s3db"
   ```
2. Or configure via `.env` overrides:
   ```bash
   DB_TYPE=sqlite
   SQLITE_PATH=stocks.s3db
   ```

> [!NOTE]
> **Strict SQLite Extension Rule**: The SQLite database path must always use the `.s3db` extension (e.g. `stocks.s3db`). This ensures:
> - The database file is automatically excluded from git tracking via `.gitignore`.
> - In development mode (`PORTFOLIO_ENV=development`) and test mode (`PORTFOLIO_ENV=test`), the system automatically resolves to `<name>_dev.s3db` or in-memory `:memory:` to safeguard your production data.

---

### Option B: MariaDB / MySQL Setup (Production & Multi-Container)

MariaDB / MySQL is suited for multi-container Docker deployments, multi-user access, or networked environments.

#### One-Time Database Instance Creation
Before running the application against a MariaDB server for the first time, the database instances themselves must exist on your server. Log in to your MariaDB server as an administrator (e.g., via `mariadb -u root -p`) and create the production and development databases:

```sql
-- 1. Create production and development database instances
CREATE DATABASE IF NOT EXISTS stocks;
CREATE DATABASE IF NOT EXISTS stocks_dev;

-- 2. Create the application user and grant permissions
CREATE USER IF NOT EXISTS 'stocks'@'%' IDENTIFIED BY 'your_secure_password';
GRANT ALL PRIVILEGES ON stocks.* TO 'stocks'@'%';
GRANT ALL PRIVILEGES ON stocks_dev.* TO 'stocks'@'%';
FLUSH PRIVILEGES;
```

*(Note: You only need to create the database containers `stocks` and `stocks_dev`. Do **NOT** create tables manually; the platform creates all tables automatically).*

#### Configuration Instructions:
1. Specify connection parameters in `apps/dashboard/config.yaml` and `apps/pipeline/config.yaml`:
   ```yaml
   db:
     type: "mariadb"
     host: "192.168.178.40"   # or "localhost"
     port: 3306
     user: "stocks"
     database: "stocks"
     databases:
       - "stocks"
   ```
2. Add your secret password to `.env`:
   ```bash
   DB_PASSWORD=your_secure_password
   ```
3. Optional environment overrides in `.env`:
   ```bash
   DB_HOST=192.168.178.40
   DB_PORT=3306
   DB_USER=stocks
   DB_NAME=stocks
   ```

---

### 🛡️ Environment Safety & Database Isolation

The platform enforces strict safety rules across environments via the `PORTFOLIO_ENV` variable:

| Environment | Setting | Database Target | Behavior |
| :--- | :--- | :--- | :--- |
| **Production** | `PORTFOLIO_ENV=production` | `stocks` | Live production operations, portfolio valuations, and real trades. |
| **Development** | `PORTFOLIO_ENV=development` | `stocks_dev` / `stocks_dev.s3db` | Development and staging. Connecting to production `stocks` is strictly blocked by safety assertions. |
| **Testing** | `PORTFOLIO_ENV=test` or `TEST_MODE=1` | `stocks_dev` / `:memory:` | Automated tests, test runners, and diagnostic scripts strictly target dev databases or on-demand SQLite databases. Production `stocks` is never touched. |

---

## 🛠️ Database CLI & Maintenance Utilities

The shared core library includes built-in command-line maintenance tools:

```bash
# 1. Initialize all database tables without running data ingestion:
python -m portfolio_core.db --init-tables

# 2. Run full end-to-end data ingestion, backfilling, and portfolio valuation:
python -m portfolio_core.db

# 3. Clear all computed pipeline tables (strictly preserves manual TRANSACTIONS):
python -m portfolio_core.db --clear-tables

# 4. Prune records after a specific date (preview dry-run):
python -m portfolio_core.db --delete-after 2026-09-24 --dry-run

# 5. Prune records after a specific date (execute deletion):
python -m portfolio_core.db --delete-after 2026-09-24
```

---

## 🪟 Windows Setup & Support (Beta)

> [!WARNING]
> ### ⚠️ Health Warning: Windows Setup is in BETA & Not Fully Tested
> - **Primary Target**: The Portfolio Analytics Platform is engineered and tested primarily on **Linux** (Debian, Ubuntu, and containerized Linux via Docker).
> - **Beta Status**: Native Windows operation (running directly under Windows 10 or 11 using PowerShell or Command Prompt) is currently in **Beta** and has **not been fully tested** across all Windows editions, execution environments, or terminal configurations.
> - **Potential Friction Points**: While core financial mathematics, the Streamlit dashboard, and SQLite / MariaDB database connectivity are cross-platform, native Windows users may encounter differences in Python execution policies, path resolution formatting, and Dask / Dagster multiprocessing (`spawn` vs `fork`).
> - **Recommended Alternative (WSL2)**: If you are developing on a Windows machine, running inside **WSL2 (Windows Subsystem for Linux - Ubuntu)** or **Docker Desktop** is strongly recommended as it provides 100% feature parity with the production Linux environment.

---

### Native Windows Prerequisites

1. **Python 3.10+**: Download from [python.org](https://www.python.org/downloads/windows/).
   - ⚠️ **Important**: Ensure the installer option **"Add python.exe to PATH"** is checked.
2. **Git for Windows**: Download from [git-scm.com](https://git-scm.com/).
3. **PowerShell 7+ or Windows Terminal**: Highly recommended over legacy Command Prompt (`cmd.exe`).

---

### Step-by-Step Native Windows Setup

#### 1. Create and Activate Virtual Environment

Open PowerShell and navigate to the project directory:

```powershell
cd path\to\portfolio-analytics-platform

# Create virtual environment
python -m venv .venv
```

By default, Windows PowerShell restricts running scripts (`PSSecurityException`). If you encounter an execution policy error when activating the environment, run:

```powershell
# Temporarily permit signed scripts in the current PowerShell process:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned

# Activate the virtual environment:
.venv\Scripts\Activate.ps1
```

*(If using legacy `cmd.exe`, run `.venv\Scripts\activate.bat` instead).*

#### 2. Install Packages

```powershell
pip install -e packages\portfolio-core
pip install -r apps\pipeline\requirements.txt
pip install -r apps\dashboard\requirements.txt
```

> [!NOTE]
> All core dependencies (`numpy`, `pandas`, `scipy`, `statsmodels`, `SQLAlchemy`, `PyMySQL`, `streamlit`, `plotly`) provide pre-compiled Windows binary wheels (`win_amd64`) on PyPI, and `pymysql` is pure Python. Microsoft Visual C++ Build Tools are **not required** for standard setup.

#### 3. Configure Environment

Copy the example environment file:

```powershell
# PowerShell:
Copy-Item .env.example .env

# Command Prompt (cmd.exe):
copy .env.example .env
```

Because `portfolio_core.config` automatically parses `.env` files via `python-dotenv`, you can define all credentials and settings directly in `.env` without setting system environment variables.

---

### Windows Specific Nuances & Health Warnings

#### 1. SQLite Database Path Formatting
- **Relative Paths (Default)**: Relative paths such as `sqlite_path: "stocks.s3db"` in `config.yaml` work without modification.
- **Absolute Paths**: If you configure an absolute path on Windows, do **NOT** use unescaped backslashes (`\`). SQLAlchemy requires forward slashes (`/`):
  ```yaml
  # ❌ Incorrect:
  sqlite_path: "C:\Users\username\portfolio\stocks.s3db"

  # ✅ Correct:
  sqlite_path: "C:/Users/username/portfolio/stocks.s3db"
  ```

#### 2. Shell Environment Variables
Setting shell environment variables manually in Windows requires different syntax:
- **PowerShell**: `$env:PORTFOLIO_ENV="development"`
- **Command Prompt**: `set PORTFOLIO_ENV=development`
*(Recommendation: Always set `PORTFOLIO_ENV=development` or `PORTFOLIO_ENV=production` directly inside your `.env` file instead).*

#### 3. Dask & Dagster Multiprocessing (`spawn` vs `fork`)
- Linux environments spawn child processes using `fork()`, whereas Windows strictly uses `spawn()`.
- If Dagster pipeline tasks hang or terminate unexpectedly during parallel execution on native Windows, set Dagster to single-process execution or run the pipeline inside WSL2 / Docker.

---

### Windows Quick-Reference Command Table

| Action | Linux / macOS | Windows (PowerShell) |
| :--- | :--- | :--- |
| **Activate Virtualenv** | `source .venv/bin/activate` | `.venv\Scripts\Activate.ps1` |
| **Initialize Tables** | `python -m portfolio_core.db --init-tables` | `python -m portfolio_core.db --init-tables` |
| **Run Unit Tests** | `PORTFOLIO_ENV=test TEST_MODE=1 pytest` | `$env:PORTFOLIO_ENV="test"; $env:TEST_MODE="1"; pytest` |
| **Run Streamlit Dashboard** | `cd apps/dashboard && streamlit run app.py` | `cd apps\dashboard; streamlit run app.py` |
| **Run Dagster Pipeline** | `cd apps/pipeline && dagster dev -f repo.py -p 3000` | `cd apps\pipeline; dagster dev -f repo.py -p 3000` |

---

### 🐧 Recommended Alternative: Running via WSL2 (Ubuntu)

For Windows developers seeking zero configuration friction and 100% production parity:

1. Open PowerShell as Administrator and run:
   ```powershell
   wsl --install -d Ubuntu
   ```
2. Restart your computer if prompted, and open the new **Ubuntu** terminal.
3. Clone the repository and follow the standard [Linux Quickstart](#-quickstart--installation):
   ```bash
   git clone <repo-url>
   cd portfolio-analytics-platform
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e packages/portfolio-core
   pip install -r apps/pipeline/requirements.txt
   pip install -r apps/dashboard/requirements.txt
   streamlit run apps/dashboard/app.py
   ```
4. Access the dashboard from any Windows browser (Chrome, Edge, Firefox) at `http://localhost:8501`. WSL2 automatically forwards localhost ports to the Windows host!

---

## 🧪 Running Unit & Integration Tests

The repository includes a comprehensive test suite with over **100 automated tests** covering core analytics, empirical distributions, VaR backtesting diagnostics, benchmark synchronization, database isolation, Dagster reporting, and Streamlit UI components:

```bash
# Run the complete test suite (Linux / macOS):
pytest -v

# Run the complete test suite (Windows PowerShell):
$env:PORTFOLIO_ENV="test"; $env:TEST_MODE="1"; pytest -v

# Run tests for shared analytics & database core:
pytest -v packages/portfolio-core/tests

# Run tests for Dagster pipeline reporting:
pytest -v apps/pipeline/tests

# Run tests for Streamlit dashboard UI:
pytest -v apps/dashboard/tests
```

> [!TIP]
> All tests automatically adhere to safe database isolation (`PORTFOLIO_ENV=test`, `TEST_MODE=1`), utilizing isolated SQLite `:memory:` instances or the MariaDB `stocks_dev` database.

---

## 🖥️ Running the Applications

### Launch Streamlit Dashboard

**Linux / macOS:**
```bash
cd apps/dashboard
streamlit run app.py
```

**Windows (PowerShell):**
```powershell
cd apps\dashboard
streamlit run app.py
```
*Access in browser at `http://localhost:8501`.*

### Launch Dagster Pipeline Webserver

**Linux / macOS:**
```bash
cd apps/pipeline
dagster dev -f repo.py -p 3000
```

**Windows (PowerShell):**
```powershell
cd apps\pipeline
dagster dev -f repo.py -p 3000
```
*Access Dagster UI at `http://localhost:3000`.*

---

## 🐳 Docker Deployment

Run all services together with Docker Compose:

```bash
docker compose up --build -d
```
