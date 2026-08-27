---
name: safe-test-database-isolation
description: >-
  Enforces strict test database isolation so that automated tests, test scripts, and test runners
  never touch or execute against the production 'stocks' database. Directs tests exclusively to the
  development database ('stocks_dev') or on-demand isolated SQLite test databases. Use when running,
  authoring, or debugging tests and database fixtures across portfolio-core, pipeline, and dashboard.
---

# Safe Test Database Isolation Skill

This skill enforces strict isolation between production and testing database environments across the Portfolio Analytics Platform.

## Core Directives & Safety Rule

> [!CAUTION]
> **NEVER connect to, query, modify, or execute tests against the production database (`stocks`).**
> All tests, test fixtures, integration runs, and validation scripts MUST strictly target:
> 1. **MariaDB Development Database**: `<database_name>_dev` (e.g., `stocks_dev`)
> 2. **On-Demand SQLite Test Database**: In-memory (`sqlite:///:memory:`) or temporary `.s3db`/`.s2db` file (`stocks_dev.s3db`).
>
> **Automatic File Cleanup**: Any SQLite files (`.s3db`, `.s2db`, `*_dev.s3db`, `test_*.s3db`) created on disk during tests MUST be automatically deleted upon test completion.

---

## Test Execution Guidelines

### 1. Environment Markers
Always ensure test environment variables are active before running test commands or test scripts:
```bash
export PORTFOLIO_ENV=test
export TEST_MODE=1
```

### 2. Standard Test Invocation Commands

- **Run all platform tests:**
  ```bash
  pytest -v
  ```
- **Run portfolio-core tests:**
  ```bash
  pytest -v packages/portfolio-core/tests
  ```
- **Run pipeline tests:**
  ```bash
  pytest -v apps/pipeline/tests
  ```
- **Run dashboard tests:**
  ```bash
  pytest -v apps/dashboard/tests
  ```

### 3. Using Test Database Engines & Fixtures

When authoring or updating tests in Python, always utilize the core test helpers or standard pytest fixtures:

#### Available Fixtures (`conftest.py`):
- `sqlite_test_engine`: Provides an isolated on-demand in-memory SQLite database with all tables pre-initialized.
- `mariadb_dev_engine`: Connects strictly to `stocks_dev` with connection assertion and table schema initialization.
- `test_engine`: Primary test database fixture (MariaDB dev with on-demand SQLite fallback).

#### Programmatic Test Engine Creation:
```python
from portfolio_core.db import (
    create_test_sqlite_engine,
    get_test_engine,
    get_dev_database_name,
    is_test_environment,
)

# 1. On-demand isolated in-memory SQLite test engine
sqlite_engine = create_test_sqlite_engine(sqlite_path=":memory:", initialize_schema=True)

# 2. Strict MariaDB Dev database engine ('stocks_dev')
dev_engine = get_test_engine(db_type="mariadb", initialize_schema=True)
```

---

## Verification & Pre-Flight Checks

Before executing tests:
1. Verify `get_connection_string(is_test=True)` or `PORTFOLIO_ENV=test` produces a URL ending with `/stocks_dev` (for MariaDB) or `sqlite:///:memory:` (for SQLite).
2. Ensure that any database assertion checks `assert str(engine.url.database).endswith("_dev")` when interacting with MariaDB.
