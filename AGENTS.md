# Portfolio Analytics Platform - Agent Directives & Safety Rules

## 1. Database Test Isolation & Cleanup (Strict Rule)
- **NEVER run tests against the production database (`stocks`).**
- All automated tests, integration test suites, and diagnostic test scripts must strictly target:
  - The development MariaDB database: `stocks_dev` (or `<database_name>_dev`)
  - An on-demand SQLite test database (`:memory:`, `.s3db`, or `.s2db`) generated via `create_test_sqlite_engine()` or `get_test_engine()`.
- The `PORTFOLIO_ENV=test` and `TEST_MODE=1` environment variables must always be respected to ensure connection strings automatically resolve to `stocks_dev`.
- **Automatic Test Artifact Cleanup**: Any SQLite test database files (`*.s3db`, `*.s2db`, `*_dev.s3db`, `test_*.s3db`) created on disk during test runs must be automatically cleaned up and deleted immediately after tests complete.
