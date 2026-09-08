"""
Portfolio Core CLI entry point.
Allows running:
    python -m portfolio_core --delete-after YYYY-MM-DD [--dry-run] [--include-transactions]
    python -m portfolio_core --clear-tables
"""

import sys
from portfolio_core.db import clear_all_tables_except_transactions, delete_data_after_date, run_full_pipeline

if __name__ == "__main__":
    if "--clear-tables" in sys.argv:
        print("Clearing all pipeline tables (preserving TRANSACTIONS)...")
        deleted = clear_all_tables_except_transactions()
        for tbl, cnt in deleted.items():
            print(f"  - {tbl}: {cnt:,} rows deleted")
        print("Done.")
    elif any(arg.startswith("--delete-after") for arg in sys.argv):
        idx = next(i for i, arg in enumerate(sys.argv) if arg.startswith("--delete-after"))
        if "=" in sys.argv[idx]:
            dt = sys.argv[idx].split("=", 1)[1]
        elif idx + 1 < len(sys.argv):
            dt = sys.argv[idx + 1]
        else:
            print("Usage: python -m portfolio_core --delete-after YYYY-MM-DD [--include-transactions] [--dry-run]")
            sys.exit(1)
        inc_tx = "--include-transactions" in sys.argv
        dry = "--dry-run" in sys.argv
        mode_str = "Previewing rows to delete" if dry else "Deleting rows"
        print(f"{mode_str} with date > {dt} (include_transactions={inc_tx})...")
        res = delete_data_after_date(dt, include_transactions=inc_tx, dry_run=dry)
        for tbl, cnt in res.items():
            print(f"  - {tbl}: {cnt:,} rows {'matched' if dry else 'deleted'}")
        print("Done.")
    else:
        run_full_pipeline()
