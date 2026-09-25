"""
Portfolio Core CLI entry point.
Allows running:
    python -m portfolio_core --delete-after YYYY-MM-DD [--dry-run] [--include-transactions]
    python -m portfolio_core --delete-date YYYY-MM-DD [--dry-run] [--include-transactions]
    python -m portfolio_core --download-data [YYYY-MM-DD]
    python -m portfolio_core --clear-tables
"""

import sys
from portfolio_core.db import (
    clear_all_tables_except_transactions,
    delete_data_after_date,
    delete_data_for_date,
    trigger_data_download,
    get_target_download_date,
    get_current_day_delete_date,
    run_full_pipeline
)

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
    elif any(arg.startswith("--delete-date") for arg in sys.argv):
        idx = next(i for i, arg in enumerate(sys.argv) if arg.startswith("--delete-date"))
        if "=" in sys.argv[idx]:
            dt = sys.argv[idx].split("=", 1)[1]
        elif idx + 1 < len(sys.argv) and not sys.argv[idx + 1].startswith("--"):
            dt = sys.argv[idx + 1]
        else:
            dt = get_current_day_delete_date()
        inc_tx = "--include-transactions" in sys.argv
        dry = "--dry-run" in sys.argv
        mode_str = "Previewing rows to delete" if dry else "Deleting rows"
        print(f"{mode_str} with date == {dt} (include_transactions={inc_tx})...")
        res = delete_data_for_date(dt, include_transactions=inc_tx, dry_run=dry)
        for tbl, cnt in res.items():
            print(f"  - {tbl}: {cnt:,} rows {'matched' if dry else 'deleted'}")
        print("Done.")
    elif any(arg.startswith("--download-data") for arg in sys.argv):
        idx = next(i for i, arg in enumerate(sys.argv) if arg.startswith("--download-data"))
        dt = None
        if "=" in sys.argv[idx]:
            dt = sys.argv[idx].split("=", 1)[1]
        elif idx + 1 < len(sys.argv) and not sys.argv[idx + 1].startswith("--"):
            dt = sys.argv[idx + 1]
        target_dt = get_target_download_date(dt)
        print(f"Triggering data download for {target_dt}...")
        res = trigger_data_download(target_date=target_dt)
        print(f"Result: {res.get('message', res)}")
    else:
        run_full_pipeline()
