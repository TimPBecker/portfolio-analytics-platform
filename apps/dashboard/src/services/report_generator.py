"""
Service module for programmatic Quarto + Typst PDF Report Generation.
Compiles high-resolution executive financial & risk reports from .qmd templates.
"""

import os
import subprocess
import tempfile
import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def find_report_template() -> Optional[Path]:
    """Finds the persistent Quarto report template across known paths."""
    candidates = [
        Path("/app/reports/portfolio_report.qmd"),
        Path("/reports/portfolio_report.qmd"),
        Path(__file__).resolve().parent.parent.parent / "reports" / "portfolio_report.qmd",
        Path(__file__).resolve().parent.parent.parent.parent.parent / "reports" / "portfolio_report.qmd",
        Path("reports/portfolio_report.qmd"),
        Path("apps/dashboard/reports/portfolio_report.qmd"),
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    return None


def find_quarto_binary() -> Optional[str]:
    """Finds the quarto executable across PATH and known binary locations."""
    import shutil
    candidates = [
        shutil.which("quarto"),
        "/usr/local/bin/quarto",
        "/usr/bin/quarto",
        "/opt/quarto/bin/quarto",
        os.path.expanduser("~/.local/bin/quarto"),
    ]
    for c in candidates:
        if c and os.path.exists(c) and os.access(c, os.X_OK):
            return c
    return None


def generate_portfolio_pdf_report(
    asof_date: Optional[str] = None,
    output_dir: Optional[str] = None,
    db_name: Optional[str] = None
) -> Tuple[bool, Optional[str], Optional[bytes], Optional[str]]:
    """
    Executes Quarto CLI to render the Typst portfolio report into a standalone PDF.

    Returns:
        (success, pdf_path, pdf_bytes, error_message)
    """
    quarto_bin = find_quarto_binary()
    if not quarto_bin:
        err = "The 'quarto' CLI executable is not installed or not available on the system PATH."
        logger.error(err)
        return False, None, None, err

    template_path = find_report_template()
    if not template_path or not template_path.exists():
        err = "Quarto report template 'portfolio_report.qmd' was not found in candidate paths."
        logger.error(err)
        return False, None, None, err

    target_dir = Path(output_dir) if output_dir else Path(tempfile.gettempdir())
    target_dir.mkdir(parents=True, exist_ok=True)
    
    date_tag = asof_date.replace("-", "") if asof_date else "latest"
    pdf_filename = f"Portfolio_Analytics_Report_{date_tag}.pdf"
    output_pdf_path = target_dir / pdf_filename

    env = os.environ.copy()
    # Add common paths to PATH in case not inherited
    env["PATH"] = f"/usr/local/bin:/usr/bin:/bin:/opt/quarto/bin:{env.get('PATH', '')}"
    if asof_date:
        env["REPORT_ASOF_DATE"] = str(asof_date)[:10]
    if db_name:
        env["DB_NAME"] = str(db_name)

    # Remove stale files before rendering
    expected_in_template_dir = template_path.parent / output_pdf_path.name
    if output_pdf_path.exists():
        try:
            output_pdf_path.unlink()
        except Exception:
            pass
    if expected_in_template_dir.exists():
        try:
            expected_in_template_dir.unlink()
        except Exception:
            pass

    # Command: <quarto> render <template> --to typst --output <pdf_filename> --no-cache
    cmd = [
        quarto_bin,
        "render",
        str(template_path),
        "--to",
        "typst",
        "--output",
        output_pdf_path.name,
        "--no-cache"
    ]

    logger.info(f"Rendering Quarto report: {' '.join(cmd)} (working_dir={target_dir})")

    try:
        res = subprocess.run(
            cmd,
            cwd=str(template_path.parent),
            env=env,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        # Check generated PDF in template dir or target dir
        expected_in_template_dir = template_path.parent / output_pdf_path.name
        default_rendered_pdf = template_path.parent / f"{template_path.stem}.pdf"

        final_pdf_path = None
        if expected_in_template_dir.exists():
            final_pdf_path = expected_in_template_dir
        elif default_rendered_pdf.exists():
            final_pdf_path = default_rendered_pdf
        elif output_pdf_path.exists():
            final_pdf_path = output_pdf_path

        if res.returncode != 0 or final_pdf_path is None or not final_pdf_path.exists():
            err_msg = res.stderr or res.stdout or "Unknown rendering failure."
            logger.error(f"Quarto render failed with code {res.returncode}: {err_msg}")
            return False, None, None, f"Quarto execution error:\n{err_msg}"

        # Copy or move to output_pdf_path if different
        if final_pdf_path != output_pdf_path:
            import shutil
            shutil.copy2(final_pdf_path, output_pdf_path)
            final_pdf_path = output_pdf_path

        pdf_bytes = final_pdf_path.read_bytes()
        logger.info(f"Report successfully compiled to {final_pdf_path} ({len(pdf_bytes):,} bytes)")
        return True, str(final_pdf_path), pdf_bytes, None

    except subprocess.TimeoutExpired:
        err = "Quarto rendering timed out after 120 seconds."
        logger.error(err)
        return False, None, None, err
    except FileNotFoundError:
        err = "The 'quarto' CLI executable is not installed or not available on the system PATH."
        logger.error(err)
        return False, None, None, err
    except Exception as ex:
        err = f"Unexpected error during report generation: {ex}"
        logger.error(err, exc_info=True)
        return False, None, None, err
