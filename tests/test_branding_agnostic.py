"""Regression: the portfolio demo must stay vendor/customer-agnostic.

Guards against accidental reintroduction of real-company branding after the
Northstar / synthetic-account rebrand.
"""

from pathlib import Path

import pytest

from api._copilot.data_access import load_opportunities
from api._copilot.config import COLLECTION_NAME, DATA_DIR, KNOWLEDGE_DIR

FORBIDDEN = (
    "Socure",
    "socure",
    "Meridian Digital Bank",
    "Northwind Marketplace",
    "Apex Digital Exchange",
    "GigBridge",
    "PlayPeak",
    "Granite State",
    "BrightPath Health",
    "Helios Telecom",
    "LoopRide",
    "Sigma Synthetic",
    "Sigma Identity",
    "RiskOS",
    "socure_angle",
    "retrieve_socure_knowledge",
    "socure_knowledge",
)

SOURCE_ROOT = Path(__file__).resolve().parents[1]
SCAN_GLOBS = (
    "api/**/*.py",
    "api/**/*.md",
    "api/**/*.json",
    "app/**/*.tsx",
    "app/**/*.css",
    "tests/**/*.py",
    "README.md",
)


def _scanned_files() -> list[Path]:
    files: list[Path] = []
    for pattern in SCAN_GLOBS:
        files.extend(SOURCE_ROOT.glob(pattern))
    # Exclude this regression file — it intentionally names the forbidden strings.
    return sorted(
        {
            f
            for f in files
            if f.is_file() and f.name != "test_branding_agnostic.py"
        }
    )


@pytest.mark.parametrize("forbidden", FORBIDDEN)
def test_no_forbidden_branding_in_source(forbidden: str):
    hits: list[str] = []
    for path in _scanned_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        if forbidden in text:
            hits.append(str(path.relative_to(SOURCE_ROOT)))
    assert not hits, f"{forbidden!r} still appears in: {', '.join(hits)}"


def test_synthetic_account_names_are_present():
    accounts = {o["account"] for o in load_opportunities()}
    assert "Acme Digital Bank" in accounts
    assert "ZipRide" in accounts
    assert "NovaLink Telecom" in accounts
    assert "ClearPath Health" in accounts


def test_knowledge_corpus_uses_northstar_product_ids():
    stems = {p.stem for p in KNOWLEDGE_DIR.glob("*.md")}
    assert "northstar-verify" in stems
    assert "atlas-synthetic-fraud" in stems
    assert "atlas-identity-fraud" in stems
    assert "decisionos-platform" in stems
    assert "predictive-doccheck" in stems
    assert "socure-verify" not in stems
    assert "sigma-synthetic-fraud" not in stems
    assert "riskos-platform" not in stems


def test_collection_name_is_vendor_agnostic():
    assert COLLECTION_NAME == "approved_knowledge"
    assert DATA_DIR.exists()
