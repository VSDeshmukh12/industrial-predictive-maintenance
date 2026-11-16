"""Compare protected Phase 2 files against the SHA-256 snapshot taken before Phase 3 began."""
from __future__ import annotations

from pathlib import Path

from pm import config
from pm.data.loader import sha256_of

SNAPSHOT = config.ARTIFACTS_DIR / "phase3" / "phase2_integrity_snapshot.sha256"

# The ONLY intentional differences, each with its reason (Phase 3 requirement 6 and documentation).
EXPECTED_CHANGES = {
    "requirements.txt": "Dependencies updated and fully pinned (CPU-only xgboost-cpu, imbalanced-learn, lightgbm, catboost, matplotlib + transitive pins); required by Phase 3 requirement 6.",
    "requirements-dev.txt": "Regenerated from the clean CPU-only environment (pytest + its dependencies) so it matches the new runtime pins.",
    "pyproject.toml": "Package version bumped 0.2.0 -> 0.3.0 and description updated for Phase 3; no build or dependency semantics changed.",
    "README.md": "Rewritten to describe Phase 3 (status, setup and run commands, results location, synthetic-data caveat); the Phase 2 status text is superseded.",
}


def read_snapshot(path: Path = SNAPSHOT) -> dict[str, str]:
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            h, name = line.split(maxsplit=1)
            out[name.strip()] = h
    return out


def compare(root: Path = config.PROJECT_ROOT, snapshot: Path = SNAPSHOT) -> dict:
    snap = read_snapshot(snapshot)
    identical, changed, missing = [], [], []
    for name, h in snap.items():
        f = root / name
        if not f.exists():
            missing.append(name)
        elif sha256_of(f) == h:
            identical.append(name)
        else:
            changed.append(name)
    return {"n_snapshot_files": len(snap), "identical": identical, "changed": changed, "missing": missing,
            "unexpected_changes": [c for c in changed if c not in EXPECTED_CHANGES],
            "explained_changes": {c: EXPECTED_CHANGES[c] for c in changed if c in EXPECTED_CHANGES}}


# Forbidden tokens are written as concatenated literals so this module does not itself match the scan.
_FORBIDDEN = ("get_" "test_data", "test" "_udi")


def holdout_violations(root: Path = config.PROJECT_ROOT) -> list[str]:
    """Phase 3 scripts (04+) and experiment/model/evaluation modules that reference the hold-out API."""
    files = [f for f in (root / "scripts").glob("*.py") if "04" <= f.name[:2] <= "14"]   # Phase 3 scripts; Phase 4 (15+) is guarded by tests/test_phase4_*.py
    for sub in ("experiments", "models", "evaluation"):
        files += list((root / "src" / "pm" / sub).glob("*.py"))
    return sorted(str(f.relative_to(root)) for f in files if any(t in f.read_text() for t in _FORBIDDEN))
