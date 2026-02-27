from pathlib import Path

def find_project_root(start: Path | None = None) -> Path:
    if start is None:
        start = Path(__file__).resolve()

    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists() or (p / ".git").exists():
            return p
    raise RuntimeError("Project-root not found (pyproject.toml or .git missing)")

