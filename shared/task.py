"""The common task all three topologies solve."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PAPER_PATH = REPO_ROOT / "sample_paper.md"

TASK_BRIEF = (
    "Write a 300-word summary of the paper for a researcher in the field. "
    "Cover: (1) the central claim in one sentence, (2) the mechanism or method, "
    "(3) the evidence with specific numbers where present, (4) the limitations the "
    "authors acknowledge, and (5) one open question this raises."
)


def load_paper(path: str | Path | None = None) -> str:
    """Load the paper text. Defaults to the bundled sample paper."""
    path = Path(path) if path is not None else DEFAULT_PAPER_PATH
    return path.read_text()
