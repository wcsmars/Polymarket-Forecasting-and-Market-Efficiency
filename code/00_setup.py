"""Create local input and output directories for the research pipeline."""
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    for directory in (
        "data/raw", "data/processed", "results/models",
        "figures/calibration", "figures/models",
    ):
        (root / directory).mkdir(parents=True, exist_ok=True)
    print("Research input and output directories are ready.")


if __name__ == "__main__":
    main()
