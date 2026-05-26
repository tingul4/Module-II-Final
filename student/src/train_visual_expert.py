import argparse
from pathlib import Path

from visual_expert import train_visual_expert


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train lightweight visual expert for LPCVC derived dataset.")
    parser.add_argument(
        "--derived_data_path",
        type=Path,
        default=Path("/ssd4/LPCVC2026/Module-II-Final/teacher/derived_deterministic_v1/derived.jsonl"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/ssd4/LPCVC2026/Module-II-Final/student/experts/default"),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_samples", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train_visual_expert(
        derived_jsonl=args.derived_data_path,
        output_root=args.output_dir,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
