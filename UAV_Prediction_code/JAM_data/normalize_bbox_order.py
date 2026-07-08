from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent
CSV_PATH = DATA_DIR / "test.csv"


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    swap_mask = df["x"] > df["a"]
    df.loc[swap_mask, ["x", "y", "a", "b"]] = df.loc[
        swap_mask, ["a", "b", "x", "y"]
    ].to_numpy()
    df.to_csv(CSV_PATH, index=False)
    print(f"Normalized bounding-box coordinate order in: {CSV_PATH}")


if __name__ == "__main__":
    main()
