from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent
INPUT_CSV = DATA_DIR / "train.csv"
OUTPUT_CSV = DATA_DIR / "re_train.csv"


def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    df["image_path"] = df["image_path"].str.replace("images", "re_images", regex=False)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Updated image paths saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
