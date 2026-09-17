"""
Step 1: Data collection.

Loads the 20 Newsgroups subset (alt.atheism vs soc.religion.christian),
a standard, publicly documented text-classification benchmark, and writes
a reproducible train/test split to data/processed/.
"""
import json

import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

from config import DATA_PROCESSED, DATA_RAW, NEWSGROUPS_CATEGORIES, RANDOM_STATE


def load_dataset() -> pd.DataFrame:
    bunch = fetch_20newsgroups(
        subset="all",
        categories=NEWSGROUPS_CATEGORIES,
        remove=("headers", "footers", "quotes"),
        data_home=str(DATA_RAW),
        random_state=RANDOM_STATE,
    )
    df = pd.DataFrame({"text": bunch.data, "label": bunch.target})
    df["target_name"] = df["label"].map(dict(enumerate(bunch.target_names)))
    # Drop empty documents left after header/footer/quote stripping.
    df = df[df["text"].str.strip().str.len() > 0].reset_index(drop=True)
    df["input_id"] = df.index.map(lambda i: f"doc_{i:05d}")
    return df


def main():
    df = load_dataset()
    train_df, test_df = train_test_split(
        df, test_size=0.25, stratify=df["label"], random_state=RANDOM_STATE
    )

    train_df.to_csv(DATA_PROCESSED / "train.csv", index=False)
    test_df.to_csv(DATA_PROCESSED / "test.csv", index=False)

    summary = {
        "categories": NEWSGROUPS_CATEGORIES,
        "n_total": len(df),
        "n_train": len(train_df),
        "n_test": len(test_df),
        "class_balance_total": df["label"].value_counts(normalize=True).to_dict(),
    }
    with open(DATA_PROCESSED / "dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Loaded {summary['n_total']} documents "
          f"({summary['n_train']} train / {summary['n_test']} test)")
    print(f"Class balance: {summary['class_balance_total']}")


if __name__ == "__main__":
    main()
