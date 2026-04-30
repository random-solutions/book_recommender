import numpy as np
import pandas as pd


def load_data():
    df = pd.read_csv("Ratings.csv")

    # userid, rating
    assert all(df.userid == df.userid.astype(np.int32))
    assert all(df.rating == df.rating.astype(np.int8))

    df["userid"] = df.userid.astype(np.int32)
    df["rating"] = df.rating.astype(np.int8)

    # isbn
    df["isbn"] = df.isbn.str.upper()
    df["isbn"] = df.isbn.str.replace(r"[^0-9X]", "", regex=True)
    df["isbn"] = df.isbn.str.replace(r"^(.)\1*$", "", regex=True)
    df = df[df.isbn.str.fullmatch(r"\d+[X]?")]
    df = df[df.isbn.str.len().isin([10, 13])]
    df = df[~((df.isbn.str.len() == 13) & df.isbn.str.contains("X"))]
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset=["userid", "isbn"], keep=False)

    isbn10 = df.loc[df.isbn.map(len) == 10, "isbn"].str.split("", expand=True)
    isbn10 = isbn10.iloc[:, 1:-1].replace("X", 10).astype(np.int8)
    weights10 = np.arange(1, 11, dtype=np.int8)

    isbn13 = df.loc[df.isbn.map(len) == 13, "isbn"].str.split("", expand=True)
    isbn13 = isbn13.iloc[:, 1:-1].astype(np.int8)
    weights13 = np.array([1, 3] * 6 + [1], dtype=np.int8)

    df = df.drop(isbn10.index[(isbn10 * weights10).sum(axis=1) % 11 != 0])
    df = df.drop(isbn13.index[(isbn13 * weights13).sum(axis=1) % 10 != 0])

    df = df.loc[df.sort_values(by="isbn").iloc[21:-12].index]
    df = df.sort_values(["userid", "isbn", "rating"]).reset_index(drop=True)

    return df
