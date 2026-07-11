from pathlib import Path

from src.config import load_config
from src.data.clean import clean_data
from src.data.load import load_raw
from src.data.split import split_data


def test_split_data_is_reproducible_and_returns_expected_shapes():
    data_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "telco.csv"
    config = load_config(Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    df = clean_data(load_raw(data_path))

    X_train, X_test, y_train, y_test, idx_train, idx_test = split_data(df, config)

    assert len(X_train) == len(y_train) == len(idx_train)
    assert len(X_test) == len(y_test) == len(idx_test)
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert set(y_train.unique()).issubset({"Yes", "No"})
    assert set(y_test.unique()).issubset({"Yes", "No"})


def test_split_has_no_customer_overlap_and_is_reproducible():
    data_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "telco.csv"
    config = load_config(Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    df = clean_data(load_raw(data_path))

    X_train1, X_test1, y_train1, y_test1, idx_train1, idx_test1 = split_data(df, config)
    X_train2, X_test2, y_train2, y_test2, idx_train2, idx_test2 = split_data(df, config)

    assert idx_train1 == idx_train2
    assert idx_test1 == idx_test2

    assert set(idx_train1).isdisjoint(set(idx_test1))

    total = len(idx_train1) + len(idx_test1)
    actual_test_ratio = len(idx_test1) / total
    assert abs(actual_test_ratio - config["split"]["test_size"]) < 0.01

    train_churn_rate = (y_train1 == "Yes").mean()
    test_churn_rate = (y_test1 == "Yes").mean()
    assert abs(train_churn_rate - test_churn_rate) < 0.02


def test_split_is_invariant_to_row_order():
    data_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "telco.csv"
    config = load_config(Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    df = clean_data(load_raw(data_path))

    shuffled_df = df.sample(frac=1, random_state=7).reset_index(drop=True)

    _, _, _, _, idx_train, idx_test = split_data(df, config)
    _, _, _, _, idx_train_shuffled, idx_test_shuffled = split_data(shuffled_df, config)

    train_ids = set(df.loc[idx_train, "customerID"])
    test_ids = set(df.loc[idx_test, "customerID"])
    train_ids_shuffled = set(shuffled_df.loc[idx_train_shuffled, "customerID"])
    test_ids_shuffled = set(shuffled_df.loc[idx_test_shuffled, "customerID"])

    assert train_ids == train_ids_shuffled
    assert test_ids == test_ids_shuffled
