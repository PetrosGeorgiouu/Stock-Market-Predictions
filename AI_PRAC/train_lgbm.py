# train_lightgbm_multi.py

import joblib
import pandas as pd
from collections import Counter
from sklearn.metrics import accuracy_score, classification_report
from lightgbm import LGBMClassifier
import lightgbm as lgb
from data_prep import (
    download_price_data,
    add_technical_features,
    add_labels,
    build_feature_matrix_and_labels,
    split_train_test_by_date,
)
USE_MULTI_TICKERS = True
MULTI_TICKERS = ["SPY", "QQQ", "VOO", "VTI", "IVV"]
def split_train_val_by_date(df_train, val_years=1):
    last_date = df_train.index.max()
    cutoff_year = last_date.year - val_years + 1
    cutoff_date = pd.Timestamp(year=cutoff_year, month=1, day=1)

    return (
        df_train[df_train.index < cutoff_date],
        df_train[df_train.index >= cutoff_date],
    )

def build_multi_ticker_dataset(tickers, start, end, test_size_years=2):
    dfs = []

    for t in tickers:
        df = download_price_data(t, start, end)
        df = add_technical_features(df)
        df = add_labels(df)   # next-day label
        df["ticker"] = t
        dfs.append(df)

    big_df = pd.concat(dfs).sort_index()
    big_df = pd.get_dummies(big_df, columns=["ticker"], drop_first=False)
    df_train, df_test = split_train_test_by_date(big_df, test_size_years)
    return df_train, df_test
def main():
    start = "2015-01-01"
    end = "2024-12-31"

    df_train, df_test = build_multi_ticker_dataset(
        MULTI_TICKERS, start, end, test_size_years=2
    )

    df_tr, df_val = split_train_val_by_date(df_train, val_years=1)

    X_tr, y_tr = build_feature_matrix_and_labels(df_tr)
    X_val, y_val = build_feature_matrix_and_labels(df_val)
    X_test, y_test = build_feature_matrix_and_labels(df_test)

    print("\nTrain class distribution:", Counter(y_tr))
    print("Validation class distribution:", Counter(y_val))
    print("Test class distribution:", Counter(y_test))

    model = LGBMClassifier(
        objective="binary",
        n_estimators=5000,
        learning_rate=0.02,
        num_leaves=15,
        min_child_samples=200,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_lambda=5.0,
        reg_alpha=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(200, verbose=False)],
    )

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\nTest accuracy:" , accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["DOWN", "UP"]))
    #baseline = max(Counter(y_test).values()) / len(y_test)
    #print(f"\nBaseline accuracy: {baseline:.3f}")
    joblib.dump(model, "lgbm_model.pkl")
    print("\nSaved model to lgbm_model.pkl")

if __name__ == "__main__":
    main()
