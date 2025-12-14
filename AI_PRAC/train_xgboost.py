# train_xgboost.py

import joblib
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from data_prep import (
    create_dataset,
    download_price_data,
    add_technical_features,
    add_labels,
    build_feature_matrix_and_labels,
    split_train_test_by_date,
)
#-----------------------
USE_MULTI_TICKERS = True

# If using multi-ticker training, these are the tickers we'll include.
MULTI_TICKERS = ["SPY", "QQQ", "VOO", "VTI", "IVV"]


def build_single_ticker_dataset(ticker: str, start: str, end: str, test_size_years: int = 2):
    """
    Uses the existing create_dataset() pipeline for a single ticker.
    This is your original SPY-only training mode.
    """
    X_train, X_test, y_train, y_test = create_dataset(
        ticker=ticker,
        start=start,
        end=end,
        test_size_years=test_size_years,
    )
    print(f"\nDataset built using ONLY {ticker}.")
    print(f"Train size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


def build_multi_ticker_dataset(tickers, start: str, end: str, test_size_years: int = 2):
    """
    Build a combined dataset from multiple tickers.

    Steps per ticker:
    - download price data
    - add technical features
    - add labels (next-day up/down)
    - append to a big DataFrame

    Then:
    - split by date into train/test using the combined DataFrame
    - build feature matrices and labels
    """
    dfs = []
    for t in tickers:
        try:
            df = download_price_data(t, start=start, end=end)
            df = add_technical_features(df)
            df = add_labels(df)
            df["ticker"] = t  
            dfs.append(df)
            print(f"✓ Processed {t}, rows: {len(df)}")
        except Exception as e:
            print(f"Skipping {t} due to error: {e}")

    big_df = pd.concat(dfs).sort_index()
    # Split by date across the combined dataset
    df_train, df_test = split_train_test_by_date(big_df, test_size_years=test_size_years)

    X_train, y_train = build_feature_matrix_and_labels(df_train)
    X_test, y_test = build_feature_matrix_and_labels(df_test)

    print("\nCombined multi-ticker dataset built.")
    print(f"Tickers used: {tickers}")
    print(f"Total train samples: {X_train.shape[0]}  |  Total test samples: {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test

def main():
    start = "2015-01-01"
    end = "2024-12-31"

    if USE_MULTI_TICKERS:
        # Multi-ticker training mode 
        X_train, X_test, y_train, y_test = build_multi_ticker_dataset(
            MULTI_TICKERS, start=start, end=end, test_size_years=2
        )
        mode_label = f"XGBoost trained on multiple tickers: {', '.join(MULTI_TICKERS)}"
    else:
        # Original single-ticker (SPY-only) mode
        ticker = "SPY"
        X_train, X_test, y_train, y_test = build_single_ticker_dataset(
            ticker=ticker, start=start, end=end, test_size_years=2
        )
        mode_label = f"XGBoost trained on {ticker} only"
    #class imbalance handling
    class_counts = Counter(y_train)
    if class_counts[1] > 0:
        scale_pos_weight = class_counts[0] / class_counts[1]
    else:
        scale_pos_weight = 1.0
    print(f"\n{mode_label}")
    print(f"Class distribution in training data: {class_counts}")
    print(f"Computed scale_pos_weight: {scale_pos_weight:.3f}")

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("xgb", XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                scale_pos_weight=scale_pos_weight,
            )),
        ]
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nTest accuracy: {acc:.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, "xgb_model.pkl")
    print("\nSaved model to xgb_model.pkl")

    f1_up = f1_score(y_test, y_pred, pos_label=1)
    joblib.dump(f1_up, "xgb_f1.pkl")

if __name__ == "__main__":
    main()
