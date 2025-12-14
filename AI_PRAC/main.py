# main.py

from os import path
import joblib
import yfinance as yf
import pandas as pd
import numpy as np
import joblib

from data_prep import add_technical_features, build_feature_matrix_and_labels
def prepare_recent_data(ticker):
    """
    Downloads recent price data and prepares the most recent row of features.
    This mirrors the feature pipeline used during training.
    """
    df = yf.download(
        ticker,
        period="6mo",    
        interval="1d",
        progress=False,
        auto_adjust=False,
    )
    df = df.dropna()

    df = add_technical_features(df)
    df = df.dropna()

    feature_cols = [
    "log_return_1d",
    "pct_return_1d",
    "pct_return_5d",
    "pct_return_10d",
    "vol_5d",
    "vol_10d",
    "vol_20d",
    "close_over_ma5",
    "close_over_ma10",
    "close_over_ma20",
    "volume_zscore_20",
    "volume_change",
    #"rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_width",
    "lag_ret_1",
    "lag_ret_2",
    "lag_vol_1",
    ]
    latest_row = df.iloc[-1][feature_cols].values.reshape(1, -1)
    return latest_row

def interpret_prediction(pred):

    #Converts predicted class 0/1 into a BUY or HOLD recommendation.
  
    if pred == 1:
        return "BUY (model predicts upward price movement)"
    else:
        return "HOLD (model predicts downward or uncertain movement)"


def main():
    # Load both models
    logreg_model = joblib.load("logreg_model.pkl")
    linreg_model = joblib.load("linreg_model.pkl")
    xgb_model = joblib.load("xgb_model.pkl")
    lgbm_model = joblib.load("lgbm_model.pkl")

    ticker = input("Enter a stock ticker (e.g., AAPL): ").upper()

    print(f"\nFetching recent (past 6 months) data for {ticker}...")
    X_latest = prepare_recent_data(ticker)

    # ----- LOGISTIC REGRESSION PREDICTION (classification) -----
    logreg_pred = logreg_model.predict(X_latest)[0]
    logreg_prob_up = logreg_model.predict_proba(X_latest)[0][1]

    logreg_name = "Logistic Regression"

    print("\n===== {} Prediction Result =====".format(logreg_name))

    if logreg_pred == 1:
        logreg_direction_text = "Price is likely to go UP"
        logreg_recommendation_text = "BUY (model expects upward movement)"
    else:
        logreg_direction_text = "Price is likely to go DOWN"
        logreg_recommendation_text = "HOLD (model expects downward or uncertain movement)"

    print(f"- Price Direction: {logreg_direction_text}")
    print(f"- Confidence: {logreg_prob_up:.2%} chance stock will rise tomorrow")
    print(f"- Recommendation: {logreg_recommendation_text}")
    print("==========================================")

# ----- LINEAR REGRESSION PREDICTION (regression/expected return) -----
    linreg_pred = linreg_model.predict(X_latest)[0]  
    linreg_name = "Linear Regression"
    print("\n===== {} Prediction Result =====".format(linreg_name))
    pred_return_pct = linreg_pred * 100
    if linreg_pred > 0:
        linreg_direction_text = "Price is expected to INCREASE (positive expected return)"
        linreg_recommendation_text = "BUY (model expects positive return)"
    else:
        linreg_direction_text = "Price is expected to DECREASE or stay steady (non-positive expected return)"
        linreg_recommendation_text = "HOLD (model does not expect a clear positive return)"

    print(f"- Expected Return: {pred_return_pct:.2f}% for the next trading day (anticipated profit/loss)")
    print(f"- Interpretation: {linreg_direction_text}")
    print(f"- Recommendation: {linreg_recommendation_text}")
    print("==========================================\n")
    
# ===== XGBOOST (classification) =====
    xgb_pred = xgb_model.predict(X_latest)[0]
    xgb_prob_up = xgb_model.predict_proba(X_latest)[0][1]
    xgb_name = "XGBoost"
    print("\n===== {} Prediction Result =====".format(xgb_name))

    if xgb_pred == 1:
        xgb_direction_text = "Price is likely to go UP"
        xgb_recommendation_text = "BUY (model expects upward movement)"
    else:
        xgb_direction_text = "Price is likely to go DOWN"
        xgb_recommendation_text = "HOLD (model expects downward or uncertain movement)"
    print(f"- Price Direction: {xgb_direction_text}")
    print(f"- Confidence: {xgb_prob_up:.2%} chance stock will rise tomorrow")
    print(f"- Recommendation: {xgb_recommendation_text}")
    print("==========================================")

 # ===== MLP (Neural Network) Prediction =====
    try:
        mlp_model = joblib.load("mlp_model.pkl")
        mlp_proba = mlp_model.predict_proba(X_latest.reshape(1, -1))[0, 1]
        mlp_class = int(mlp_proba >= 0.55)

        print("\n")
        print("===== MLP Neural Network Prediction Result =====")
        if mlp_class == 1:
            print("- Price Direction: Price is likely to go UP")
        else:
            print("- Price Direction: Price is likely to go DOWN")
        print(f"- Confidence: {mlp_proba * 100:.2f}% chance stock will rise tomorrow")
        if mlp_class == 1:
            print("- Recommendation: BUY (model expects upward movement)")
        else:
            print("- Recommendation: HOLD (model expects downward or uncertain movement)")
        print("==========================================")
    except FileNotFoundError:
        print("\nMLP model not found. Run train_mlp.py to train and save it.")

# ===== LIGHTGBM (classification) =====
    lgbm_pred = lgbm_model.predict(X_latest)[0]
    lgbm_prob_up = lgbm_model.predict_proba(X_latest)[0][1]
    lgbm_name = "LightGBM"
    print("\n===== {} Prediction Result =====".format(lgbm_name))
    if lgbm_pred == 1:
        lgbm_direction_text = "Price is likely to go UP"
        lgbm_recommendation_text = "BUY (model expects upward movement)"
    else:
        lgbm_direction_text = "Price is likely to go DOWN"
        lgbm_recommendation_text = "HOLD (model expects downward or uncertain movement)"
    print(f"- Price Direction: {lgbm_direction_text}")
    print(f"- Confidence: {lgbm_prob_up:.2%} chance stock will rise tomorrow")
    print(f"- Recommendation: {lgbm_recommendation_text}")
    print("==========================================\n")

# ===== CONSENSUS SUMMARY =====
    predictions = [linreg_pred, logreg_pred, xgb_pred, mlp_class, lgbm_pred]
    probabilities = [linreg_pred, logreg_prob_up, xgb_prob_up, mlp_proba, lgbm_prob_up]
    
    buy_votes = sum(predictions)
    avg_confidence = np.mean(probabilities)
    
    print("\n===== CONSENSUS SUMMARY =====")
    print(f"Models voting BUY: {round(buy_votes)} out of 5")  
    print(f"Average confidence: {avg_confidence:.2%}")
    if buy_votes >= 3:
        print(f"CONSENSUS: BUY (majority agreement)")
    else:
        print(f"CONSENSUS: HOLD (majority expects downward/uncertain movement)")
    print("==========================================\n")

if __name__ == "__main__":
    main()
