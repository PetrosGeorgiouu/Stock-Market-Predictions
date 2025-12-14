# train_linreg.py

import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from data_prep import create_regression_dataset

def main():
    ticker = "SPY"
    start = "2015-01-01"
    end = "2024-12-31"

    X_train, X_test, y_train, y_test = create_regression_dataset(
        ticker=ticker,
        start=start,
        end=end,
        test_size_years=2,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("linreg", LinearRegression()),
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Linear Regression trained on {ticker}")
    print(f"Test Mean Squared Error: {mse:.6f}")
    print(f"Test R^2: {r2:.4f}")

    joblib.dump(model, "linreg_model.pkl")
    print("Saved model to linreg_model.pkl")
    
    y_pred_class = (y_pred > 0).astype(int)
    y_test_class = (y_test > 0).astype(int)

    f1_up = f1_score(y_test_class, y_pred_class, pos_label=1)
    joblib.dump(f1_up, "linreg_f1.pkl")


if __name__ == "__main__":
    main()
