# train_logreg.py

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from data_prep import create_dataset

def main():
    ticker = "SPY"
    start = "2015-01-01"
    end = "2024-12-31"

    X_train, X_test, y_train, y_test = create_dataset(
        ticker=ticker,
        start=start,
        end=end,
        test_size_years=2,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            )),
        ]
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"Logistic Regression trained on {ticker}")
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, "logreg_model.pkl")
    print("Saved model to logreg_model.pkl")
    
    f1_up = f1_score(y_test, y_pred, pos_label=1)
    joblib.dump(f1_up, "logreg_f1.pkl")

if __name__ == "__main__":
    main()
