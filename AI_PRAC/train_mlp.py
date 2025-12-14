# train_mlp.py
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score
import numpy as np
from sklearn.utils import resample
from data_prep import create_dataset

def balance_train(X, y, seed=42):
    X = np.asarray(X)
    y = np.asarray(y)

    X0 = X[y == 0]; y0 = y[y == 0]
    X1 = X[y == 1]; y1 = y[y == 1]

    n0, n1 = len(y0), len(y1)
    n = min(n0, n1)
    X0b, y0b = resample(X0, y0, replace=False, n_samples=n, random_state=seed)
    X1b, y1b = resample(X1, y1, replace=False, n_samples=n, random_state=seed)
    Xb = np.vstack([X0b, X1b])
    yb = np.hstack([y0b, y1b])

    perm = np.random.permutation(len(yb))
    return Xb[perm], yb[perm]

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
    print(f"MLP: Train size = {X_train.shape[0]}, Test size = {X_test.shape[0]}")

    base_mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),
        activation="relu",
        #solver="adam",
        alpha=0.01,
        learning_rate_init=0.001,
        batch_size=32,
        max_iter=1000, 
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,  
        #random_state=42,
        verbose=False,
        
    )
    tscv = TimeSeriesSplit(n_splits=3)
    calibrated_mlp = CalibratedClassifierCV(
        estimator=base_mlp,
        cv=tscv,
        method="sigmoid",
    )
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("mlp", calibrated_mlp),
        ]
    )

    # train
    X_train_bal, y_train_bal = balance_train(X_train, y_train)
    model.fit(X_train_bal, y_train_bal)

    # evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nMLP (Neural Network) test accuracy: {acc:.3f}\n")
    print("Classification report:")
    print(classification_report(y_test, y_pred))
    
    joblib.dump(model, "mlp_model.pkl")
    print("Saved MLP model to mlp_model.pkl")
    
    f1_up = f1_score(y_test, y_pred, pos_label=1)
    joblib.dump(f1_up, "mlp_f1.pkl")

if __name__ == "__main__":
    main()
