import argparse
import random
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from preprocess import load_data, split_data, scale_features


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def main(args):

    set_seed(args.seed)

    # load dataset
    X, y = load_data()

    # split dataset
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, args.seed)

    # scale features
    X_train, X_val, X_test = scale_features(X_train, X_val, X_test)

    # model
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.seed,
        class_weight="balanced"
    )

    # train
    model.fit(X_train, y_train)

    # validation predictions
    preds = model.predict(X_val)
    probs = model.predict_proba(X_val)[:,1]

    # metrics
    f1 = f1_score(y_val, preds)
    roc = roc_auc_score(y_val, probs)

    print(f"VAL_F1={f1:.4f} VAL_ROC_AUC={roc:.4f}")

    # log experiment
    with open("experiments.log", "a") as f:
        f.write(
            f"n_estimators={args.n_estimators}, "
            f"max_depth={args.max_depth}, "
            f"seed={args.seed}, "
            f"F1={f1:.4f}, "
            f"ROC_AUC={roc:.4f}\n"
        )

    # save 
    joblib.dump(model, "model.joblib")
    print("Model saved to model.joblib")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    main(args)