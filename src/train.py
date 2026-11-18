import json
import logging
import os
import joblib
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from src.data_cleaner import load_and_clean_data
from src.features import build_encoder, transform_features

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("occupancy-train")


def log_event(event: str, **kwargs):
    logger.info(json.dumps({"event": event, **kwargs}))


def run_training_pipeline():
    # Step 1
    log_event("data_ingestion_started", step=1, message="Loading and cleaning datasets")
    train_df = load_and_clean_data("data/datatraining.txt")
    test_df = load_and_clean_data("data/datatest.txt")
    log_event("data_ingestion_completed", train_rows=len(train_df), test_rows=len(test_df))

    # Step 2
    log_event("feature_engineering_started", step=2, message="One-Hot Encoding datetime")
    encoder = build_encoder()
    X_train = transform_features(train_df, encoder, fit=True)
    y_train = train_df["Occupancy"]
    X_test = transform_features(test_df, encoder, fit=False)
    y_test = test_df["Occupancy"]
    log_event("feature_engineering_completed", step=2, message="One-Hot Encoding completed")

    # Step 3
    log_event("hyperparameter_search_started", step=3, message="Running RandomizedSearchCV")
    base_model = xgb.XGBClassifier(eval_metric="logloss", random_state=42)

    param_distributions = {
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "n_estimators": [50, 100, 150, 200],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=10,
        scoring="roc_auc",
        cv=3,
        random_state=42,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    log_event("hyperparameter_search_completed", best_params=search.best_params_)

    # Step 4
    log_event("holdout_evaluation_started", step=4, message="Evaluating model on holdout test set")
    test_probs = best_model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, test_probs)
    log_event("holdout_evaluation_completed", step=4, roc_auc=round(float(auc_score), 4))

    # Step 5
    log_event(
        "serializing_model_and_encoder_started",
        step=5,
        message="Serializing immutable model and encoder artifacts")
    os.makedirs("models", exist_ok=True)
    best_model.save_model("models/occupancy_xgb.json")
    joblib.dump(encoder, "models/encoder.joblib")
    log_event(
        "serializing_model_and_encoder_completed",
        step=5,
        message="Training pipeline complete. Artifacts saved to models/")


if __name__ == "__main__":
    run_training_pipeline()
