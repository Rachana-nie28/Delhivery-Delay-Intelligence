import os
import pandas as pd
import numpy as np

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from src.shipment_prep import FEATURES, NOISE_COLUMNS

os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("data/modelling_table.csv")
X = df[FEATURES]
y = df["is_delayed"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

pd.DataFrame({
    "metric": [
        "train_rows", "validation_rows", "test_rows",
        "train_delayed", "validation_delayed", "test_delayed"
    ],
    "value": [
        len(X_train), len(X_val), len(X_test),
        int(y_train.sum()), int(y_val.sum()), int(y_test.sum())
    ]
}).to_csv("outputs/part_c_checks.csv", index=False)


def get_metrics(y_true, predictions):
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0)
    }


def evaluate_validation(model):
    predictions = model.predict(X_val)
    return get_metrics(y_val, predictions)


baseline_predictions = np.zeros(len(y_val), dtype=int)

model_results = [{
    "model": "blind_all_on_time",
    **get_metrics(y_val, baseline_predictions)
}]



logistic = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=2000, random_state=42))
])

logistic.fit(X_train, y_train)

model_results.append({
    "model": "logistic_regression",
    **evaluate_validation(logistic)
})


tree_results = []

for depth in [2, 3, 4, 5, 6, 8, 10]:
    tree = DecisionTreeClassifier(
        criterion="gini",
        max_depth=depth,
        random_state=42
    )

    tree.fit(X_train, y_train)
    metrics = evaluate_validation(tree)

    tree_results.append({
        "depth": depth,
        **metrics
    })

tree_depth_df = pd.DataFrame(tree_results)

tree_depth_df.to_csv(
    "outputs/part_c_tree_depth.csv", index=False
)

best_tree_row = tree_depth_df.sort_values(
    ["f1", "depth"],
    ascending=[False, True]
).iloc[0]

best_tree_depth = int(best_tree_row["depth"])

best_tree = DecisionTreeClassifier(
    criterion="gini",
    max_depth=best_tree_depth,
    random_state=42
)

best_tree.fit(X_train, y_train)

model_results.append({
    "model": f"decision_tree_depth_{best_tree_depth}",
    **evaluate_validation(best_tree)
})


default_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

default_forest.fit(X_train, y_train)

model_results.append({
    "model": "random_forest_default",
    **evaluate_validation(default_forest)
})

pd.DataFrame(model_results)[
    ["model", "TN", "FP", "FN", "TP",
     "accuracy", "precision", "recall", "f1"]
].round(4).to_csv(
    "outputs/part_c_models.csv", index=False
)


selector_all = SelectKBest(score_func=f_classif, k="all")
selector_all.fit(X_train, y_train)

feature_scores = pd.DataFrame({
    "feature": FEATURES,
    "f_score": np.nan_to_num(selector_all.scores_, nan=0.0),
    "rf_importance": default_forest.feature_importances_,
    "is_noise": [feature in NOISE_COLUMNS for feature in FEATURES]
})

feature_scores["f_rank"] = (
    feature_scores["f_score"]
    .rank(method="min", ascending=False)
    .astype(int)
)

feature_scores = feature_scores[
    ["feature", "f_score", "f_rank", "rf_importance", "is_noise"]
].sort_values("f_rank")

feature_scores.round(4).to_csv(
    "outputs/part_c_feature_scores.csv", index=False
)

noise_importance_share = feature_scores.loc[
    feature_scores["is_noise"], "rf_importance"
].sum()

pd.DataFrame([{
    "noise_importance_share": noise_importance_share
}]).round(4).to_csv(
    "outputs/part_c_noise_share.csv", index=False
)


selectk_results = []

for k in [5, 7, 10, 12, 17]:
    model = Pipeline([
        ("selector", SelectKBest(score_func=f_classif, k=k)),
        ("model", RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    selectk_results.append({
        "k": k,
        "val_f1": f1_score(
            y_val, predictions, zero_division=0
        )
    })

pd.DataFrame(selectk_results).round(4).to_csv(
    "outputs/part_c_selectk.csv", index=False
)


selector = SelectKBest(score_func=f_classif, k=7)

X_train_selected = selector.fit_transform(X_train, y_train)
X_val_selected = selector.transform(X_val)


parameter_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [4, 6, 8],
    "min_samples_leaf": [1, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid=parameter_grid,
    cv=5,
    n_jobs=-1
)

grid_search.fit(X_train_selected, y_train)

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=parameter_grid,
    n_iter=9,
    cv=5,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train_selected, y_train)


default_seven = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

default_seven.fit(X_train_selected, y_train)

def validation_f1(model):
    probabilities = model.predict_proba(X_val_selected)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return f1_score(y_val, predictions, zero_division=0)
tuning_results = [
    {
        "search": "grid_search",
        "candidates": 27,
        "cv_folds": 5,
        "total_fits": 135,
        "best_params": str(grid_search.best_params_),
        "best_cv_accuracy": grid_search.best_score_,
        "val_f1": validation_f1(grid_search.best_estimator_)
    },
    {
        "search": "randomized_search",
        "candidates": 9,
        "cv_folds": 5,
        "total_fits": 45,
        "best_params": str(random_search.best_params_),
        "best_cv_accuracy": random_search.best_score_,
        "val_f1": validation_f1(random_search.best_estimator_)
    },
    {
        "search": "default_forest_same_7_features",
        "candidates": 1,
        "cv_folds": 0,
        "total_fits": 1,
        "best_params": "defaults",
        "best_cv_accuracy": np.nan,
        "val_f1": validation_f1(default_seven)
    }
]

tuning_df = pd.DataFrame(tuning_results)

tuning_df.to_csv(
    "outputs/part_c_tuning.csv", index=False
)


priority = {
    "default_forest_same_7_features": 0,
    "grid_search": 1,
    "randomized_search": 2
}

tuning_df["tie_priority"] = tuning_df["search"].map(priority)

tuning_df = tuning_df.sort_values(
    ["val_f1", "tie_priority"],
    ascending=[False, True]
)

chosen_from = tuning_df.iloc[0]["search"]

if chosen_from == "grid_search":
    chosen_params = grid_search.best_params_
elif chosen_from == "randomized_search":
    chosen_params = random_search.best_params_
else:
    chosen_params = {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_leaf": 1
    }


final_model = Pipeline([
    ("selector", SelectKBest(score_func=f_classif, k=7)),
    ("model", RandomForestClassifier(
        random_state=42,
        **chosen_params
    ))
])

final_model.fit(X_train, y_train)

val_probabilities = final_model.predict_proba(X_val)[:, 1]

threshold_results = []

for threshold in [
    0.1, 0.2, 0.3, 0.4, 0.5,
    0.6, 0.7, 0.8, 0.9
]:
    predictions = (val_probabilities >= threshold).astype(int)

    threshold_results.append({
        "threshold": threshold,
        **get_metrics(y_val, predictions)
    })

threshold_df = pd.DataFrame(threshold_results)

threshold_df.round(4).to_csv(
    "outputs/part_c_thresholds.csv", index=False
)

eligible = threshold_df[threshold_df["recall"] >= 0.80]

if eligible.empty:
    chosen_threshold = 0.1
else:
    chosen_threshold = float(eligible["threshold"].max())


test_probabilities = final_model.predict_proba(X_test)[:, 1]
test_predictions = (
    test_probabilities >= chosen_threshold
).astype(int)

test_metrics = {
    "threshold": chosen_threshold,
    **get_metrics(y_test, test_predictions)
}

pd.DataFrame([test_metrics]).round(4).to_csv(
    "outputs/part_c_final_test.csv", index=False
)


max_depth_value = chosen_params["max_depth"]

final_config = pd.DataFrame([{
    "selector_k": 7,
    "n_estimators": chosen_params["n_estimators"],
    "max_depth": (
        "unlimited" if max_depth_value is None
        else max_depth_value
    ),
    "min_samples_leaf": chosen_params["min_samples_leaf"],
    "threshold": chosen_threshold,
    "chosen_from": chosen_from
}])

final_config.to_csv(
    "outputs/part_c_final_config.csv", index=False
)

print("Part C completed successfully.")
print("Selected tuning approach:", chosen_from)
print("Selected threshold:", chosen_threshold)
print("Noise importance share:", round(noise_importance_share, 4))
