import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.shipment_prep import FEATURES
os.makedirs("outputs", exist_ok=True)
df = pd.read_csv("data/modelling_table.csv")
X = df[FEATURES]
y = df["delivery_hours"]
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
checks = pd.DataFrame({
    "metric": [
        "train_rows",
        "val_rows",
        "test_rows",
        "train_scaled_max_abs_mean",
        "test_scaled_max_abs_mean"
    ],
    "value": [
        len(X_train),
        len(X_val),
        len(X_test),
        np.abs(X_train_scaled.mean(axis=0)).max(),
        np.abs(X_test_scaled.mean(axis=0)).max()
    ]
})
checks["value"] = checks["value"].round(4)
checks.to_csv(
    "outputs/part_b_checks.csv",
    index=False
)
def calculate_metrics(model, X_data, y_data):
    predictions = model.predict(X_data)
    mse = mean_squared_error(y_data, predictions)
    return {
        "MAE": mean_absolute_error(y_data, predictions),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(y_data, predictions)
    }
models = {
    "linear_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),
    "random_forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),
    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )
}
tree_depths = [1, 2, 3, 4, 6, 8, 12, None]
tree_results = []
for depth in tree_depths:
    tree = DecisionTreeRegressor(
        max_depth=depth,
        random_state=42
    )
    tree.fit(X_train, y_train)
    train_metrics = calculate_metrics(
        tree, X_train, y_train
    )
    val_metrics = calculate_metrics(
        tree, X_val, y_val
    )
    tree_results.append({
        "max_depth": depth if depth is not None else "unlimited",
        "train_rmse": train_metrics["RMSE"],
        "val_rmse": val_metrics["RMSE"]
    })
tree_results_df = pd.DataFrame(tree_results)
tree_results_df[
    ["train_rmse", "val_rmse"]
] = tree_results_df[
    ["train_rmse", "val_rmse"]
].round(4)
tree_results_df.to_csv(
    "outputs/part_b_tree_depth.csv",
    index=False
)
best_depth_row = tree_results_df.loc[
    tree_results_df["val_rmse"].idxmin()
]
best_depth = best_depth_row["max_depth"]
if best_depth == "unlimited":
    best_depth = None
else:
    best_depth = int(best_depth)
models["decision_tree"] = DecisionTreeRegressor(
    max_depth=best_depth,
    random_state=42
)
gb_results = []
for rate in [0.01, 0.1, 1.0]:
    gb_model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=rate,
        random_state=42
    )
    gb_model.fit(X_train, y_train)
    metrics = calculate_metrics(
        gb_model, X_val, y_val
    )
    gb_results.append({
        "learning_rate": rate,
        "val_rmse": metrics["RMSE"],
        "val_r2": metrics["R2"]
    })
gb_results_df = pd.DataFrame(gb_results)
gb_results_df[
    ["val_rmse", "val_r2"]
] = gb_results_df[
    ["val_rmse", "val_r2"]
].round(4)
gb_results_df.to_csv(
    "outputs/part_b_gb_learning_rate.csv",
    index=False
)
baseline_prediction = np.full(
    len(y_val),
    y_train.mean()
)
baseline_mse = mean_squared_error(
    y_val,
    baseline_prediction
)
validation_results = [{
    "model": "mean_baseline",
    "split": "validation",
    "MAE": mean_absolute_error(y_val, baseline_prediction),
    "MSE": baseline_mse,
    "RMSE": np.sqrt(baseline_mse),
    "R2": r2_score(y_val, baseline_prediction)
}]
fitted_models = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    fitted_models[name] = model
    metrics = calculate_metrics(
        model, X_val, y_val
    )
    validation_results.append({
        "model": name,
        "split": "validation",
        **metrics
    })
metrics_df = pd.DataFrame(validation_results)
non_baseline = metrics_df[
    metrics_df["model"] != "mean_baseline"
]
top_two = non_baseline.nsmallest(
    2,
    "RMSE"
)["model"].tolist()
cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
cv_results = []
for name in top_two:
    scores = cross_val_score(
        models[name],
        X_train,
        y_train,
        cv=cv,
        scoring="r2"
    )
    cv_results.append({
        "model": name,
        "mean_r2": scores.mean(),
        "std_r2": scores.std()
    })
cv_df = pd.DataFrame(cv_results)
cv_df[
    ["mean_r2", "std_r2"]
] = cv_df[
    ["mean_r2", "std_r2"]
].round(4)
cv_df.to_csv(
    "outputs/part_b_cv.csv",
    index=False
)
best_model_name = non_baseline.loc[
    non_baseline["RMSE"].idxmin(),
    "model"
]
best_model = fitted_models[best_model_name]
test_metrics = calculate_metrics(
    best_model, X_test, y_test
)
metrics_df = pd.concat([
    metrics_df,
    pd.DataFrame([{
        "model": best_model_name,
        "split": "test",
        **test_metrics
    }])
], ignore_index=True)
metric_columns = ["MAE", "MSE", "RMSE", "R2"]
metrics_df[metric_columns] = metrics_df[
    metric_columns
].round(4)
metrics_df.to_csv(
    "outputs/part_b_metrics.csv",
    index=False
)
print("Part B completed successfully.")
print("Best regression model:", best_model_name)
print("Final test R2:", round(test_metrics["R2"], 4))
