import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from src.shipment_prep import FEATURES, clean_shipments

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("data/modelling_table.csv")
X = df[FEATURES]
y = df["is_delayed"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

config = pd.read_csv("outputs/part_c_final_config.csv").iloc[0]

max_depth_value = config["max_depth"]
max_depth = (
    None
    if pd.isna(max_depth_value) or str(max_depth_value).lower() in ("unlimited", "none", "nan", "")
    else int(float(max_depth_value))
)

model = Pipeline([
    ("selector", SelectKBest(score_func=f_classif, k=int(config["selector_k"]))),
    ("model", RandomForestClassifier(
        n_estimators=int(config["n_estimators"]),
        max_depth=max_depth,
        min_samples_leaf=int(config["min_samples_leaf"]),
        random_state=42
    ))
])

model.fit(X_train, y_train)
joblib.dump(model, "models/delay_model.pkl")

threshold = float(config["threshold"])

pd.DataFrame({
    "threshold": [threshold]
}).to_csv(
    "models/delay_threshold.csv",
    index=False
)

pd.DataFrame({
    "feature": FEATURES
}).to_csv(
    "models/feature_order.csv",
    index=False
)

raw_df = pd.read_csv("data/raw_shipments.csv")
_, medians = clean_shipments(raw_df)

pd.DataFrame({
    "column": list(medians.keys()),
    "median": list(medians.values())
}).to_csv(
    "models/impute_values.csv",
    index=False
)

loaded_model = joblib.load("models/delay_model.pkl")

original_probabilities = model.predict_proba(X_test)[:, 1]
reloaded_probabilities = loaded_model.predict_proba(X_test)[:, 1]

reload_difference = np.max(
    np.abs(original_probabilities - reloaded_probabilities)
)

reload_check = pd.DataFrame({
    "metric": [
        "test_rows_scored",
        "reload_max_abs_probability_diff",
        "threshold"
    ],
    "value": [
        len(X_test),
        reload_difference,
        threshold
    ]
})

reload_check["value"] = reload_check["value"].round(4)

reload_check.to_csv(
    "outputs/reload_check.csv",
    index=False
)

print("Model packaging completed.")
print(reload_check)
