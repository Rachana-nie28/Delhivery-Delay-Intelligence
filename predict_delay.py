import joblib
import numpy as np
import pandas as pd
from src.shipment_prep import clean_shipments, build_features
new_raw_df = pd.read_csv("data/new_shipments.csv")
hubs_df = pd.read_csv("data/hubs.csv")
stored_medians_df = pd.read_csv("models/impute_values.csv")
stored_medians = dict(zip(
    stored_medians_df["column"],
    stored_medians_df["median"]
))
clean_new_df, _ = clean_shipments(
    new_raw_df,
    stored_medians
)
new_feature_df = build_features(
    clean_new_df,
    hubs_df
)
feature_order = pd.read_csv(
    "models/feature_order.csv"
)["feature"].tolist()
X_new = new_feature_df[feature_order]
loaded_model = joblib.load(
    "models/delay_model.pkl"
)
threshold = pd.read_csv(
    "models/delay_threshold.csv"
)["threshold"].iloc[0]
probabilities = loaded_model.predict_proba(X_new)[:, 1]
rounded_probabilities = np.round(
    probabilities,
    4
)
flagged = (
    rounded_probabilities >= threshold
).astype(int)

hub_segments_df = pd.read_csv(
    "outputs/hub_segments.csv"
)
new_actions_df = pd.DataFrame({
    "shipment_id": clean_new_df["shipment_id"].values,
    "hub_id": clean_new_df["hub_id"].values,
    "delay_probability": rounded_probabilities,
    "flagged": flagged
})
new_actions_df = new_actions_df.merge(
    hub_segments_df[["hub_id", "segment"]],
    on="hub_id",
    how="left",
    validate="many_to_one",
    sort=False
)
new_actions_df["action"] = np.select(
    [
        (new_actions_df["flagged"] == 1) &
        (new_actions_df["segment"] == "Escalate"),
        new_actions_df["flagged"] == 1
    ],
    [
        "Escalate to hub manager",
        "Proactive customer call"
    ],
    default="No action"
)
new_actions_df = new_actions_df[
    [
        "shipment_id",
        "hub_id",
        "segment",
        "delay_probability",
        "flagged",
        "action"
    ]
]
new_actions_df.to_csv(
    "outputs/new_shipment_actions.csv",
    index=False
)
print("Scoring completed.")
print("Total shipments:", len(new_actions_df))
print("Flagged shipments:", int(new_actions_df["flagged"].sum()))
print("Hub manager escalations:", int(
    (new_actions_df["action"] == "Escalate to hub manager").sum()
))
print("Missing values:", int(new_actions_df.isna().sum().sum()))
print("Unique shipment IDs:", new_actions_df["shipment_id"].nunique())
