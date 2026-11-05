
import os
import pandas as pd
from src.shipment_prep import clean_shipments, build_features, FEATURES

os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

raw_ship_df = pd.read_csv("data/raw_shipments.csv")
hubs_df = pd.read_csv("data/hubs.csv")

raw_rows = len(raw_ship_df)
duplicate_rows = int(raw_ship_df.duplicated().sum())

missing_distance_before = int(raw_ship_df["distance_km"].isna().sum())
missing_weight_before = int(raw_ship_df["parcel_weight_kg"].isna().sum())
sentinel_weight_rows = int((raw_ship_df["parcel_weight_kg"] > 100).sum())
missing_value_before = int(raw_ship_df["declared_value"].isna().sum())
missing_payment_before = int(raw_ship_df["payment_mode"].isna().sum())

service_spellings_before = int(raw_ship_df["service_type"].nunique())

clean_df, medians = clean_shipments(raw_ship_df)

service_spellings_after = int(clean_df["service_type"].nunique())

feature_df = build_features(clean_df, hubs_df)

rows_after_hub_merge = len(feature_df)

feature_df["is_delayed"] = (
    feature_df["delivery_hours"] > feature_df["promised_hours"]
).astype(int)

feature_df["distance_band"] = pd.cut(
    feature_df["distance_km"],
    bins=[0, 200, 400, 600, 2000],
    labels=["0-200", "201-400", "401-600", "601+"]
)

modelling_df = feature_df.copy()

modelling_df.to_csv(
    "data/modelling_table.csv",
    index=False
)

numeric_columns = modelling_df.select_dtypes(include="number").columns

nulls_in_features_and_target = int(
    modelling_df[list(FEATURES) + ["is_delayed"]].isna().sum().sum()
)

delayed_count = int(modelling_df["is_delayed"].sum())
delay_rate = float(modelling_df["is_delayed"].mean())

q1 = modelling_df["delivery_hours"].quantile(0.25)
q3 = modelling_df["delivery_hours"].quantile(0.75)
iqr = q3 - q1
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr

outlier_count = int(
    (
        (modelling_df["delivery_hours"] < lower_fence)
        | (modelling_df["delivery_hours"] > upper_fence)
    ).sum()
)

declared_mean = modelling_df["declared_value_inr"].mean()
declared_median = modelling_df["declared_value_inr"].median()
log_mean = modelling_df["log_declared_value"].mean()
log_median = modelling_df["log_declared_value"].median()

check_metrics = {
    "raw_rows": raw_rows,
    "duplicate_rows_removed": duplicate_rows,
    "rows_after_dedup": len(clean_df),
    "missing_distance_km_before": missing_distance_before,
    "missing_parcel_weight_before": missing_weight_before,
    "sentinel_weight_rows": sentinel_weight_rows,
    "missing_declared_value_before": missing_value_before,
    "missing_payment_mode_before": missing_payment_before,
    "service_type_spellings_before": service_spellings_before,
    "service_type_spellings_after": service_spellings_after,
    "distance_km_median": medians["distance_km"],
    "parcel_weight_kg_median": medians["parcel_weight_kg"],
    "declared_value_inr_median": medians["declared_value_inr"],
    "rows_after_hub_merge": rows_after_hub_merge,
    "nulls_in_features_and_target": nulls_in_features_and_target,
    "delayed_count": delayed_count,
    "delay_rate": delay_rate,
    "delivery_hours_q1": q1,
    "delivery_hours_q3": q3,
    "delivery_hours_iqr": iqr,
    "delivery_hours_lower_fence": lower_fence,
    "delivery_hours_upper_fence": upper_fence,
    "delivery_hours_outlier_count": outlier_count,
    "declared_value_mean": declared_mean,
    "declared_value_median": declared_median,
    "log_declared_value_mean": log_mean,
    "log_declared_value_median": log_median
}

checks = pd.DataFrame(
    list(check_metrics.items()),
    columns=["metric", "value"]
)

checks["value"] = checks["value"].round(4)

checks.to_csv(
    "outputs/part_a_checks.csv",
    index=False
)

group_results = []

group_columns = {
    "service_type": "service_type",
    "hub_tier": "hub_tier",
    "distance_band": "distance_band"
}

for dimension, column in group_columns.items():
    grouped = modelling_df.groupby(
        column,
        observed=True
    ).agg(
        shipments=("shipment_id", "count"),
        delayed=("is_delayed", "sum"),
        delay_rate=("is_delayed", "mean"),
        mean_hours=("delivery_hours", "mean")
    ).reset_index()

    grouped = grouped.rename(columns={column: "group"})
    grouped.insert(0, "dimension", dimension)
    group_results.append(grouped)

delay_by_group = pd.concat(
    group_results,
    ignore_index=True
)

delay_by_group["delay_rate"] = delay_by_group["delay_rate"].round(4)
delay_by_group["mean_hours"] = delay_by_group["mean_hours"].round(4)

delay_by_group.to_csv(
    "outputs/part_a_delay_by_group.csv",
    index=False
)

correlations = modelling_df.select_dtypes(
    include="number"
).corr().round(4)

correlations.to_csv(
    "outputs/part_a_correlations.csv",
    index=True
)

print("Part A completed successfully.")
print("Modelling table:", modelling_df.shape)
print("Checks:", checks.shape)
print("Delay groups:", delay_by_group.shape)
print("Correlation matrix:", correlations.shape)
