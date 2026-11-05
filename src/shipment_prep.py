import numpy as np
import pandas as pd
PROMISED_HOURS = {
    "Express": 36,
    "Standard": 60,
    "Economy": 96
}
FEATURES = [
    "promised_hours",
    "distance_km",
    "parcel_weight_kg",
    "log_declared_value",
    "n_stops",
    "rain_flag",
    "sale_event_flag",
    "is_cod",
    "is_weekend",
    "dispatch_hour",
    "hub_tier",
    "hub_daily_capacity",
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no"
]
NOISE_COLUMNS = [
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no"
]
def clean_shipments(raw, medians=None):
    clean = raw.drop_duplicates().copy()
    clean["service_type"] = (
        clean["service_type"]
        .astype("string")
        .str.strip()
        .str.title()
    )
    clean["payment_mode"] = clean["payment_mode"].fillna("Unknown")
    clean["declared_value_inr"] = pd.to_numeric(
        clean["declared_value"]
        .astype("string")
        .str.replace("₹", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip(),
        errors="coerce"
    )
    clean = clean.drop(columns=["declared_value"])
    clean["dispatch_time"] = pd.to_datetime(
        clean["dispatch_time"],
        errors="coerce"
    )
    clean.loc[
        clean["parcel_weight_kg"] > 100,
        "parcel_weight_kg"
    ] = np.nan
    median_columns = [
        "distance_km",
        "parcel_weight_kg",
        "declared_value_inr"
    ]
    if medians is None:
        medians = {
            column: float(clean[column].median())
            for column in median_columns
        }
    else:
        medians = {
            column: float(medians[column])
            for column in median_columns
        }
    for column in median_columns:
        clean[column] = clean[column].astype("float64")
        clean[column] = clean[column].fillna(medians[column])
    return clean, medians
def build_features(clean, hubs):
    features = clean.merge(
        hubs[
            [
                "hub_id",
                "hub_city",
                "region",
                "hub_tier",
                "hub_daily_capacity"
            ]
        ],
        on="hub_id",
        how="left",
        validate="many_to_one"
    )
    features["promised_hours"] = features["service_type"].map(
        PROMISED_HOURS
    )
    features["is_cod"] = (
        features["payment_mode"]
        .astype("string")
        .str.strip()
        .str.upper()
        .eq("COD")
        .astype(int)
    )
    features["is_weekend"] = (
        features["dispatch_time"].dt.dayofweek >= 5
    ).astype(int)
    features["dispatch_hour"] = (
        features["dispatch_time"].dt.hour
    )
    features["log_declared_value"] = np.log1p(
        features["declared_value_inr"]
    )
    return features
