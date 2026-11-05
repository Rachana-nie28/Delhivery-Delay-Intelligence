
import os
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("data/modelling_table.csv")

hub_profile = df.groupby("hub_id").agg(
    shipments=("shipment_id", "count"),
    mean_delivery_hours=("delivery_hours", "mean"),
    delay_rate=("is_delayed", "mean"),
    mean_distance_km=("distance_km", "mean"),
    mean_n_stops=("n_stops", "mean"),
    hub_tier=("hub_tier", "first"),
    hub_city=("hub_city", "first")
).reset_index()

hub_profile = hub_profile.sort_values(
    "hub_id"
).reset_index(drop=True)

profile_columns = [
    "hub_id",
    "shipments",
    "mean_delivery_hours",
    "delay_rate",
    "mean_distance_km",
    "mean_n_stops"
]

hub_profile[profile_columns].to_csv(
    "outputs/hub_profile.csv",
    index=False
)

profile_features = [
    "mean_delivery_hours",
    "delay_rate",
    "mean_distance_km",
    "mean_n_stops"
]

scaler = StandardScaler()

scaled_profiles = scaler.fit_transform(
    hub_profile[profile_features]
)

separation_results = []

for k in [2, 3, 4, 5]:
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(scaled_profiles)

    within_distances = []
    across_distances = []

    for i, j in combinations(range(len(labels)), 2):
        distance = np.linalg.norm(
            scaled_profiles[i] - scaled_profiles[j]
        )

        if labels[i] == labels[j]:
            within_distances.append(distance)
        else:
            across_distances.append(distance)

    cluster_sizes = sorted(
        pd.Series(labels).value_counts().tolist(),
        reverse=True
    )

    separation_results.append({
        "K": k,
        "across_over_within": (
            np.mean(across_distances) /
            np.mean(within_distances)
        ),
        "cluster_sizes": "-".join(
            str(size) for size in cluster_sizes
        )
    })

separation_df = pd.DataFrame(separation_results)

separation_df["across_over_within"] = (
    separation_df["across_over_within"].round(4)
)

separation_df.to_csv(
    "outputs/part_d_separation.csv",
    index=False
)

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

hub_profile["cluster"] = kmeans.fit_predict(
    scaled_profiles
)

cluster_order = (
    hub_profile.groupby("cluster")["delay_rate"]
    .mean()
    .sort_values()
    .index.tolist()
)

segment_names = {
    cluster_order[0]: "Standard",
    cluster_order[1]: "Watch",
    cluster_order[2]: "Escalate"
}

hub_profile["segment"] = hub_profile["cluster"].map(
    segment_names
)

hub_segments = hub_profile[
    ["hub_id", "segment", "hub_city", "hub_tier"]
].copy()

hub_segments.to_csv(
    "outputs/hub_segments.csv",
    index=False
)

segment_profile = hub_profile.groupby("segment").agg(
    hubs=("hub_id", "count"),
    mean_delivery_hours=("mean_delivery_hours", "mean"),
    delay_rate=("delay_rate", "mean"),
    mean_distance_km=("mean_distance_km", "mean"),
    mean_n_stops=("mean_n_stops", "mean")
).reset_index()

segment_order = ["Standard", "Watch", "Escalate"]

segment_profile["segment"] = pd.Categorical(
    segment_profile["segment"],
    categories=segment_order,
    ordered=True
)

segment_profile = segment_profile.sort_values(
    "segment"
).reset_index(drop=True)

segment_profile["segment"] = segment_profile["segment"].astype(str)

segment_profile[
    [
        "mean_delivery_hours",
        "delay_rate",
        "mean_distance_km",
        "mean_n_stops"
    ]
] = segment_profile[
    [
        "mean_delivery_hours",
        "delay_rate",
        "mean_distance_km",
        "mean_n_stops"
    ]
].round(4)

segment_profile.to_csv(
    "outputs/segment_profile.csv",
    index=False
)

tier_segment_map = {
    1: "Standard",
    2: "Watch",
    3: "Escalate"
}

hub_profile["expected_segment"] = (
    hub_profile["hub_tier"].map(tier_segment_map)
)

mismatch_df = hub_profile[
    hub_profile["segment"] != hub_profile["expected_segment"]
][
    [
        "hub_id",
        "hub_city",
        "hub_tier",
        "expected_segment",
        "segment"
    ]
].copy()

mismatch_df = mismatch_df.rename(
    columns={"segment": "actual_segment"}
)

mismatch_df.to_csv(
    "outputs/tier_vs_segment_mismatch.csv",
    index=False
)

print("Part D segmentation completed successfully.")
print("Hub profile:", hub_profile.shape)
print("Separation:", separation_df.shape)
print("Hub segments:", hub_segments.shape)
print("Segment profile:", segment_profile.shape)
print("Tier mismatches:", mismatch_df.shape)
