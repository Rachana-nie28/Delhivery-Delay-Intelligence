# Delhivery Delay Intelligence

## How to run

### Prerequisites
- Python 3.10 or later
- pip
- Required Python libraries listed in `requirements.txt`

### Installation

```bash
pip install -r requirements.txt
```

### Execution order

Run the following scripts from the project root directory:

```bash
python generate_data.py
python part_a_clean_eda.py
python part_b_regression.py
python part_c_classification.py
python part_d_segments.py
python package_model.py
python predict_delay.py
```

The scripts generate the cleaned modelling dataset, analytical outputs, trained model, and shipment-level delay predictions.

The final trained model is saved as `models/delay_model.pkl`. The prediction script uses the saved model, feature order, imputation values, and alert threshold to score new shipments.

## Data findings

The raw shipment dataset contained 1,224 records. After removing 24 duplicate records, 1,200 unique shipment records remained for analysis.

Data cleaning addressed missing distance, parcel weight, declared value, and payment mode values. Invalid parcel-weight sentinel values were also handled, and inconsistent service-type spellings were standardized. Hub information was merged into the shipment dataset, resulting in no missing values across the modelling features and target.

The final dataset contained 278 delayed shipments, representing a delay rate of approximately 23.17%.

### Key observations

- **Service type:** Express shipments had a delay rate of 48.97%, compared with 20.56% for Standard and 8.57% for Economy shipments.
- **Hub tier:** Tier 3 hubs recorded a 44% delay rate, followed by Tier 2 at 21.5% and Tier 1 at 4%.
- **Distance:** Shipments travelling more than 600 km had an 85.23% delay rate, while shipments travelling up to 200 km had a 2.19% delay rate.
- **Operational factors:** Promised delivery hours, distance, and number of stops showed the strongest positive correlations with delivery time.

These findings indicate that shipment distance, hub capacity, service type, and operational workload are important factors associated with delivery delays.

## Model results (regression and classification)

### Regression

Four regression models were evaluated alongside a mean-prediction baseline.

| Model | Validation MAE | Validation RMSE | Validation R² |
|---|---:|---:|---:|
| Mean baseline | 15.7952 | 18.8552 | -0.0212 |
| Linear Regression | 3.9222 | 4.8912 | 0.9313 |
| Random Forest | 4.2882 | 5.5095 | 0.9128 |
| Gradient Boosting | 3.9166 | 4.9165 | 0.9306 |
| Decision Tree | 5.0971 | 6.2796 | 0.8867 |

Linear Regression was selected as the final regression model because it achieved the strongest overall validation performance.

On the test dataset, Linear Regression achieved:

- MAE: 3.5671 hours
- RMSE: 4.5236 hours
- R²: 0.9493

This indicates that the regression model can estimate delivery time with relatively low prediction error on the held-out test data.

### Classification

The classification task predicts whether a shipment will be delayed.

The dataset was split using stratified sampling:

| Dataset | Records | Delayed shipments |
|---|---:|---:|
| Training | 840 | 195 |
| Validation | 180 | 42 |
| Test | 180 | 41 |

The blind baseline predicted every shipment as on time. Although it achieved 76.67% accuracy, its recall and F1-score were both zero. This demonstrates why accuracy alone is insufficient for an imbalanced classification problem.

The following classifiers were evaluated on the validation dataset:

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.9056 | 0.7778 | 0.8333 | 0.8046 |
| Decision Tree (depth 6) | 0.8833 | 0.7561 | 0.7381 | 0.7470 |
| Random Forest | 0.9056 | 0.8378 | 0.7381 | 0.7848 |

### Feature selection and noise analysis

The seven highest-ranked features according to the ANOVA F-score were:

1. distance_km
2. hub_tier
3. hub_daily_capacity
4. n_stops
5. promised_hours
6. rain_flag
7. sale_event_flag

All five designated noise columns ranked outside the top seven and had F-scores below 10.

Their combined Random Forest feature importance was 22.44%. This indicates that the noise features contributed to the model, but the strongest statistical signals were concentrated in the operational features.

The feature-selection comparison showed that seven features provided the strongest validation F1-score among the tested feature counts using the classifier's default predictions.

### Hyperparameter tuning

Grid Search evaluated 27 parameter combinations across five cross-validation folds, resulting in 135 model fits. Randomized Search evaluated nine combinations across five folds, resulting in 45 model fits.

| Search method | Candidates | Total fits | Best CV accuracy | Validation F1 |
|---|---:|---:|---:|---:|
| Grid Search | 27 | 135 | 0.8952 | 0.7750 |
| Randomized Search | 9 | 45 | 0.8905 | 0.7848 |
| Default Forest with 7 features | 1 | 1 | — | 0.8140 |

The default Random Forest with seven selected features achieved the highest validation F1-score. Therefore, it was selected instead of the tuned alternatives.

The validation set contained only 180 records and 42 delayed shipments. Consequently, small changes in predictions can noticeably affect precision, recall, and F1-score. Cross-validation accuracy and validation F1-score were considered separately when comparing the models.

### Alert threshold selection

The final model uses a probability threshold of 0.5. A shipment is flagged as delayed when its predicted delay probability is greater than or equal to this threshold.

The threshold was selected as the highest tested threshold that maintained validation recall of at least 0.80.

At a threshold of 0.5, validation precision was 0.7955 and recall was 0.8333.

A lower threshold increases the number of delayed shipments identified but also increases false alerts. A higher threshold reduces false alerts but can miss more genuinely delayed shipments.

### Final test evaluation

The final model was evaluated once on the untouched test dataset using the selected threshold.

| Metric | Test result |
|---|---:|
| Threshold | 0.5 |
| Accuracy | 0.8667 |
| Precision | 0.6889 |
| Recall | 0.7561 |
| F1-score | 0.7209 |
| True Positives | 31 |
| True Negatives | 125 |
| False Positives | 14 |
| False Negatives | 10 |

The model identified 31 of the 41 delayed shipments in the test dataset. Its test recall was lower than its validation recall, showing that validation performance does not guarantee identical performance on unseen data.

### Final classification configuration

- Feature selector: SelectKBest
- Selected features: 7
- Classifier: Random Forest
- Number of estimators: 100
- Maximum depth: Unlimited
- Minimum samples per leaf: 1
- Alert threshold: 0.5

## Operations recommendation

The analysis indicates that shipment delays are influenced by distance, hub capacity, service type, and the number of delivery stops.

Operational teams should prioritize the following actions:

1. **Prioritize high-risk shipments:** Use the classification model to flag shipments with a predicted delay probability of at least 0.5.
2. **Monitor high-risk hubs:** Tier 3 hubs recorded a 44% delay rate. These hubs should receive closer monitoring of workload, capacity, and dispatch operations.
3. **Review long-distance shipments:** Shipments travelling more than 600 km recorded an 85.23% delay rate. Earlier dispatch planning and route monitoring may help reduce delays.
4. **Improve hub-level operations:** Compare hub performance against the segment profiles to identify operational mismatches and investigate possible capacity or workload issues.
5. **Use human review:** Model-generated alerts should support hub managers rather than automatically determine operational decisions.

### Hub segmentation

K-Means clustering was evaluated using different cluster counts. Three clusters were selected as a practical balance between cluster separation and manageable segment sizes.

The resulting segments were:

| Segment | Hubs | Average delivery hours | Delay rate |
|---|---:|---:|---:|
| Standard | 7 | 47.24 | 2.57% |
| Watch | 8 | 56.81 | 15.50% |
| Escalate | 9 | 64.65 | 46.00% |

Four hubs showed differences between their expected tier-based category and actual performance-based segment. These mismatches should be investigated rather than assuming that hub tier alone determines operational performance.

## Production plan

The proposed production solution can be developed into a batch-based shipment monitoring system with the following characteristics:

| Production characteristic | Proposed implementation |
|---|---|
| Data ingestion | Import shipment and hub records from operational databases or APIs. |
| Data validation | Check missing values, duplicate shipment IDs, invalid measurements, and unexpected feature values before scoring. |
| Reproducible preprocessing | Use the shipment preparation logic in `src/shipment_prep.py`. |
| Model deployment | Load the serialized model from `models/delay_model.pkl`. |
| Alert generation | Apply the saved threshold from `models/delay_threshold.csv` to generate delay-risk alerts. |
| Operational review | Provide flagged shipments to hub managers for investigation and prioritization. |
| Model monitoring | Track prediction quality, input data quality, and changes in shipment patterns. |

### Monitoring metrics and triggers

The following are proposed initial monitoring thresholds for a production deployment. They should be reviewed and calibrated using real operational data.

- **Missing data rate:** Investigate when more than 1% of incoming shipment records contain missing required values.
- **Delay recall:** Review model performance and initiate retraining assessment if measured recall falls below 0.70 over a sufficiently large labelled evaluation window.
- **Feature drift:** Investigate when the Population Stability Index (PSI) for an important model feature exceeds 0.20.
- **Alert volume:** Review operational capacity if the proportion of flagged shipments changes substantially from the established baseline.

### Retraining strategy

The model should initially be reviewed quarterly and retrained when sufficient new labelled shipment data becomes available. Retraining should also be considered after major operational changes, such as new hubs, route changes, capacity changes, or significant shifts in shipment patterns.

Sale-event periods should be compared with equivalent sale-event periods rather than ordinary weeks because promotional demand can create different shipment patterns.

Data drift occurs when the distribution of incoming shipment features changes over time. Concept drift occurs when the relationship between shipment features and actual delays changes. Both should be monitored because they can reduce model reliability.

## Limitations

1. The dataset contains 1,200 cleaned shipment records, which may not represent the full range of real-world delivery conditions.
2. The classification validation and test sets contain relatively few delayed shipments, so individual prediction changes can affect performance metrics.
3. The model relies on the available shipment and hub features. Unexpected disruptions such as vehicle breakdowns, labour shortages, or severe weather events may not be fully represented.
4. Feature importance and statistical ranking indicate associations with the target but do not establish causal relationships.
5. Hub segmentation is based on the available operational measurements and may change as shipment volumes and hub conditions evolve.
6. The alert threshold and monitoring triggers require further validation using real production data before operational deployment.

The project provides a reproducible analytical workflow for shipment delay prediction, model evaluation, hub segmentation, and operational alert generation. Real-world deployment would require integration with live logistics systems, continuous monitoring, and periodic model validation.
