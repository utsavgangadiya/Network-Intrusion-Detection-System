# Network Intrusion Detection System

## Problem Statement

The Network Intrusion Detection System uses an XGBoost classifier and a Streamlit application to classify network-flow records as normal traffic or attack categories.

A 40-row test dataset contains:

| Actual class | Records |
|---|---:|
| Normal Traffic | 37 |
| Port Scanning | 2 |
| DoS | 1 |

However, the Streamlit dashboard originally displayed:

| Displayed prediction | Records |
|---|---:|
| Bots | 37 |
| Brute Force | 2 |
| Port Scanning | 1 |
| Normal Traffic | 0 |
| Attack Rate | 100% |

The application was not changing the model predictions. It was decoding the numeric predictions with an incorrect label mapping.

## Verified Root Cause

The saved XGBoost model and saved label encoder did not describe the same class system.

### Model artifact

The saved model contains seven numeric classes:

```text
[0, 1, 2, 3, 4, 5, 6]
```

It also contains 52 feature names, and the Streamlit application selects uploaded columns in the model's feature order.

### Incorrect encoder artifact

The original saved encoder contained only numeric classes:

```text
[0, 1, 2, 3, 4, 5]
```

This encoder could not safely decode a seven-class model prediction.

### Authoritative mapping from the notebook

The notebook recorded the actual `LabelEncoder.classes_` values:

```text
0 -> Bots
1 -> Brute Force
2 -> DDoS
3 -> DoS
4 -> Normal Traffic
5 -> Port Scanning
6 -> Web Attacks
```

The model predictions observed on the test file were numeric classes 3, 4, and 5. Therefore the correct decoding is:

```text
3 -> DoS
4 -> Normal Traffic
5 -> Port Scanning
```

The old Streamlit fallback used this unrelated list as a numeric index:

```python
KNOWN_ATTACK_CLASSES = [
    "Normal Traffic", "DoS", "DDoS", "Port Scanning",
    "Bots", "Brute Force", "Web Attacks",
]
```

That fallback decoded numeric class 4 as `Bots` and numeric class 5 as `Brute Force`. This caused the false 100% attack rate.

## Training Pipeline Issue

The notebook also contains an unsafe training order:

```python
le = LabelEncoder()
df["Attack Type"] = le.fit_transform(df["Attack Type"])
joblib.dump(le, "models/label_encoder.pkl")
```

The notebook then filters rare classes after encoding:

```python
counts = df["Attack Type"].value_counts()
valid_classes = counts[counts >= 2].index
df = df[df["Attack Type"].isin(valid_classes)]
```

This makes it possible for the final training dataframe, model, and saved encoder to represent different class sets. The encoder must be fitted after all label cleaning and filtering is complete.

## Solution Applied

### 1. Repaired the saved encoder

The encoder was rebuilt using the exact class order recorded in `nids.ipynb`:

```python
AUTHORITATIVE_CLASSES = [
    "Bots",
    "Brute Force",
    "DDoS",
    "DoS",
    "Normal Traffic",
    "Port Scanning",
    "Web Attacks",
]

encoder = LabelEncoder()
encoder.fit(AUTHORITATIVE_CLASSES)
joblib.dump(encoder, "models/label_encoder.pkl")
```

This repair is not a cosmetic relabeling. The mapping was recovered from the original training notebook.

### 2. Removed unsafe Streamlit remapping

The application now decodes predictions only through the saved encoder:

```python
def safe_inverse_transform(predictions, encoder):
    predictions = np.asarray(predictions)
    if predictions.size == 0:
        return predictions

    decoded = encoder.inverse_transform(predictions)
    return np.asarray([normalize_prediction_label(label) for label in decoded])
```

The application no longer silently converts a numeric prediction into an item from `KNOWN_ATTACK_CLASSES`.

### 3. Added artifact validation

The application validates the model and encoder before showing the dashboard:

```python
if model_count != encoder_count:
    raise RuntimeError(
        f"Class-count mismatch: model={model_count}, encoder={encoder_count}"
    )
```

This makes deployment fail clearly instead of displaying incorrect attack names.

### 4. Added a corrected training script

`train_model.py` demonstrates the correct order:

1. Load the complete labeled training dataset.
2. Clean the target labels.
3. Remove unsupported or rare classes.
4. Fit `LabelEncoder` once on the final target labels.
5. Train XGBoost using the encoded labels.
6. Validate model and encoder class counts.
7. Save both artifacts from the same run.

Run it with a full labeled training CSV:

```powershell
py -3.13 train_model.py path\to\full_training_dataset.csv
```

Do not train on the 40-row test file. That file is a validation sample, not the original training dataset.

## Do We Need to Retrain?

For the immediate label-display problem, retraining is not required because the authoritative mapping was recovered from the notebook and the encoder was repaired.

Retraining is recommended before a production deployment because the notebook's original filtering and encoding order was unsafe. The corrected `train_model.py` ensures that the new model and encoder are produced together.

## Feature Order and Preprocessing Check

The saved model exposes 52 feature names. The Streamlit application uses:

```python
feature_names = [str(column) for column in model.feature_names_in_]
input_df = df[feature_names].copy()
```

This preserves the model's training feature order and prevents column-order errors during prediction.

The 40-row file should contain the same 52 model features. Its `Attack Type` column should be used only as ground truth for evaluation, not as an input feature and not to manufacture predictions.

## Expected Result After the Fix

The corrected decoding should report:

```text
DoS: 1
Normal Traffic: 37
Port Scanning: 2
```

The corresponding prediction attack rate should be:

```text
3 / 40 = 7.5%
```

The application should no longer report 37 Bots, 2 Brute Force, or a 100% attack rate for this test file.

## Changed Files

- `app.py`: strict artifact validation and encoder-only prediction decoding.
- `models/label_encoder.pkl`: repaired using the authoritative notebook class order.
- `repair_label_encoder.py`: reproducible encoder-repair utility.
- `train_model.py`: corrected model-training pipeline for future retraining.
