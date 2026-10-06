import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# STEP 1: LOAD CLEAN DATA
# ==========================================

tickets = pd.read_csv("data/clean_tickets.csv")

print("=" * 70)
print("AI REFUND REASON CLASSIFIER")
print("=" * 70)

print(f"\nTotal clean tickets: {len(tickets):,}")


# ==========================================
# STEP 2: SELECT REFUND TICKETS
# ==========================================

refunds = tickets[
    tickets["refund_amount_inr"].notna()
    & (tickets["refund_amount_inr"] > 0)
].copy()

print(f"Total refund tickets: {len(refunds):,}")


# ==========================================
# STEP 3: CREATE TEXT FEATURE
# ==========================================

refunds["customer_message"] = (
    refunds["customer_message"]
    .fillna("")
    .astype(str)
)

refunds["agent_notes"] = (
    refunds["agent_notes"]
    .fillna("")
    .astype(str)
)

refunds["category"] = (
    refunds["category"]
    .fillna("")
    .astype(str)
)

# Combine the available textual context.
refunds["text"] = (
    refunds["customer_message"]
    + " "
    + refunds["agent_notes"]
    + " "
    + refunds["category"]
)


# ==========================================
# STEP 4: SEPARATE KNOWN LABELS
# ==========================================

known = refunds[
    refunds["refund_reason_code"].notna()
    & (refunds["refund_reason_code"] != "GW-OTHER")
].copy()

gw_other = refunds[
    refunds["refund_reason_code"] == "GW-OTHER"
].copy()

print(f"\nKnown labelled refunds: {len(known):,}")
print(f"GW-OTHER tickets for classification: {len(gw_other):,}")


# ==========================================
# STEP 5: SHOW LABEL DISTRIBUTION
# ==========================================

print("\n" + "=" * 70)
print("TRAINING LABEL DISTRIBUTION")
print("=" * 70)

print(
    known["refund_reason_code"]
    .value_counts()
    .to_string()
)


# ==========================================
# STEP 6: TRAIN / TEST SPLIT
# ==========================================

X = known["text"]
y = known["refund_reason_code"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

print(f"\nTraining samples: {len(X_train):,}")
print(f"Testing samples:  {len(X_test):,}")


# ==========================================
# STEP 7: BUILD ML PIPELINE
# ==========================================

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        )
    )
])


# ==========================================
# STEP 8: TRAIN MODEL
# ==========================================

print("\n" + "=" * 70)
print("TRAINING MODEL")
print("=" * 70)

model.fit(
    X_train,
    y_train
)

print("\nModel training complete.")


# ==========================================
# STEP 9: EVALUATE MODEL
# ==========================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

print(
    f"\nAccuracy: {accuracy:.4f}"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ==========================================
# STEP 10: CONFUSION MATRIX
# ==========================================

labels = sorted(
    y_test.unique()
)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

confusion_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(confusion_df)


# ==========================================
# STEP 11: PREDICT GW-OTHER
# ==========================================

print("\n" + "=" * 70)
print("CLASSIFYING GW-OTHER")
print("=" * 70)

gw_other["predicted_reason"] = model.predict(
    gw_other["text"]
)

probabilities = model.predict_proba(
    gw_other["text"]
)

gw_other["confidence"] = probabilities.max(
    axis=1
)


# ==========================================
# STEP 12: CONFIDENCE BUCKETS
# ==========================================

def confidence_bucket(score):

    if score >= 0.80:
        return "High"

    elif score >= 0.60:
        return "Medium"

    return "Low"


gw_other["confidence_bucket"] = (
    gw_other["confidence"]
    .apply(confidence_bucket)
)


# ==========================================
# STEP 13: PREDICTED REASON SUMMARY
# ==========================================

predicted_summary = (
    gw_other
    .groupby("predicted_reason")
    .agg(
        tickets=("ticket_id", "count"),
        total_refund_inr=("refund_amount_inr", "sum"),
        average_refund_inr=("refund_amount_inr", "mean"),
        average_confidence=("confidence", "mean")
    )
    .sort_values(
        "total_refund_inr",
        ascending=False
    )
)

print("\n" + "=" * 70)
print("PREDICTED GW-OTHER REASONS")
print("=" * 70)

print(
    predicted_summary.to_string()
)


# ==========================================
# STEP 14: CONFIDENCE SUMMARY
# ==========================================

confidence_summary = (
    gw_other
    .groupby("confidence_bucket")
    .agg(
        tickets=("ticket_id", "count"),
        total_refund_inr=("refund_amount_inr", "sum"),
        average_confidence=("confidence", "mean")
    )
)

print("\n" + "=" * 70)
print("CONFIDENCE SUMMARY")
print("=" * 70)

print(
    confidence_summary.to_string()
)


# ==========================================
# STEP 15: HIGH-CONFIDENCE OPPORTUNITY
# ==========================================

high_confidence = gw_other[
    gw_other["confidence"] >= 0.80
].copy()

high_confidence_value = (
    high_confidence["refund_amount_inr"].sum()
)

high_confidence_tickets = len(
    high_confidence
)

print("\n" + "=" * 70)
print("HIGH-CONFIDENCE CLASSIFICATIONS")
print("=" * 70)

print(
    f"\nHigh-confidence tickets: "
    f"{high_confidence_tickets:,}"
)

print(
    f"High-confidence refund value: "
    f"₹{high_confidence_value:,.2f}"
)


# ==========================================
# STEP 16: SHOW SAMPLE PREDICTIONS
# ==========================================

print("\n" + "=" * 70)
print("SAMPLE GW-OTHER PREDICTIONS")
print("=" * 70)

sample_columns = [
    "ticket_id",
    "refund_amount_inr",
    "category",
    "predicted_reason",
    "confidence",
    "confidence_bucket",
    "customer_message",
    "agent_notes"
]

print(
    gw_other[
        sample_columns
    ]
    .sort_values(
        "confidence",
        ascending=False
    )
    .head(30)
    .to_string(index=False)
)


# ==========================================
# STEP 17: SAVE RESULTS
# ==========================================

gw_other.to_csv(
    "data/gw_other_ai_predictions.csv",
    index=False
)

predicted_summary.to_csv(
    "data/gw_other_ai_summary.csv"
)

confidence_summary.to_csv(
    "data/gw_other_ai_confidence.csv"
)

confusion_df.to_csv(
    "data/ai_confusion_matrix.csv"
)


# ==========================================
# FINAL
# ==========================================

print("\n" + "=" * 70)
print("AI CLASSIFICATION COMPLETE")
print("=" * 70)

print("\nFiles created:")
print("1. data/gw_other_ai_predictions.csv")
print("2. data/gw_other_ai_summary.csv")
print("3. data/gw_other_ai_confidence.csv")
print("4. data/ai_confusion_matrix.csv")