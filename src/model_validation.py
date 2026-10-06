import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)


# ==========================================
# STEP 1: LOAD CLEAN DATA
# ==========================================

tickets = pd.read_csv("data/clean_tickets.csv")

refunds = tickets[
    tickets["refund_amount_inr"].notna()
    & (tickets["refund_amount_inr"] > 0)
].copy()

# Exclude GW-OTHER because it is the population
# we ultimately want the model to classify.
known = refunds[
    refunds["refund_reason_code"].notna()
    & (refunds["refund_reason_code"] != "GW-OTHER")
].copy()


# ==========================================
# STEP 2: PREPARE INDIVIDUAL FEATURES
# ==========================================

known["customer_message"] = (
    known["customer_message"]
    .fillna("")
    .astype(str)
)

known["agent_notes"] = (
    known["agent_notes"]
    .fillna("")
    .astype(str)
)

known["category"] = (
    known["category"]
    .fillna("")
    .astype(str)
)


# ==========================================
# STEP 3: CREATE THREE FEATURE CONFIGURATIONS
# ==========================================

known["message_only"] = (
    known["customer_message"]
)

known["message_notes"] = (
    known["customer_message"]
    + " "
    + known["agent_notes"]
)

known["message_notes_category"] = (
    known["customer_message"]
    + " "
    + known["agent_notes"]
    + " "
    + known["category"]
)


# ==========================================
# STEP 4: TRAIN / TEST SPLIT
# ==========================================

y = known["refund_reason_code"]

indices = known.index

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# STEP 5: MODEL CONFIGURATION
# ==========================================

def create_model():

    return Pipeline([
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
# STEP 6: EVALUATE EACH FEATURE SET
# ==========================================

feature_sets = {
    "Customer Message Only": "message_only",
    "Message + Agent Notes": "message_notes",
    "Message + Notes + Category": "message_notes_category"
}

results = []


print("=" * 70)
print("MODEL FEATURE VALIDATION")
print("=" * 70)

print(f"\nKnown labelled refunds: {len(known):,}")
print(f"Training samples: {len(train_idx):,}")
print(f"Testing samples: {len(test_idx):,}")


for model_name, feature_column in feature_sets.items():

    print("\n" + "-" * 70)
    print(model_name)
    print("-" * 70)

    X_train = known.loc[
        train_idx,
        feature_column
    ]

    X_test = known.loc[
        test_idx,
        feature_column
    ]

    y_train = known.loc[
        train_idx,
        "refund_reason_code"
    ]

    y_test = known.loc[
        test_idx,
        "refund_reason_code"
    ]

    model = create_model()

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        )
    )

    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )
    )

    print(
        f"Accuracy:          {accuracy:.4f}"
    )

    print(
        f"Macro Precision:   {precision:.4f}"
    )

    print(
        f"Macro Recall:      {recall:.4f}"
    )

    print(
        f"Macro F1:          {f1:.4f}"
    )

    print(
        f"Weighted F1:       {weighted_f1:.4f}"
    )

    results.append({
        "feature_set": model_name,
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "weighted_f1": weighted_f1
    })


# ==========================================
# STEP 7: RESULTS TABLE
# ==========================================

results_df = pd.DataFrame(
    results
).sort_values(
    "macro_f1",
    ascending=False
)

print("\n" + "=" * 70)
print("FEATURE COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# ==========================================
# STEP 8: SELECT BEST MODEL
# ==========================================

best_model_row = results_df.iloc[0]

best_feature_set = (
    best_model_row["feature_set"]
)

best_f1 = (
    best_model_row["macro_f1"]
)

best_accuracy = (
    best_model_row["accuracy"]
)

print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print(
    f"\nSelected feature set: "
    f"{best_feature_set}"
)

print(
    f"Accuracy: {best_accuracy:.4f}"
)

print(
    f"Macro F1: {best_f1:.4f}"
)


# ==========================================
# STEP 9: SAVE RESULTS
# ==========================================

results_df.to_csv(
    "data/model_feature_comparison.csv",
    index=False
)

print("\nSaved:")
print("data/model_feature_comparison.csv")


# ==========================================
# FINAL
# ==========================================

print("\n" + "=" * 70)
print("MODEL VALIDATION COMPLETE")
print("=" * 70)