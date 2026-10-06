import pandas as pd
import re


# ==========================================
# STEP 1: LOAD CLEAN DATA
# ==========================================

tickets = pd.read_csv("data/clean_tickets.csv")

gw_other = tickets[
    tickets["refund_reason_code"] == "GW-OTHER"
].copy()

print("=" * 70)
print("GW-OTHER INVESTIGATION")
print("=" * 70)

print(f"\nGW-OTHER tickets: {len(gw_other):,}")

print(
    f"GW-OTHER refund value: "
    f"₹{gw_other['refund_amount_inr'].sum():,.2f}"
)


# ==========================================
# STEP 2: BASIC BREAKDOWN
# ==========================================

print("\n" + "=" * 70)
print("GW-OTHER BY CATEGORY")
print("=" * 70)

category_summary = (
    gw_other
    .groupby("category", dropna=False)
    .agg(
        tickets=("ticket_id", "count"),
        total_refund_inr=("refund_amount_inr", "sum"),
        average_refund_inr=("refund_amount_inr", "mean")
    )
    .sort_values(
        "total_refund_inr",
        ascending=False
    )
)

print(category_summary.to_string())


# ==========================================
# STEP 3: BY ASSIGNED TEAM
# ==========================================

print("\n" + "=" * 70)
print("GW-OTHER BY ASSIGNED TEAM")
print("=" * 70)

team_summary = (
    gw_other
    .groupby("assigned_team", dropna=False)
    .agg(
        tickets=("ticket_id", "count"),
        total_refund_inr=("refund_amount_inr", "sum"),
        average_refund_inr=("refund_amount_inr", "mean")
    )
    .sort_values(
        "total_refund_inr",
        ascending=False
    )
)

print(team_summary.to_string())


# ==========================================
# STEP 4: BY AGENT
# ==========================================

print("\n" + "=" * 70)
print("GW-OTHER BY AGENT")
print("=" * 70)

agent_summary = (
    gw_other
    .groupby("agent_id", dropna=False)
    .agg(
        tickets=("ticket_id", "count"),
        total_refund_inr=("refund_amount_inr", "sum"),
        average_refund_inr=("refund_amount_inr", "mean")
    )
    .sort_values(
        "total_refund_inr",
        ascending=False
    )
)

print(agent_summary.to_string())


# ==========================================
# STEP 5: REPLACEMENT BREAKDOWN
# ==========================================

print("\n" + "=" * 70)
print("GW-OTHER REPLACEMENT BREAKDOWN")
print("=" * 70)

replacement_summary = (
    gw_other
    .groupby("replacement_issued", dropna=False)
    .agg(
        tickets=("ticket_id", "count"),
        total_refund_inr=("refund_amount_inr", "sum")
    )
)

print(replacement_summary.to_string())


# ==========================================
# STEP 6: COMBINE CUSTOMER MESSAGE + NOTES
# ==========================================

gw_other["text"] = (
    gw_other["customer_message"].fillna("")
    + " "
    + gw_other["agent_notes"].fillna("")
)

gw_other["text"] = (
    gw_other["text"]
    .astype(str)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)


# ==========================================
# STEP 7: KEYWORD INVESTIGATION
# ==========================================

keyword_patterns = {
    "quality_issue": [
        "defect",
        "defective",
        "broken",
        "not working",
        "damaged",
        "faulty",
        "dead",
        "stopped working"
    ],

    "delivery_issue": [
        "delivery",
        "delayed",
        "late",
        "shipping",
        "courier",
        "transit",
        "package"
    ],

    "duplicate_payment": [
        "charged twice",
        "double charged",
        "duplicate payment",
        "paid twice",
        "twice"
    ],

    "cancellation": [
        "cancel",
        "cancellation"
    ],

    "price_issue": [
        "price",
        "discount",
        "cheaper",
        "refund difference",
        "price difference"
    ],

    "return": [
        "return",
        "send back",
        "sent back"
    ],

    "warranty": [
        "warranty",
        "warranty claim",
        "repair"
    ],

    "customer_dissatisfaction": [
        "unhappy",
        "dissatisfied",
        "disappointed",
        "angry",
        "complaint",
        "complain",
        "poor experience"
    ],

    "goodwill": [
        "goodwill",
        "gesture",
        "apology",
        "compensation",
        "compensate",
        "credit"
    ]
}


for label, patterns in keyword_patterns.items():

    pattern = "|".join(
        re.escape(p)
        for p in patterns
    )

    gw_other[label] = (
        gw_other["text"]
        .str.lower()
        .str.contains(
            pattern,
            regex=True,
            na=False
        )
    )


# ==========================================
# STEP 8: KEYWORD SUMMARY
# ==========================================

print("\n" + "=" * 70)
print("GW-OTHER KEYWORD SIGNALS")
print("=" * 70)

keyword_results = []

for label in keyword_patterns:

    matched = gw_other[label].sum()

    refund_value = gw_other.loc[
        gw_other[label],
        "refund_amount_inr"
    ].sum()

    keyword_results.append({
        "signal": label,
        "tickets": matched,
        "refund_value_inr": refund_value
    })

keyword_summary = pd.DataFrame(
    keyword_results
).sort_values(
    "refund_value_inr",
    ascending=False
)

print(keyword_summary.to_string(index=False))


# ==========================================
# STEP 9: SHOW SAMPLE TICKETS
# ==========================================

print("\n" + "=" * 70)
print("GW-OTHER SAMPLE TICKETS")
print("=" * 70)

sample_columns = [
    "ticket_id",
    "refund_amount_inr",
    "category",
    "assigned_team",
    "agent_id",
    "customer_message",
    "agent_notes"
]

print(
    gw_other[
        sample_columns
    ]
    .head(30)
    .to_string(index=False)
)


# ==========================================
# STEP 10: SAVE INVESTIGATION DATA
# ==========================================

gw_other.to_csv(
    "data/gw_other_investigation.csv",
    index=False
)

category_summary.to_csv(
    "data/gw_other_by_category.csv"
)

team_summary.to_csv(
    "data/gw_other_by_team.csv"
)

agent_summary.to_csv(
    "data/gw_other_by_agent.csv"
)

keyword_summary.to_csv(
    "data/gw_other_keyword_signals.csv",
    index=False
)


# ==========================================
# FINAL
# ==========================================

print("\n" + "=" * 70)
print("GW-OTHER INVESTIGATION COMPLETE")
print("=" * 70)

print("\nFiles created:")
print("1. data/gw_other_investigation.csv")
print("2. data/gw_other_by_category.csv")
print("3. data/gw_other_by_team.csv")
print("4. data/gw_other_by_agent.csv")
print("5. data/gw_other_keyword_signals.csv")