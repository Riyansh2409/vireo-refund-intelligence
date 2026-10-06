import pandas as pd


# ==========================================
# STEP 1: LOAD DATA
# ==========================================

tickets = pd.read_csv(
    "data/clean_tickets.csv"
)

ai_predictions = pd.read_csv(
    "data/gw_other_ai_predictions.csv"
)

agents = pd.read_csv(
    "data/agents.csv"
)


print("=" * 70)
print("REFUND OPPORTUNITY ANALYSIS")
print("=" * 70)


# ==========================================
# STEP 2: BASIC REFUND METRICS
# ==========================================

refunds = tickets[
    tickets["refund_amount_inr"].notna()
    & (tickets["refund_amount_inr"] > 0)
].copy()

total_refund = (
    refunds["refund_amount_inr"].sum()
)

gw_other = refunds[
    refunds["refund_reason_code"] == "GW-OTHER"
].copy()

gw_other_tickets = len(gw_other)

gw_other_value = (
    gw_other["refund_amount_inr"].sum()
)

print("\n" + "=" * 70)
print("OVERALL REFUND POOL")
print("=" * 70)

print(
    f"\nTotal refund tickets: "
    f"{len(refunds):,}"
)

print(
    f"Total refund value: "
    f"₹{total_refund:,.2f}"
)

print(
    f"\nGW-OTHER tickets: "
    f"{gw_other_tickets:,}"
)

print(
    f"GW-OTHER refund value: "
    f"₹{gw_other_value:,.2f}"
)


gw_other_value_share = (
    gw_other_value / total_refund * 100
)

gw_other_ticket_share = (
    gw_other_tickets / len(refunds) * 100
)

print(
    f"\nGW-OTHER share of refund value: "
    f"{gw_other_value_share:.2f}%"
)

print(
    f"GW-OTHER share of refund tickets: "
    f"{gw_other_ticket_share:.2f}%"
)


# ==========================================
# STEP 3: AI CONFIDENCE ANALYSIS
# ==========================================

print("\n" + "=" * 70)
print("AI CLASSIFICATION OPPORTUNITY")
print("=" * 70)

ai_predictions["confidence"] = pd.to_numeric(
    ai_predictions["confidence"],
    errors="coerce"
)

high_confidence = ai_predictions[
    ai_predictions["confidence"] >= 0.80
].copy()

medium_confidence = ai_predictions[
    (ai_predictions["confidence"] >= 0.60)
    & (ai_predictions["confidence"] < 0.80)
].copy()

low_confidence = ai_predictions[
    ai_predictions["confidence"] < 0.60
].copy()


# ==========================================
# STEP 4: CONFIDENCE METRICS
# ==========================================

high_tickets = len(high_confidence)
medium_tickets = len(medium_confidence)
low_tickets = len(low_confidence)

high_value = (
    high_confidence["refund_amount_inr"].sum()
)

medium_value = (
    medium_confidence["refund_amount_inr"].sum()
)

low_value = (
    low_confidence["refund_amount_inr"].sum()
)

print(
    f"\nHigh confidence (>= 80%): "
    f"{high_tickets:,} tickets | "
    f"₹{high_value:,.2f}"
)

print(
    f"Medium confidence (60-79.9%): "
    f"{medium_tickets:,} tickets | "
    f"₹{medium_value:,.2f}"
)

print(
    f"Low confidence (< 60%): "
    f"{low_tickets:,} tickets | "
    f"₹{low_value:,.2f}"
)


# ==========================================
# STEP 5: HIGH-CONFIDENCE OPPORTUNITY %
# ==========================================

high_ticket_share = (
    high_tickets / gw_other_tickets * 100
)

high_value_share = (
    high_value / gw_other_value * 100
)

print(
    f"\nHigh-confidence ticket coverage: "
    f"{high_ticket_share:.2f}%"
)

print(
    f"High-confidence value coverage: "
    f"{high_value_share:.2f}%"
)


# ==========================================
# STEP 6: PREDICTED REASON BREAKDOWN
# ==========================================

print("\n" + "=" * 70)
print("HIGH-CONFIDENCE AI OPPORTUNITY BY REASON")
print("=" * 70)

high_reason = (
    high_confidence
    .groupby("predicted_reason")
    .agg(
        tickets=("ticket_id", "count"),
        refund_value_inr=("refund_amount_inr", "sum"),
        average_refund_inr=("refund_amount_inr", "mean"),
        average_confidence=("confidence", "mean")
    )
    .sort_values(
        "refund_value_inr",
        ascending=False
    )
)

print(
    high_reason.to_string()
)


# ==========================================
# STEP 7: TEAM-LEVEL OPPORTUNITY
# ==========================================

high_confidence_team = (
    high_confidence
    .groupby("assigned_team")
    .agg(
        tickets=("ticket_id", "count"),
        refund_value_inr=("refund_amount_inr", "sum"),
        average_refund_inr=("refund_amount_inr", "mean"),
        average_confidence=("confidence", "mean")
    )
    .sort_values(
        "refund_value_inr",
        ascending=False
    )
)

print("\n" + "=" * 70)
print("HIGH-CONFIDENCE OPPORTUNITY BY TEAM")
print("=" * 70)

print(
    high_confidence_team.to_string()
)


# ==========================================
# STEP 8: AGENT-LEVEL OPPORTUNITY
# ==========================================

high_confidence_agent = (
    high_confidence
    .groupby("agent_id")
    .agg(
        tickets=("ticket_id", "count"),
        refund_value_inr=("refund_amount_inr", "sum"),
        average_refund_inr=("refund_amount_inr", "mean"),
        average_confidence=("confidence", "mean")
    )
    .sort_values(
        "refund_value_inr",
        ascending=False
    )
)

print("\n" + "=" * 70)
print("HIGH-CONFIDENCE OPPORTUNITY BY AGENT")
print("=" * 70)

print(
    high_confidence_agent.head(20).to_string()
)


# ==========================================
# STEP 9: JOIN AGENT DETAILS
# ==========================================

# agents.csv can contain multiple assignment rows
# for the same agent, so keep one row per agent_id
# for this reporting view.

agent_lookup = (
    agents[
        [
            "agent_id",
            "name",
            "team",
            "tier"
        ]
    ]
    .drop_duplicates(
        subset="agent_id"
    )
)

high_confidence_agent_detail = (
    high_confidence_agent
    .reset_index()
    .merge(
        agent_lookup,
        on="agent_id",
        how="left"
    )
    .sort_values(
        "refund_value_inr",
        ascending=False
    )
)

print("\n" + "=" * 70)
print("HIGH-CONFIDENCE AGENT DETAILS")
print("=" * 70)

print(
    high_confidence_agent_detail
    .head(20)
    .to_string(index=False)
)


# ==========================================
# STEP 10: REPLACEMENT CONTROL
# ==========================================

print("\n" + "=" * 70)
print("HIGH-CONFIDENCE REFUND + REPLACEMENT")
print("=" * 70)

replacement_values = (
    high_confidence[
        high_confidence["replacement_issued"]
        .astype(str)
        .str.upper()
        .eq("Y")
    ]
)

replacement_tickets = len(
    replacement_values
)

replacement_value = (
    replacement_values["refund_amount_inr"].sum()
)

print(
    f"\nHigh-confidence refund + replacement tickets: "
    f"{replacement_tickets:,}"
)

print(
    f"Refund value involved: "
    f"₹{replacement_value:,.2f}"
)


# ==========================================
# STEP 11: ESTIMATED MANUAL REVIEW QUEUE
# ==========================================

manual_review = ai_predictions[
    ai_predictions["confidence"] < 0.80
].copy()

manual_review_tickets = len(
    manual_review
)

manual_review_value = (
    manual_review["refund_amount_inr"].sum()
)

print("\n" + "=" * 70)
print("MANUAL REVIEW QUEUE")
print("=" * 70)

print(
    f"\nTickets requiring manual review: "
    f"{manual_review_tickets:,}"
)

print(
    f"Refund value requiring manual review: "
    f"₹{manual_review_value:,.2f}"
)


# ==========================================
# STEP 12: BUSINESS OPPORTUNITY SUMMARY
# ==========================================

business_summary = pd.DataFrame([
    {
        "metric": "Total refund value",
        "value": total_refund
    },
    {
        "metric": "GW-OTHER refund value",
        "value": gw_other_value
    },
    {
        "metric": "GW-OTHER ticket count",
        "value": gw_other_tickets
    },
    {
        "metric": "High-confidence AI tickets",
        "value": high_tickets
    },
    {
        "metric": "High-confidence AI refund value",
        "value": high_value
    },
    {
        "metric": "High-confidence value coverage %",
        "value": high_value_share
    },
    {
        "metric": "Manual review tickets",
        "value": manual_review_tickets
    },
    {
        "metric": "Manual review refund value",
        "value": manual_review_value
    }
])


# ==========================================
# STEP 13: SAVE OUTPUTS
# ==========================================

high_reason.to_csv(
    "data/high_confidence_by_reason.csv"
)

high_confidence_team.to_csv(
    "data/high_confidence_by_team.csv"
)

high_confidence_agent_detail.to_csv(
    "data/high_confidence_by_agent.csv",
    index=False
)

manual_review.to_csv(
    "data/manual_review_queue.csv",
    index=False
)

business_summary.to_csv(
    "data/business_opportunity_summary.csv",
    index=False
)


# ==========================================
# STEP 14: FINAL BUSINESS STATEMENT
# ==========================================

print("\n" + "=" * 70)
print("BUSINESS OPPORTUNITY")
print("=" * 70)

print(
    f"""
GW-OTHER currently represents
₹{gw_other_value:,.2f} of refund value.

The AI classifier identifies
₹{high_value:,.2f}
across {high_tickets:,} tickets
with confidence >= 80%.

This represents
{high_value_share:.2f}%
of GW-OTHER refund value.

Tickets below the 80% confidence threshold
remain in a manual-review queue.
"""
)


# ==========================================
# FINAL
# ==========================================

print("=" * 70)
print("OPPORTUNITY ANALYSIS COMPLETE")
print("=" * 70)

print("\nFiles created:")
print("1. data/high_confidence_by_reason.csv")
print("2. data/high_confidence_by_team.csv")
print("3. data/high_confidence_by_agent.csv")
print("4. data/manual_review_queue.csv")
print("5. data/business_opportunity_summary.csv")