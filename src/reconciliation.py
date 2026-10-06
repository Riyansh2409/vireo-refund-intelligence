import pandas as pd


# ==========================================
# STEP 1: LOAD CLEAN DATA
# ==========================================

clean_tickets = pd.read_csv("data/clean_tickets.csv")

print("=" * 60)
print("REFUND RECONCILIATION")
print("=" * 60)

print(f"\nClean tickets: {len(clean_tickets):,}")
print(
    f"Unique ticket IDs: "
    f"{clean_tickets['ticket_id'].nunique():,}"
)
print(
    f"Duplicate ticket IDs: "
    f"{clean_tickets['ticket_id'].duplicated().sum():,}"
)


# ==========================================
# STEP 2: SELECT VALID REFUNDS
# ==========================================

refunds = clean_tickets[
    clean_tickets["refund_amount_inr"].notna()
    & (clean_tickets["refund_amount_inr"] > 0)
].copy()

print("\n" + "=" * 60)
print("REFUND BASE")
print("=" * 60)

print(f"\nRefund tickets: {len(refunds):,}")

total_refund = refunds["refund_amount_inr"].sum()

print(f"Total refund value: ₹{total_refund:,.2f}")


# ==========================================
# STEP 3: BASIC DATA QUALITY CHECKS
# ==========================================

print("\n" + "=" * 60)
print("DATA QUALITY CHECKS")
print("=" * 60)

missing_reason = refunds["refund_reason_code"].isna().sum()
missing_agent = refunds["agent_id"].isna().sum()

print(f"\nRefunds missing reason code: {missing_reason}")
print(f"Refunds missing agent ID: {missing_agent}")


# ==========================================
# STEP 4: MONTHLY RECONCILIATION
# ==========================================

refunds["created_at"] = pd.to_datetime(
    refunds["created_at"],
    errors="coerce"
)

refunds["month"] = (
    refunds["created_at"]
    .dt.to_period("M")
    .astype(str)
)

monthly = (
    refunds
    .groupby("month")
    .agg(
        refund_tickets=("ticket_id", "count"),
        total_refund_inr=("refund_amount_inr", "sum")
    )
    .sort_index()
)

monthly_total = monthly["total_refund_inr"].sum()

monthly_check = abs(
    monthly_total - total_refund
) < 0.01

print("\n" + "=" * 60)
print("MONTHLY RECONCILIATION")
print("=" * 60)

print(monthly.to_string())

print(
    f"\nMonthly total: ₹{monthly_total:,.2f}"
)

print(
    f"Monthly reconciliation: "
    f"{'PASS' if monthly_check else 'FAIL'}"
)


# ==========================================
# STEP 5: REASON CODE RECONCILIATION
# ==========================================

reason = (
    refunds
    .groupby("refund_reason_code")
    .agg(
        refund_tickets=("ticket_id", "count"),
        total_refund_inr=("refund_amount_inr", "sum")
    )
    .sort_values(
        "total_refund_inr",
        ascending=False
    )
)

reason_total = reason["total_refund_inr"].sum()

reason_check = abs(
    reason_total - total_refund
) < 0.01

print("\n" + "=" * 60)
print("REASON CODE RECONCILIATION")
print("=" * 60)

print(reason.to_string())

print(
    f"\nReason-code total: ₹{reason_total:,.2f}"
)

print(
    f"Reason-code reconciliation: "
    f"{'PASS' if reason_check else 'FAIL'}"
)


# ==========================================
# STEP 6: AGENT RECONCILIATION
# ==========================================

agent = (
    refunds
    .groupby("agent_id")
    .agg(
        refund_tickets=("ticket_id", "count"),
        total_refund_inr=("refund_amount_inr", "sum")
    )
    .sort_values(
        "total_refund_inr",
        ascending=False
    )
)

agent_total = agent["total_refund_inr"].sum()

agent_check = abs(
    agent_total - total_refund
) < 0.01

print("\n" + "=" * 60)
print("AGENT RECONCILIATION")
print("=" * 60)

print(agent.to_string())

print(
    f"\nAgent total: ₹{agent_total:,.2f}"
)

print(
    f"Agent reconciliation: "
    f"{'PASS' if agent_check else 'FAIL'}"
)


# ==========================================
# STEP 7: GRAND RECONCILIATION
# ==========================================

print("\n" + "=" * 60)
print("GRAND RECONCILIATION")
print("=" * 60)

print(f"\nMaster refund total: ₹{total_refund:,.2f}")
print(f"Monthly total:       ₹{monthly_total:,.2f}")
print(f"Reason total:        ₹{reason_total:,.2f}")
print(f"Agent total:         ₹{agent_total:,.2f}")

grand_check = (
    monthly_check
    and reason_check
    and agent_check
    and missing_reason == 0
    and missing_agent == 0
)

print("\nOverall reconciliation:")

if grand_check:
    print("PASS")
    print(
        "All refund views reconcile to the master refund total."
    )
else:
    print("FAIL")
    print(
        "One or more reconciliation checks failed."
    )


# ==========================================
# STEP 8: SAVE RECONCILIATION OUTPUTS
# ==========================================

monthly.to_csv(
    "data/reconciliation_monthly.csv"
)

reason.to_csv(
    "data/reconciliation_reason.csv"
)

agent.to_csv(
    "data/reconciliation_agent.csv"
)

print("\n" + "=" * 60)
print("OUTPUT FILES")
print("=" * 60)

print("\nCreated:")
print("1. data/reconciliation_monthly.csv")
print("2. data/reconciliation_reason.csv")
print("3. data/reconciliation_agent.csv")