import pandas as pd

print("=" * 70)
print("REFUND + REPLACEMENT POLICY CHECK")
print("=" * 70)

df = pd.read_csv("data/clean_tickets.csv")

# Normalize fields
df["refund_amount_inr"] = pd.to_numeric(
    df["refund_amount_inr"], errors="coerce"
).fillna(0)

df["replacement_flag"] = (
    df["replacement_issued"]
    .astype(str)
    .str.strip()
    .str.upper()
)

# Policy: customer must not receive both refund and replacement
exceptions = df[
    (df["refund_amount_inr"] > 0) &
    (df["replacement_flag"] == "Y")
].copy()

print(f"\nRefund tickets: {(df['refund_amount_inr'] > 0).sum():,}")
print(f"Refund + replacement tickets: {len(exceptions):,}")
print(
    f"Refund value involved: "
    f"₹{exceptions['refund_amount_inr'].sum():,.2f}"
)

print("\n" + "=" * 70)
print("EXCEPTIONS BY REASON")
print("=" * 70)

print(
    exceptions.groupby("refund_reason_code")
    .agg(
        tickets=("ticket_id", "count"),
        refund_value_inr=("refund_amount_inr", "sum")
    )
    .sort_values("refund_value_inr", ascending=False)
)

print("\n" + "=" * 70)
print("EXCEPTIONS BY TEAM")
print("=" * 70)

print(
    exceptions.groupby("assigned_team")
    .agg(
        tickets=("ticket_id", "count"),
        refund_value_inr=("refund_amount_inr", "sum")
    )
    .sort_values("refund_value_inr", ascending=False)
)

print("\n" + "=" * 70)
print("EXCEPTION DETAILS")
print("=" * 70)

cols = [
    "ticket_id",
    "order_id",
    "agent_id",
    "assigned_team",
    "refund_amount_inr",
    "refund_reason_code",
    "replacement_issued",
    "customer_message",
    "agent_notes"
]

print(exceptions[cols].to_string(index=False))

exceptions.to_csv(
    "data/refund_replacement_exceptions.csv",
    index=False
)

print("\nSaved:")
print("data/refund_replacement_exceptions.csv")