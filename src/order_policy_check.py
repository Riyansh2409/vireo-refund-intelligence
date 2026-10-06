import pandas as pd

INPUT = "data/clean_tickets.csv"
OUTPUT = "data/order_policy_exceptions.csv"

df = pd.read_csv(INPUT)

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

# Keep tickets with a usable order_id
with_order = df[
    df["order_id"].notna()
    & df["order_id"].astype(str).str.strip().ne("")
].copy()

# Ticket-level exception:
# same ticket has refund + replacement
with_order["same_ticket_exception"] = (
    (with_order["refund_amount_inr"] > 0)
    & (with_order["replacement_flag"] == "Y")
)

# Group at order level
order_flags = (
    with_order
    .groupby("order_id")
    .agg(
        refund_tickets=(
            "refund_amount_inr",
            lambda s: (s > 0).sum()
        ),
        refund_value_inr=(
            "refund_amount_inr",
            "sum"
        ),
        replacement_tickets=(
            "replacement_flag",
            lambda s: (s == "Y").sum()
        ),
        same_ticket_exception=(
            "same_ticket_exception",
            "any"
        ),
        ticket_count=(
            "ticket_id",
            "nunique"
        )
    )
    .reset_index()
)

# Order-level exception
exceptions = order_flags[
    (order_flags["refund_tickets"] > 0)
    & (order_flags["replacement_tickets"] > 0)
].copy()

exceptions["cross_ticket_exception"] = (
    exceptions["same_ticket_exception"] == False
)

# Save
exceptions.to_csv(OUTPUT, index=False)

print("=" * 60)
print("ORDER-LEVEL POLICY CHECK")
print("=" * 60)

print(f"Orders with usable order_id: {len(order_flags):,}")
print(f"Orders with refund + replacement: {len(exceptions):,}")

print(
    f"Refund value involved: "
    f"₹{exceptions['refund_value_inr'].sum():,.2f}"
)

print(
    f"Same-ticket exceptions: "
    f"{exceptions['same_ticket_exception'].sum():,}"
)

print(
    f"Cross-ticket exceptions: "
    f"{exceptions['cross_ticket_exception'].sum():,}"
)

print("\nTop exceptions:")
print(
    exceptions
    .sort_values("refund_value_inr", ascending=False)
    .head(20)
    .to_string(index=False)
)

print(f"\nSaved to: {OUTPUT}")
