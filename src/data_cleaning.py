import pandas as pd

# ==========================================
# STEP 1: LOAD DATA
# ==========================================

tickets = pd.read_csv("data/tickets.csv")
agents = pd.read_csv("data/agents.csv")
customers = pd.read_csv("data/customers.csv")
orders = pd.read_csv("data/orders.csv")
products = pd.read_csv("data/products.csv")

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)

print("\nTICKETS")
print(tickets.shape)
print(tickets.columns.tolist())

print("\nAGENTS")
print(agents.shape)

print("\nCUSTOMERS")
print(customers.shape)

print("\nORDERS")
print(orders.shape)

print("\nPRODUCTS")
print(products.shape)


# ==========================================
# STEP 2: RAW DATA AUDIT
# ==========================================

print("\n" + "=" * 60)
print("RAW TICKET AUDIT")
print("=" * 60)

print("\nDuplicate ticket IDs:")
print(tickets["ticket_id"].duplicated().sum())

print("\nUnique ticket IDs:")
print(tickets["ticket_id"].nunique())

print("\nSource system:")
print(tickets["source_system"].value_counts())

print("\nRefund statistics by source system:")

print(
    tickets.groupby("source_system")["refund_amount_inr"]
    .agg(["count", "sum", "mean"])
)


# ==========================================
# STEP 3: IDENTIFY DUPLICATE TICKETS
# ==========================================

duplicates = tickets[
    tickets["ticket_id"].duplicated(keep=False)
].sort_values("ticket_id")

print("\n" + "=" * 60)
print("DUPLICATE TICKET AUDIT")
print("=" * 60)

print("\nDuplicate ticket examples:")

print(
    duplicates[
        [
            "ticket_id",
            "source_system",
            "refund_amount_inr",
            "refund_reason_code",
            "replacement_issued"
        ]
    ].head(30).to_string(index=False)
)

print("\nDuplicate ticket IDs by source system:")

print(
    duplicates.groupby("ticket_id")["source_system"]
    .agg(lambda x: ", ".join(sorted(x.unique())))
    .value_counts()
)


# ==========================================
# STEP 4: VALIDATE LEGACY MONEY SCALE
# ==========================================

pivot = duplicates.pivot_table(
    index="ticket_id",
    columns="source_system",
    values="refund_amount_inr",
    aggfunc="first"
)

refund_pairs = pivot.dropna(
    subset=["helpdesk", "legacy_fd"]
).copy()

refund_pairs["ratio"] = (
    refund_pairs["legacy_fd"] /
    refund_pairs["helpdesk"]
)

print("\n" + "=" * 60)
print("LEGACY MONEY SCALE VALIDATION")
print("=" * 60)

print("\nLegacy / Helpdesk refund ratios:")
print(refund_pairs["ratio"].value_counts())

print("\nRatio statistics:")
print(refund_pairs["ratio"].describe())

# Check whether every paired refund has exactly 100x difference
all_100x = (refund_pairs["ratio"] == 100).all()

print("\nIs legacy amount consistently 100x helpdesk amount?")
print(all_100x)


# ==========================================
# STEP 5: NORMALIZE LEGACY REFUND AMOUNTS
# ==========================================

clean_tickets = tickets.copy()

legacy_mask = clean_tickets["source_system"].eq("legacy_fd")

clean_tickets.loc[
    legacy_mask & clean_tickets["refund_amount_inr"].notna(),
    "refund_amount_inr"
] = (
    clean_tickets.loc[
        legacy_mask & clean_tickets["refund_amount_inr"].notna(),
        "refund_amount_inr"
    ] / 100
)

print("\n" + "=" * 60)
print("LEGACY AMOUNT NORMALIZATION")
print("=" * 60)

print("Legacy refund amounts divided by 100.")


# ==========================================
# STEP 6: PRIORITIZE CURRENT HELPDESK RECORD
# ==========================================

# Current helpdesk is treated as authoritative
# when the same ticket exists in both systems.

clean_tickets["source_priority"] = clean_tickets["source_system"].map({
    "helpdesk": 0,
    "legacy_fd": 1
})

clean_tickets = (
    clean_tickets
    .sort_values(
        ["ticket_id", "source_priority"]
    )
    .drop_duplicates(
        subset="ticket_id",
        keep="first"
    )
    .drop(columns=["source_priority"])
)


# ==========================================
# STEP 7: VALIDATE CLEAN DATASET
# ==========================================

print("\n" + "=" * 60)
print("CLEAN DATASET VALIDATION")
print("=" * 60)

print("\nRaw tickets:")
print(len(tickets))

print("\nClean tickets:")
print(len(clean_tickets))

print("\nUnique ticket IDs:")
print(clean_tickets["ticket_id"].nunique())

print("\nRemaining duplicate ticket IDs:")
print(clean_tickets["ticket_id"].duplicated().sum())


# ==========================================
# STEP 8: CLEAN REFUND ANALYSIS
# ==========================================

refunds_clean = clean_tickets[
    clean_tickets["refund_amount_inr"].notna()
    & (clean_tickets["refund_amount_inr"] > 0)
].copy()

print("\n" + "=" * 60)
print("CLEAN REFUND SUMMARY")
print("=" * 60)

print("\nRefund tickets:")
print(len(refunds_clean))

print("\nTotal refunds:")
print(
    f"₹{refunds_clean['refund_amount_inr'].sum():,.2f}"
)

print("\nAverage refund:")
print(
    f"₹{refunds_clean['refund_amount_inr'].mean():,.2f}"
)

print("\nMaximum refund:")
print(
    f"₹{refunds_clean['refund_amount_inr'].max():,.2f}"
)


# ==========================================
# STEP 9: REFUNDS BY SOURCE AFTER CLEANING
# ==========================================

print("\n" + "=" * 60)
print("REFUNDS BY SOURCE AFTER CLEANING")
print("=" * 60)

print(
    clean_tickets.groupby("source_system")["refund_amount_inr"]
    .agg(["count", "sum", "mean"])
)


# ==========================================
# STEP 10: REFUNDS BY REASON CODE
# ==========================================

print("\n" + "=" * 60)
print("REFUNDS BY REASON CODE")
print("=" * 60)

reason_summary = (
    refunds_clean
    .groupby("refund_reason_code", dropna=False)
    .agg(
        refund_tickets=("ticket_id", "count"),
        total_refund_inr=("refund_amount_inr", "sum"),
        average_refund_inr=("refund_amount_inr", "mean")
    )
    .sort_values(
        "total_refund_inr",
        ascending=False
    )
)

print(reason_summary)


# ==========================================
# STEP 11: REFUNDS BY AGENT
# ==========================================

print("\n" + "=" * 60)
print("REFUNDS BY AGENT")
print("=" * 60)

agent_summary = (
    refunds_clean
    .groupby("agent_id", dropna=False)
    .agg(
        refund_tickets=("ticket_id", "count"),
        total_refund_inr=("refund_amount_inr", "sum"),
        average_refund_inr=("refund_amount_inr", "mean")
    )
    .sort_values(
        "total_refund_inr",
        ascending=False
    )
)

print(agent_summary.head(20))


# ==========================================
# STEP 12: MONTHLY REFUND SUMMARY
# ==========================================

clean_tickets["created_at"] = pd.to_datetime(
    clean_tickets["created_at"],
    errors="coerce"
)

refunds_clean = clean_tickets[
    clean_tickets["refund_amount_inr"].notna()
    & (clean_tickets["refund_amount_inr"] > 0)
].copy()

refunds_clean["month"] = (
    refunds_clean["created_at"]
    .dt.to_period("M")
    .astype(str)
)

monthly_summary = (
    refunds_clean
    .groupby("month")
    .agg(
        refund_tickets=("ticket_id", "count"),
        total_refund_inr=("refund_amount_inr", "sum"),
        average_refund_inr=("refund_amount_inr", "mean")
    )
    .sort_index()
)

print("\n" + "=" * 60)
print("MONTHLY REFUND SUMMARY")
print("=" * 60)

print(monthly_summary)


# ==========================================
# STEP 13: REFUND + REPLACEMENT EXCEPTIONS
# ==========================================

refund_replacement = clean_tickets[
    (
        clean_tickets["refund_amount_inr"].notna()
        & (clean_tickets["refund_amount_inr"] > 0)
    )
    & (
        clean_tickets["replacement_issued"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    )
].copy()

print("\n" + "=" * 60)
print("REFUND + REPLACEMENT EXCEPTIONS")
print("=" * 60)

print("\nNumber of refund + replacement tickets:")
print(len(refund_replacement))

print("\nRefund value involved:")
print(
    f"₹{refund_replacement['refund_amount_inr'].sum():,.2f}"
)


# ==========================================
# STEP 14: SAVE CLEAN DATA
# ==========================================

clean_tickets.to_csv(
    "data/clean_tickets.csv",
    index=False
)

reason_summary.to_csv(
    "data/refund_by_reason.csv"
)

agent_summary.to_csv(
    "data/refund_by_agent.csv"
)

monthly_summary.to_csv(
    "data/refund_monthly.csv"
)


# ==========================================
# FINAL MESSAGE
# ==========================================

print("\n" + "=" * 60)
print("CLEANING COMPLETE")
print("=" * 60)

print("\nFiles created:")
print("1. data/clean_tickets.csv")
print("2. data/refund_by_reason.csv")
print("3. data/refund_by_agent.csv")
print("4. data/refund_monthly.csv")