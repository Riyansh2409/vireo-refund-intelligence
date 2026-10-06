import streamlit as st
import pandas as pd
from pathlib import Path

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Vireo Audio Refund Intelligence",
    page_icon="🎧",
    layout="wide"
)

DATA_DIR = Path("data")


# ============================================================
# HELPERS
# ============================================================

def load_csv(filename):
    path = DATA_DIR / filename

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


def money(value):
    return f"₹{value:,.0f}"


# ============================================================
# LOAD DATA
# ============================================================

clean = load_csv("clean_tickets.csv")

monthly = load_csv("refund_monthly.csv")
reason = load_csv("refund_by_reason.csv")
agent = load_csv("refund_by_agent.csv")

gw_category = load_csv("gw_other_by_category.csv")
gw_team = load_csv("gw_other_by_team.csv")
gw_agent = load_csv("gw_other_by_agent.csv")
gw_keywords = load_csv("gw_other_keyword_signals.csv")

ai_predictions = load_csv("gw_other_ai_predictions.csv")
ai_summary = load_csv("gw_other_ai_summary.csv")
ai_confidence = load_csv("gw_other_ai_confidence.csv")

feature_comparison = load_csv("model_feature_comparison.csv")

high_reason = load_csv("high_confidence_by_reason.csv")
high_team = load_csv("high_confidence_by_team.csv")
high_agent = load_csv("high_confidence_by_agent.csv")

manual_queue = load_csv("manual_review_queue.csv")

policy_exceptions = load_csv("order_policy_exceptions.csv")


# ============================================================
# PREPARE CLEAN DATA
# ============================================================

if not clean.empty:

    clean["refund_amount_inr"] = pd.to_numeric(
        clean["refund_amount_inr"],
        errors="coerce"
    ).fillna(0)

    refund_df = clean[
        clean["refund_amount_inr"] > 0
    ].copy()

else:

    refund_df = pd.DataFrame()


# ============================================================
# EXECUTIVE KPIs
# ============================================================

total_refunds = refund_df["refund_amount_inr"].sum()

refund_tickets = len(refund_df)

gw_df = refund_df[
    refund_df["refund_reason_code"]
    .astype(str)
    .str.upper()
    .eq("GW-OTHER")
]

gw_value = gw_df["refund_amount_inr"].sum()

gw_tickets = len(gw_df)

gw_share = (
    gw_value / total_refunds * 100
    if total_refunds > 0
    else 0
)


# ============================================================
# AI HIGH CONFIDENCE
# ============================================================

high_conf_tickets = 0
high_conf_value = 0

if not ai_predictions.empty:

    ai_predictions["confidence"] = pd.to_numeric(
        ai_predictions["confidence"],
        errors="coerce"
    )

    ai_predictions["refund_amount_inr"] = pd.to_numeric(
        ai_predictions["refund_amount_inr"],
        errors="coerce"
    ).fillna(0)

    high_conf = ai_predictions[
        ai_predictions["confidence"] >= 0.80
    ].copy()

    high_conf_tickets = len(high_conf)

    high_conf_value = high_conf[
        "refund_amount_inr"
    ].sum()


# ============================================================
# POLICY KPIs
# ============================================================

policy_order_count = len(policy_exceptions)

policy_value = 0

same_ticket_count = 0
cross_ticket_count = 0

if not policy_exceptions.empty:

    policy_exceptions["refund_value_inr"] = pd.to_numeric(
        policy_exceptions["refund_value_inr"],
        errors="coerce"
    ).fillna(0)

    policy_value = policy_exceptions[
        "refund_value_inr"
    ].sum()

    if "same_ticket_exception" in policy_exceptions.columns:
        same_ticket_count = int(
            policy_exceptions["same_ticket_exception"].sum()
        )

    if "cross_ticket_exception" in policy_exceptions.columns:
        cross_ticket_count = int(
            policy_exceptions["cross_ticket_exception"].sum()
        )


# ============================================================
# HEADER
# ============================================================

st.title("🎧 Vireo Audio Refund Intelligence")

st.markdown(
    """
    **AI-assisted refund reconciliation, classification and policy analysis**

    This tool reconciles legacy and current helpdesk data, investigates
    `GW-OTHER` refunds, evaluates AI classification confidence, and identifies
    refund/replacement policy exceptions.
    """
)

st.divider()


# ============================================================
# EXECUTIVE SUMMARY
# ============================================================

st.subheader("Executive Summary")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Total Refunds",
        money(total_refunds)
    )

with col2:
    st.metric(
        "Refund Tickets",
        f"{refund_tickets:,}"
    )

with col3:
    st.metric(
        "GW-OTHER Value",
        money(gw_value)
    )

with col4:
    st.metric(
        "GW-OTHER Share",
        f"{gw_share:.2f}%"
    )

with col5:
    st.metric(
        "Policy Orders",
        f"{policy_order_count:,}"
    )

st.caption(
    "Policy orders are orders with both refund and replacement activity "
    "in the cleaned dataset."
)


# ============================================================
# BUSINESS OPPORTUNITY
# ============================================================

st.subheader("Business Opportunity")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "GW-OTHER Refund Pool",
        money(gw_value)
    )

with col2:
    st.metric(
        "High-Confidence Tickets",
        f"{high_conf_tickets:,}"
    )

with col3:
    st.metric(
        "High-Confidence Value",
        money(high_conf_value)
    )

st.info(
    f"""
    **Current AI opportunity:** {high_conf_tickets:,} GW-OTHER tickets
    representing {money(high_conf_value)} of refund value can currently
    be classified at ≥80% model confidence.

    This is **classifiable refund value, not realized savings**.
    Lower-confidence cases should remain under manual review.
    """
)


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📊 Reconciliation",
        "🤖 AI Classification",
        "🔎 GW-OTHER Investigation",
        "⚠️ Policy Compliance",
        "📋 Review Queue"
    ]
)


# ============================================================
# TAB 1: RECONCILIATION
# ============================================================

with tab1:

    st.header("Refund Reconciliation")

    st.write(
        "All refund views should reconcile to the cleaned master refund total."
    )

    # --------------------------------------------------------
    # MONTHLY
    # --------------------------------------------------------

    st.subheader("Monthly Refunds")

    if not monthly.empty:

        monthly["total_refund_inr"] = pd.to_numeric(
            monthly["total_refund_inr"],
            errors="coerce"
        ).fillna(0)

        monthly_display = monthly.copy()

        if "month" in monthly_display.columns:

            chart_data = monthly_display.set_index(
                "month"
            )["total_refund_inr"]

            st.bar_chart(chart_data)

        st.dataframe(
            monthly_display,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning("Monthly refund data not found.")


    # --------------------------------------------------------
    # REASON
    # --------------------------------------------------------

    st.subheader("Refund by Reason")

    if not reason.empty:

        reason["total_refund_inr"] = pd.to_numeric(
            reason["total_refund_inr"],
            errors="coerce"
        ).fillna(0)

        st.dataframe(
            reason,
            use_container_width=True,
            hide_index=True
        )

        if "refund_reason_code" in reason.columns:

            chart_data = reason.set_index(
                "refund_reason_code"
            )["total_refund_inr"]

            st.bar_chart(chart_data)

    else:

        st.warning("Reason data not found.")


    # --------------------------------------------------------
    # AGENT
    # --------------------------------------------------------

    st.subheader("Refund by Agent")

    if not agent.empty:

        agent["total_refund_inr"] = pd.to_numeric(
            agent["total_refund_inr"],
            errors="coerce"
        ).fillna(0)

        st.dataframe(
            agent,
            use_container_width=True,
            hide_index=True
        )


    # --------------------------------------------------------
    # RECONCILIATION CHECK
    # --------------------------------------------------------

    st.subheader("Reconciliation Check")

    monthly_total = (
        monthly["total_refund_inr"].sum()
        if not monthly.empty
        else 0
    )

    reason_total = (
        reason["total_refund_inr"].sum()
        if not reason.empty
        else 0
    )

    agent_total = (
        agent["total_refund_inr"].sum()
        if not agent.empty
        else 0
    )

    reconciliation = pd.DataFrame({
        "View": [
            "Master Refund Total",
            "Monthly Total",
            "Reason Total",
            "Agent Total"
        ],
        "Amount": [
            total_refunds,
            monthly_total,
            reason_total,
            agent_total
        ]
    })

    reconciliation["Status"] = reconciliation["Amount"].apply(
        lambda x: "PASS"
        if abs(x - total_refunds) < 0.01
        else "CHECK"
    )

    st.dataframe(
        reconciliation,
        use_container_width=True,
        hide_index=True
    )

    if all(
        reconciliation["Status"] == "PASS"
    ):
        st.success(
            "✓ All refund views reconcile to the master refund total."
        )
    else:
        st.error(
            "Some refund views do not reconcile."
        )


# ============================================================
# TAB 2: AI CLASSIFICATION
# ============================================================

with tab2:

    st.header("AI-Assisted GW-OTHER Classification")

    st.markdown(
        """
        The classifier predicts a more specific refund reason for
        `GW-OTHER` cases using ticket text and category information.
        """
    )

    # --------------------------------------------------------
    # MODEL METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Model Accuracy",
            "92.96%"
        )

    with col2:
        st.metric(
            "Macro F1",
            "86.92%"
        )

    with col3:
        st.metric(
            "High Confidence",
            f"{high_conf_tickets:,}"
        )

    with col4:
        st.metric(
            "High-Confidence Value",
            money(high_conf_value)
        )


    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    st.subheader("Confidence Breakdown")

    if not ai_confidence.empty:

        ai_confidence["total_refund_inr"] = pd.to_numeric(
            ai_confidence["total_refund_inr"],
            errors="coerce"
        ).fillna(0)

        st.dataframe(
            ai_confidence,
            use_container_width=True,
            hide_index=True
        )

        if "confidence_bucket" in ai_confidence.columns:

            chart_data = ai_confidence.set_index(
                "confidence_bucket"
            )["total_refund_inr"]

            st.bar_chart(chart_data)

    else:

        st.warning("AI confidence data not found.")


    # --------------------------------------------------------
    # FEATURE COMPARISON
    # --------------------------------------------------------

    st.subheader("Model Feature Validation")

    if not feature_comparison.empty:

        st.dataframe(
            feature_comparison,
            use_container_width=True,
            hide_index=True
        )


    # --------------------------------------------------------
    # AI PREDICTED REASONS
    # --------------------------------------------------------

    st.subheader("GW-OTHER Predicted Reasons")

    if not ai_summary.empty:

        st.dataframe(
            ai_summary,
            use_container_width=True,
            hide_index=True
        )


    # --------------------------------------------------------
    # SAMPLE PREDICTIONS
    # --------------------------------------------------------

    if not ai_predictions.empty:

        st.subheader("AI Prediction Samples")

        sample_columns = [
            "ticket_id",
            "category",
            "predicted_reason",
            "confidence",
            "confidence_bucket",
            "refund_amount_inr"
        ]

        sample_columns = [
            c for c in sample_columns
            if c in ai_predictions.columns
        ]

        st.dataframe(
            ai_predictions[sample_columns].head(20),
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# TAB 3: GW-OTHER INVESTIGATION
# ============================================================

with tab3:

    st.header("GW-OTHER Investigation")

    st.markdown(
        f"""
        `GW-OTHER` contains **{gw_tickets:,} refund tickets** worth
        **{money(gw_value)}**, representing **{gw_share:.2f}%**
        of the cleaned refund value.
        """
    )


    # --------------------------------------------------------
    # CATEGORY
    # --------------------------------------------------------

    st.subheader("By Category")

    if not gw_category.empty:

        st.dataframe(
            gw_category,
            use_container_width=True,
            hide_index=True
        )


    # --------------------------------------------------------
    # TEAM
    # --------------------------------------------------------

    st.subheader("By Team")

    if not gw_team.empty:

        st.dataframe(
            gw_team,
            use_container_width=True,
            hide_index=True
        )


    # --------------------------------------------------------
    # KEYWORD SIGNALS
    # --------------------------------------------------------

    st.subheader("Keyword Signals")

    if not gw_keywords.empty:

        gw_keywords["refund_value_inr"] = pd.to_numeric(
            gw_keywords["refund_value_inr"],
            errors="coerce"
        ).fillna(0)

        st.dataframe(
            gw_keywords,
            use_container_width=True,
            hide_index=True
        )


    # --------------------------------------------------------
    # HIGH CONFIDENCE BY REASON
    # --------------------------------------------------------

    st.subheader(
        "High-Confidence Opportunities by Reason"
    )

    if not high_reason.empty:

        st.dataframe(
            high_reason,
            use_container_width=True,
            hide_index=True
        )


    # --------------------------------------------------------
    # HIGH CONFIDENCE BY TEAM
    # --------------------------------------------------------

    st.subheader(
        "High-Confidence Opportunities by Team"
    )

    if not high_team.empty:

        st.dataframe(
            high_team,
            use_container_width=True,
            hide_index=True
        )


    # --------------------------------------------------------
    # HIGH CONFIDENCE BY AGENT
    # --------------------------------------------------------

    st.subheader(
        "High-Confidence Opportunities by Agent"
    )

    if not high_agent.empty:

        st.dataframe(
            high_agent,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# TAB 4: POLICY COMPLIANCE
# ============================================================

with tab4:

    st.header(
        "Refund + Replacement Policy Compliance"
    )

    st.warning(
        """
        The support policy states that a customer must not receive both
        a refund and replacement for the same order.

        These records are flagged for investigation. They are not
        automatically treated as confirmed errors.
        """
    )


    # --------------------------------------------------------
    # POLICY METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Orders Flagged",
            f"{policy_order_count:,}"
        )

    with col2:
        st.metric(
            "Same-Ticket Cases",
            f"{same_ticket_count:,}"
        )

    with col3:
        st.metric(
            "Cross-Ticket Cases",
            f"{cross_ticket_count:,}"
        )

    with col4:
        st.metric(
            "Refund Value Involved",
            money(policy_value)
        )


    # --------------------------------------------------------
    # TABLE
    # --------------------------------------------------------

    if not policy_exceptions.empty:

        st.subheader(
            "Order-Level Policy Exceptions"
        )

        st.dataframe(
            policy_exceptions,
            use_container_width=True,
            hide_index=True
        )

        csv = policy_exceptions.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="⬇️ Download Policy Exception CSV",
            data=csv,
            file_name="order_policy_exceptions.csv",
            mime="text/csv"
        )

    else:

        st.success(
            "No order-level policy exceptions found."
        )


# ============================================================
# TAB 5: MANUAL REVIEW QUEUE
# ============================================================

with tab5:

    st.header("Manual Review Queue")

    st.markdown(
        """
        Lower-confidence AI classifications remain under human review.
        The classifier is decision support, not autonomous refund approval.
        """
    )

    if not manual_queue.empty:

        st.metric(
            "Manual Review Tickets",
            f"{len(manual_queue):,}"
        )

        st.dataframe(
            manual_queue,
            use_container_width=True,
            hide_index=True
        )

        csv = manual_queue.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="⬇️ Download Manual Review Queue",
            data=csv,
            file_name="manual_review_queue.csv",
            mime="text/csv"
        )

    else:

        st.info(
            "Manual review queue file not found."
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🎧 Vireo Audio")

    st.write(
        "Refund Intelligence Tool"
    )

    st.divider()

    st.subheader("Data Status")

    files_to_check = [
        "clean_tickets.csv",
        "refund_monthly.csv",
        "refund_by_reason.csv",
        "refund_by_agent.csv",
        "gw_other_by_category.csv",
        "gw_other_by_team.csv",
        "gw_other_keyword_signals.csv",
        "gw_other_ai_predictions.csv",
        "gw_other_ai_summary.csv",
        "gw_other_ai_confidence.csv",
        "model_feature_comparison.csv",
        "order_policy_exceptions.csv",
        "manual_review_queue.csv"
    ]

    for filename in files_to_check:

        if (DATA_DIR / filename).exists():

            st.success(
                f"✓ {filename}"
            )

        else:

            st.error(
                f"✗ {filename}"
            )

    st.divider()

    st.caption(
        "AI classification is decision support, "
        "not autonomous refund approval."
    )