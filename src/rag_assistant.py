# ============================================================
# VIREO AUDIO - RAG ASSISTANT
# ============================================================
#
# Grounded AI assistant for Vireo Audio refund intelligence.
#
# Pipeline:
#
# User Question
#       ↓
# Query Expansion
#       ↓
# TF-IDF Retrieval
#       ↓
# Source-aware Ranking
#       ↓
# Business-aware Evidence Selection
#       ↓
# Gemini Generation
#       ↓
# Grounded Answer
#
# ============================================================

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from dotenv import load_dotenv
from pypdf import PdfReader

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from google import genai


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    client = genai.Client(
        api_key=GEMINI_API_KEY
    )
else:
    client = None


# ============================================================
# DATA DIRECTORY
# ============================================================

DATA_DIR = Path("data")


# ============================================================
# READ TEXT FILE
# ============================================================

def read_text_file(path):

    try:

        return path.read_text(
            encoding="utf-8",
            errors="ignore"
        )

    except Exception:

        return ""


# ============================================================
# READ PDF
# ============================================================

def read_pdf_file(path):

    text_parts = []

    try:

        reader = PdfReader(
            str(path)
        )

        for page_number, page in enumerate(
            reader.pages,
            start=1
        ):

            try:

                page_text = page.extract_text()

                if page_text:

                    text_parts.append(
                        f"PAGE {page_number}\n"
                        f"{page_text}"
                    )

            except Exception:

                continue

    except Exception:

        return ""

    return "\n\n".join(
        text_parts
    )


# ============================================================
# DATAFRAME → DOCUMENTS
# ============================================================

def dataframe_to_documents(
    dataframe,
    source_name,
    context=""
):

    documents = []

    if dataframe.empty:

        return documents

    for index, row in dataframe.iterrows():

        parts = []

        if context:

            parts.append(
                f"DATASET CONTEXT:\n{context}"
            )

        parts.append(
            f"SOURCE: {source_name}"
        )

        parts.append(
            f"ROW: {index}"
        )

        for column in dataframe.columns:

            value = row[column]

            if pd.isna(value):

                value = ""

            parts.append(
                f"{column}: {value}"
            )

        documents.append(
            {
                "source": source_name,
                "text": "\n".join(parts)
            }
        )

    return documents


# ============================================================
# LOAD ALL DOCUMENTS
# ============================================================

def load_documents():

    documents = []

    # --------------------------------------------------------
    # CSV CONTEXT
    # --------------------------------------------------------

    csv_contexts = {

        "refund_monthly.csv":
            """
This dataset contains monthly refund totals.
It is used to understand refund volume and refund
value over time.
""",

        "refund_by_reason.csv":
            """
This dataset contains refund totals grouped by
audited refund reason code.
""",

        "refund_by_agent.csv":
            """
This dataset contains refund totals grouped by
resolving agent.
""",

        "gw_other_by_category.csv":
            """
This dataset breaks GW-OTHER refunds down by
support category. It is operational evidence for
understanding the types of issues associated with
GW-OTHER.
""",

        "gw_other_by_team.csv":
            """
This dataset breaks GW-OTHER refunds down by
assigned support team. It is operational evidence
for identifying where GW-OTHER activity is concentrated.
""",

        "gw_other_by_agent.csv":
            """
This dataset breaks GW-OTHER refunds down by
individual agent.
""",

        "gw_other_keyword_signals.csv":
            """
This dataset contains keyword-based analytical
signals extracted from GW-OTHER refund tickets.
Signals include delivery issues, goodwill,
duplicate payments, cancellations, returns,
quality issues, warranty and pricing.

These are analytical keyword signals and should not
automatically be interpreted as audited root causes.
""",

        "gw_other_ai_summary.csv":
            """
This dataset summarizes AI classification predictions
for GW-OTHER refund tickets.

These are model-generated signals and are not audited
ground-truth refund reasons.
""",

        "gw_other_ai_confidence.csv":
            """
This dataset contains confidence buckets for AI
classification of GW-OTHER tickets.
""",

        "high_confidence_by_reason.csv":
            """
This dataset summarizes high-confidence AI predictions
by predicted refund reason.

These are model predictions, not audited reason codes.
""",

        "high_confidence_by_team.csv":
            """
This dataset summarizes high-confidence AI predictions
by support team.
""",

        "high_confidence_by_agent.csv":
            """
This dataset summarizes high-confidence AI predictions
by agent.
""",

        "model_feature_comparison.csv":
            """
This dataset compares classifier performance using
different feature combinations.
""",

        "business_opportunity_summary.csv":
            """
This dataset summarizes the business opportunity from
high-confidence AI classification.

Values represent classifiable refund value and must not
be described as realized savings.
""",

        "order_policy_exceptions.csv":
            """
This dataset contains order-level investigation flags
where refund and replacement activity both occurred.

These are investigation candidates and should not
automatically be described as confirmed policy violations.
"""
    }

    # --------------------------------------------------------
    # LOAD CSV FILES
    # --------------------------------------------------------

    for filename, context in csv_contexts.items():

        path = DATA_DIR / filename

        if not path.exists():

            continue

        try:

            df = pd.read_csv(
                path
            )

            documents.extend(
                dataframe_to_documents(
                    df,
                    filename,
                    context
                )
            )

        except Exception as e:

            print(
                f"Could not load {filename}: {e}"
            )

    # --------------------------------------------------------
    # SUPPORT POLICY
    # --------------------------------------------------------

    policy_path = DATA_DIR / "support-policy.pdf"

    if policy_path.exists():

        policy_text = read_pdf_file(
            policy_path
        )

        if policy_text:

            documents.append(
                {
                    "source": "support-policy.pdf",

                    "text": (
                        "SOURCE: support-policy.pdf\n\n"
                        "DATASET CONTEXT:\n"
                        "Vireo Audio Customer Support Operating "
                        "Policy. Primary policy source for refund, "
                        "replacement, SLA, cost, goodwill, ownership "
                        "and data-definition questions.\n\n"
                        f"{policy_text}"
                    )
                }
            )

    # --------------------------------------------------------
    # EMAIL THREAD
    # --------------------------------------------------------

    email_path = DATA_DIR / "email-thread.txt"

    if email_path.exists():

        email_text = read_text_file(
            email_path
        )

        if email_text:

            documents.append(
                {
                    "source": "email-thread.txt",

                    "text": (
                        "SOURCE: email-thread.txt\n\n"
                        "DATASET CONTEXT:\n"
                        "Internal Vireo Audio email discussion "
                        "about refund reconciliation, legacy "
                        "Freshdesk data, duplicate tickets, "
                        "reason codes, refund increases and "
                        "Finance Controller reporting needs.\n\n"
                        f"{email_text}"
                    )
                }
            )

    return documents


# ============================================================
# RETRIEVER
# ============================================================

class VireoRetriever:

    def __init__(self):

        self.documents = load_documents()

        if not self.documents:

            self.vectorizer = None
            self.matrix = None

            return

        texts = [
            document["text"]
            for document in self.documents
        ]

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.98,
            sublinear_tf=True
        )

        self.matrix = self.vectorizer.fit_transform(
            texts
        )

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query,
        top_k=10
    ):

        # ----------------------------------------------------
        # SAFETY
        # ----------------------------------------------------

        if (
            self.vectorizer is None
            or self.matrix is None
            or not self.documents
        ):

            return []

        # ----------------------------------------------------
        # NORMALIZE QUERY
        # ----------------------------------------------------

        query_lower = query.lower().strip()

        # ----------------------------------------------------
        # QUERY EXPANSION
        # ----------------------------------------------------

        expanded_query = query_lower

        query_terms = {

            "gw-other": [
                "gw-other",
                "goodwill",
                "return",
                "refund",
                "delivery",
                "delivery issue",
                "duplicate payment",
                "cancellation",
                "quality issue",
                "warranty",
                "price issue",
                "driver",
                "drivers",
                "reason",
                "reasons",
                "keyword signals",
                "category",
                "team"
            ],

            "why is gw-other so high": [
                "gw-other",
                "drivers",
                "main drivers",
                "refund drivers",
                "keyword signals",
                "category",
                "team",
                "return",
                "delivery",
                "goodwill",
                "duplicate payment",
                "cancellation",
                "quality issue",
                "warranty",
                "refund value"
            ],

            "policy": [
                "policy",
                "refund",
                "replacement",
                "violation",
                "exception",
                "same order",
                "goodwill",
                "store credit"
            ],

            "replacement": [
                "replacement",
                "refund",
                "policy",
                "same order",
                "exception",
                "violation"
            ],

            "arjun": [
                "Arjun",
                "Finance Controller",
                "refund",
                "monthly",
                "reason code",
                "agent",
                "reconcile"
            ],

            "sameer": [
                "Sameer",
                "legacy",
                "Freshdesk",
                "helpdesk",
                "migration",
                "duplicate",
                "refund amount"
            ],

            "priya": [
                "Priya",
                "CSAT",
                "refund",
                "Q4",
                "customer"
            ],

            "neha": [
                "Neha",
                "refund",
                "replacement",
                "Returns Desk"
            ],

            "legacy": [
                "legacy",
                "Freshdesk",
                "helpdesk",
                "migration",
                "duplicate",
                "refund amount",
                "reason code"
            ],

            "freshdesk": [
                "legacy",
                "Freshdesk",
                "migration",
                "duplicate",
                "refund amount",
                "helpdesk"
            ],

            "reconcile": [
                "reconciliation",
                "refund total",
                "monthly",
                "reason",
                "agent"
            ]
        }

        for key, terms in query_terms.items():

            if key in query_lower:

                expanded_query += (
                    " " + " ".join(terms)
                )

        # ----------------------------------------------------
        # GW-OTHER DRIVER QUERY
        # ----------------------------------------------------

        is_gw_other_driver_query = (
            "gw-other" in query_lower
            and any(
                word in query_lower
                for word in [
                    "why",
                    "high",
                    "driver",
                    "drivers",
                    "reason",
                    "reasons",
                    "cause",
                    "causes",
                    "main"
                ]
            )
        )

        if is_gw_other_driver_query:

            expanded_query += """
            gw-other keyword signals
            gw-other category
            gw-other team
            gw-other agent
            delivery issue
            goodwill
            duplicate payment
            cancellation
            return
            quality issue
            warranty
            price issue
            refund value
            ticket count
            operational drivers
            financial drivers
            """

        # ----------------------------------------------------
        # TF-IDF VECTOR
        # ----------------------------------------------------

        query_vector = self.vectorizer.transform(
            [expanded_query]
        )

        similarities = cosine_similarity(
            query_vector,
            self.matrix
        ).flatten()

        # ----------------------------------------------------
        # SOURCE BOOSTING
        # ----------------------------------------------------

        boosted_scores = similarities.copy()

        for idx, document in enumerate(
            self.documents
        ):

            source = document[
                "source"
            ].lower()

            # =================================================
            # GW-OTHER
            # =================================================

            if "gw-other" in query_lower:

                if is_gw_other_driver_query:

                    if (
                        "gw_other_keyword_signals.csv"
                        in source
                    ):

                        boosted_scores[idx] += 0.35

                    elif (
                        "gw_other_by_category.csv"
                        in source
                    ):

                        boosted_scores[idx] += 0.22

                    elif (
                        "gw_other_by_team.csv"
                        in source
                    ):

                        boosted_scores[idx] += 0.20

                    elif (
                        "gw_other_by_agent.csv"
                        in source
                    ):

                        boosted_scores[idx] += 0.08

                    elif (
                        "gw_other_ai_summary.csv"
                        in source
                    ):

                        boosted_scores[idx] += 0.05

                    elif (
                        "gw_other_ai_confidence.csv"
                        in source
                    ):

                        boosted_scores[idx] += 0.03

                else:

                    if (
                        "gw_other_keyword_signals.csv"
                        in source
                    ):

                        boosted_scores[idx] += 0.20

                    elif (
                        "gw_other_by_category.csv"
                        in source
                    ):

                        boosted_scores[idx] += 0.15

                    elif (
                        "gw_other_by_team.csv"
                        in source
                    ):

                        boosted_scores[idx] += 0.12

                    elif (
                        "gw_other_ai_summary.csv"
                        in source
                    ):

                        boosted_scores[idx] += 0.08

            # =================================================
            # POLICY
            # =================================================

            if any(
                word in query_lower
                for word in [
                    "policy",
                    "replacement",
                    "violation",
                    "exception"
                ]
            ):

                if "support-policy.pdf" in source:

                    boosted_scores[idx] += 0.30

                elif (
                    "order_policy_exceptions.csv"
                    in source
                ):

                    boosted_scores[idx] += 0.25

            # =================================================
            # EMAIL
            # =================================================

            if any(
                word in query_lower
                for word in [
                    "arjun",
                    "sameer",
                    "priya",
                    "neha",
                    "email",
                    "freshdesk",
                    "legacy",
                    "migration"
                ]
            ):

                if "email-thread.txt" in source:

                    boosted_scores[idx] += 0.30

            # =================================================
            # REFUND / RECONCILIATION
            # =================================================

            if any(
                word in query_lower
                for word in [
                    "refund",
                    "refunds",
                    "monthly",
                    "reconcile",
                    "reconciliation"
                ]
            ):

                if "refund_monthly.csv" in source:

                    boosted_scores[idx] += 0.15

                elif "refund_by_reason.csv" in source:

                    boosted_scores[idx] += 0.15

                elif "refund_by_agent.csv" in source:

                    boosted_scores[idx] += 0.10

                elif "reconciliation" in source:

                    boosted_scores[idx] += 0.12

        # ----------------------------------------------------
        # RANK
        # ----------------------------------------------------

        ranked_indices = np.argsort(
            boosted_scores
        )[::-1]

        ranked_results = []

        for idx in ranked_indices:

            score = boosted_scores[idx]

            if score <= 0:

                continue

            document = self.documents[idx]

            ranked_results.append(
                {
                    "source": document["source"],
                    "text": document["text"],
                    "score": float(score)
                }
            )

        # ====================================================
        # GW-OTHER FINANCIAL DRIVER MODE
        # ====================================================

        if is_gw_other_driver_query:

            keyword_results = []
            other_results = []

            for result in ranked_results:

                source = result[
                    "source"
                ].lower()

                if (
                    "gw_other_keyword_signals.csv"
                    in source
                ):

                    text = result[
                        "text"
                    ]

                    # ----------------------------------------
                    # Extract refund value
                    # ----------------------------------------

                    value_match = re.search(
                         r"(?:total_refund_inr|refund_value_inr|refund_amount_inr|value_inr)"
                         r"\s*:\s*₹?\s*([\d,]+(?:\.\d+)?)",
                        text,
                        re.IGNORECASE
                    )

                    if value_match:

                        value_text = (
                            value_match.group(1)
                            .replace(",", "")
                            .strip()
                        )

                        try:

                            refund_value = float(
                                value_text
                            )

                        except ValueError:

                            refund_value = 0.0

                    else:

                        refund_value = 0.0

                    result["_refund_value"] = (
                        refund_value
                    )

                    keyword_results.append(
                        result
                    )

                else:

                    other_results.append(
                        result
                    )

            # --------------------------------------------
            # Rank keyword signals by refund value
            # --------------------------------------------

            keyword_results.sort(
                key=lambda item: (
                    item.get(
                        "_refund_value",
                        0.0
                    ),
                    item["score"]
                ),
                reverse=True
            )

            # Top 5 financial drivers
            keyword_results = keyword_results[:5]

            # --------------------------------------------
            # Supporting evidence
            # --------------------------------------------

            other_results.sort(
                key=lambda item: item["score"],
                reverse=True
            )

            source_limits = {

                "gw_other_by_category.csv": 3,

                "gw_other_by_team.csv": 3,

                "gw_other_by_agent.csv": 2,

                "gw_other_ai_summary.csv": 1,

                "gw_other_ai_confidence.csv": 2,

                "business_opportunity_summary.csv": 1,

                "model_feature_comparison.csv": 1,

                "order_policy_exceptions.csv": 3
            }

            supporting_results = []

            source_counts = {}

            for result in other_results:

                source = result[
                    "source"
                ]

                limit = source_limits.get(
                    source,
                    1
                )

                count = source_counts.get(
                    source,
                    0
                )

                if count >= limit:

                    continue

                supporting_results.append(
                    result
                )

                source_counts[source] = (
                    count + 1
                )

                if len(supporting_results) >= 5:

                    break

            # --------------------------------------------
            # Final evidence
            # --------------------------------------------

            selected = (
                keyword_results
                + supporting_results
            )

            selected = selected[:top_k]

            for result in selected:

                result.pop(
                    "_refund_value",
                    None
                )

            return selected

        # ====================================================
        # NORMAL QUERY MODE
        # ====================================================

        source_limits = {

            "gw_other_keyword_signals.csv": 5,

            "gw_other_by_category.csv": 3,

            "gw_other_by_team.csv": 3,

            "gw_other_by_agent.csv": 2,

            "gw_other_ai_summary.csv": 1,

            "gw_other_ai_confidence.csv": 2,

            "business_opportunity_summary.csv": 1,

            "model_feature_comparison.csv": 1,

            "order_policy_exceptions.csv": 3
        }

        selected = []

        source_counts = {}

        for result in ranked_results:

            source = result[
                "source"
            ]

            limit = source_limits.get(
                source,
                1
            )

            current_count = source_counts.get(
                source,
                0
            )

            if current_count >= limit:

                continue

            selected.append(
                result
            )

            source_counts[source] = (
                current_count + 1
            )

            if len(selected) >= top_k:

                break

        # ----------------------------------------------------
        # FALLBACK
        # ----------------------------------------------------

        if len(selected) < min(
            top_k,
            len(ranked_results)
        ):

            selected_keys = {
                (
                    item["source"],
                    item["text"]
                )
                for item in selected
            }

            for result in ranked_results:

                key = (
                    result["source"],
                    result["text"]
                )

                if key in selected_keys:

                    continue

                selected.append(
                    result
                )

                selected_keys.add(
                    key
                )

                if len(selected) >= top_k:

                    break

        return selected


# ============================================================
# CREATE RETRIEVER
# ============================================================

retriever = VireoRetriever()


# ============================================================
# GEMINI ANSWER GENERATION
# ============================================================

def generate_answer(
    question,
    results
):

    # --------------------------------------------------------
    # NO EVIDENCE
    # --------------------------------------------------------

    if not results:

        return (
            "I could not find sufficient evidence in the "
            "Vireo policy, email thread, or analysis outputs "
            "to answer this question reliably."
        )

    # --------------------------------------------------------
    # API KEY
    # --------------------------------------------------------

    if client is None:

        return (
            "Gemini API key is not configured.\n\n"
            "Please add GEMINI_API_KEY to the .env file "
            "in the project root."
        )

    # --------------------------------------------------------
    # BUILD EVIDENCE
    # --------------------------------------------------------

    evidence_blocks = []

    for i, result in enumerate(
        results,
        start=1
    ):

        evidence_blocks.append(
            f"""
EVIDENCE {i}

SOURCE:
{result['source']}

RETRIEVAL RELEVANCE:
{result['score']:.2f}

CONTENT:
{result['text']}
"""
        )

    evidence = "\n".join(
        evidence_blocks
    )

    # --------------------------------------------------------
    # GROUNDED PROMPT
    # --------------------------------------------------------

    prompt = f"""
You are the Vireo Audio Refund Intelligence Assistant.

You help a Finance Controller or Support Manager
understand Vireo Audio's refund data, support policy,
operational patterns, and AI classification results.

USER QUESTION:

{question}


============================================================
RETRIEVED VIREO EVIDENCE
============================================================

{evidence}


============================================================
TASK
============================================================

Answer the user's question using ONLY the retrieved
Vireo evidence above.

Do not use outside knowledge.

Do not invent numbers, causes, business rules, or facts.

Do not simply repeat individual CSV rows.

Reason across the available evidence and synthesize
the information carefully.


============================================================
EVIDENCE PRIORITY
============================================================

Use this priority order:

1. Directly observed or audited dataset analysis.

2. Keyword signals and operational breakdowns.

3. Category and team breakdowns.

4. Agent-level breakdowns.

5. AI-generated classification or prediction results.

6. Interpretation based on the combined evidence.

AI predictions MUST NOT be treated as audited ground truth.


============================================================
GW-OTHER RULE
============================================================

If the question is about GW-OTHER, why GW-OTHER is high,
GW-OTHER drivers, or main reasons behind GW-OTHER:

1. Prioritize gw_other_keyword_signals.csv.

2. Use gw_other_by_category.csv and
   gw_other_by_team.csv as supporting evidence.

3. Use gw_other_by_agent.csv only when relevant.

4. Treat gw_other_ai_summary.csv and AI-predicted reasons
   as MODEL SIGNALS, not confirmed underlying reasons.

5. Never state that an AI-predicted reason is the main
   driver unless independent evidence supports it.

6. When several drivers exist, report the strongest
   3-5 drivers.

7. Clearly label model predictions:

   "the AI classifier predicts..."

   or

   "the model signal suggests..."

8. Do not state that GW-OTHER is caused by an AI-predicted
   reason unless the retrieved evidence directly supports it.


============================================================
GENERAL REASONING RULES
============================================================

1. Identify facts directly supported by evidence.

2. Compare relevant numbers when useful.

3. Find patterns across evidence sources.

4. Explain what combined patterns suggest.

5. Distinguish between:

   - directly supported facts
   - reasonable interpretation
   - unsupported conclusions

6. Do not claim causation unless evidence proves it.

7. If evidence is insufficient, explicitly state
   what cannot be determined.

8. Preserve exact monetary values, percentages,
   ticket counts, and reason codes.

9. Never invent numbers or facts.

10. Never use outside knowledge.

11. Never describe "classifiable refund value"
    as realized savings.

12. Never describe policy investigation flags as
    confirmed policy violations unless explicitly proven.

13. The AI classifier is a decision-support tool.

14. It does not automatically approve refunds
    or replacements.

15. Lower-confidence classifications require
    human review.

16. Clearly distinguish actual coded reasons from
    AI-predicted reasons.

17. Do not give AI predictions more importance merely
    because their refund value is large.

18. If AI predictions conflict with direct operational
    evidence, present both and state that the model
    prediction requires validation.

19. Keep the answer concise and useful for
    a Finance Controller.

20. Do not expose hidden chain-of-thought.
    Provide conclusions and useful supporting
    explanations only.


============================================================
ANSWER FORMAT
============================================================

Give the direct answer first.

Then provide the following sections:

### Evidence

Give the most important supporting facts,
numbers, patterns, and sources.

### Interpretation

Explain what the combined evidence suggests.

### Limitation

Explain what the evidence does not prove.


============================================================
FINAL CHECK
============================================================

Before answering, verify:

- Are all numbers present in the retrieved evidence?
- Did I distinguish actual analysis from AI predictions?
- Did I avoid unsupported causation?
- For GW-OTHER, did I prioritize keyword signals,
  category data, and team data?
- Did I avoid calling classifiable value savings?
- Did I avoid calling policy flags confirmed violations?

Now answer the user's question.
"""

    # --------------------------------------------------------
    # GEMINI MODELS
    # --------------------------------------------------------

    models_to_try = [
        "gemini-3.8-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.6-flash"
    ]

    errors = []

    # --------------------------------------------------------
    # TRY MODELS
    # --------------------------------------------------------

    for model_name in models_to_try:

        try:

            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )

            if response.text:

                return response.text.strip()

        except Exception as e:

            errors.append(
                f"{model_name}: {str(e)}"
            )

            continue

    # --------------------------------------------------------
    # ALL MODELS FAILED
    # --------------------------------------------------------

    return (
        "Gemini could not generate the answer using "
        "the available models.\n\n"
        "Model errors:\n\n"
        + "\n\n".join(errors)
    )


# ============================================================
# MAIN RAG FUNCTION
# ============================================================

def ask_vireo(
    question,
    top_k=10
):

    if not question:

        return (
            "Please enter a question.",
            []
        )

    question = question.strip()

    if not question:

        return (
            "Please enter a question.",
            []
        )

    # --------------------------------------------------------
    # RETRIEVE
    # --------------------------------------------------------

    results = retriever.search(
        question,
        top_k=top_k
    )

    # --------------------------------------------------------
    # GENERATE
    # --------------------------------------------------------

    answer = generate_answer(
        question,
        results
    )

    return (
        answer,
        results
    )