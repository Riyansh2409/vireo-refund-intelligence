# Vireo Audio Refund Intelligence

AI-assisted refund intelligence and support-ticket analysis tool built for the Vireo Audio Support Tickets assignment.

The tool reconciles current and legacy helpdesk refund data, investigates the `GW-OTHER` refund pool, classifies refund reasons using machine learning, identifies refund/replacement investigation flags, and provides a grounded RAG assistant for business questions.

---

## 1. Business Goal

The primary business goal is to turn an unreliable refund export into a reconciled, explainable refund view that Finance and Support can use for monthly reporting and investigation.

### Key Results

- **2,340 refund tickets**
- **₹67,09,932 total refund value**
- **991 GW-OTHER refund tickets**
- **₹29,07,036 GW-OTHER refund value**
- **43.32% of total refund value is GW-OTHER**

The AI classifier further identifies:

- **220 GW-OTHER tickets at ≥80% confidence**
- **₹6,26,905 classifiable refund value at ≥80% confidence**
- **771 lower-confidence tickets requiring manual review**

> **Important:** ₹6,26,905 is classifiable refund value, not realized savings. The tool does not claim that this amount will automatically be saved.

---

## 2. What the Tool Does

The application provides five main capabilities.

### 1. Refund Reconciliation

Cleans and reconciles current helpdesk and legacy Freshdesk data.

The cleaning process:

1. Normalizes legacy refund amounts.
2. Detects duplicate ticket IDs.
3. Validates the legacy/current refund amount relationship.
4. Prefers the current helpdesk record when the same ticket exists in both systems.
5. Produces one clean row per ticket.
6. Reconciles monthly, reason-level, and agent-level refund totals.

### 2. GW-OTHER Investigation

Investigates the largest refund-reason pool using:

- Category breakdown
- Team breakdown
- Agent breakdown
- Keyword-based signals
- AI classification predictions
- Confidence levels

The keyword investigation identifies major operational signals including:

- Delivery issues
- Goodwill
- Cancellation
- Duplicate payment
- Returns
- Warranty
- Quality issues
- Price issues

These are analytical signals and are not automatically treated as audited root causes.

### 3. AI Refund Classification

A TF-IDF + Logistic Regression classifier predicts a more specific refund reason for `GW-OTHER` tickets.

The model was evaluated on a held-out test set.

### Model Performance

| Metric | Result |
|---|---:|
| Accuracy | **92.96%** |
| Macro F1 | **86.92%** |
| Weighted F1 | **92.70%** |

The model is used as decision support rather than autonomous approval.

### 4. Policy Compliance Investigation

The tool checks order-level refund and replacement activity.

Current investigation output:

- **192 orders** with both refund and replacement activity
- **100 same-ticket investigation flags**
- **92 cross-ticket investigation flags**
- **₹6,50,171 refund value involved**

These are investigation candidates, not automatically confirmed policy violations.

The support policy states that a customer should not receive both a refund and replacement for the same order and that errors must be escalated to the Team Lead and Finance.

### 5. RAG Assistant

The application includes an **Ask the Evidence** assistant.

Users can ask questions such as:

```text
Why is GW-OTHER so high?
```

```text
What are the main drivers behind GW-OTHER refunds?
```

```text
What are the refund and replacement policy rules?
```

```text
Why was the legacy refund export overstated?
```

The assistant:

1. Retrieves relevant evidence from the Vireo datasets, support policy, and email thread.
2. Uses TF-IDF similarity to rank relevant evidence.
3. Applies query expansion and source-aware ranking.
4. Prioritizes relevant business evidence.
5. Sends the retrieved evidence to Gemini.
6. Generates a grounded business answer using only the retrieved evidence.
7. Displays the evidence sources used for the answer.

The RAG assistant is designed to distinguish between:

- Observed facts
- Analytical signals
- Model predictions
- Reasonable interpretation
- Unsupported conclusions

It does not treat AI predictions as audited ground truth.

---

## 3. Architecture

```text
                    Vireo Audio Data
                           |
          +----------------+----------------+
          |                |                |
      tickets.csv      agents.csv      Other Data
          |
          v
    Data Cleaning
          |
          v
  clean_tickets.csv
          |
    +-----+-------------------+
    |                         |
    v                         v
Reconciliation          Refund Analysis
    |                         |
    |                  +------+------+
    |                  |             |
    |               GW-OTHER       AI Model
    |               Analysis       Validation
    |                  |             |
    |                  +------+------+
    |                         |
    v                         v
Policy Compliance       Opportunity Analysis
    |                         |
    +-------------+-----------+
                  |
                  v
          Streamlit Dashboard
                  |
                  v
           Ask the Evidence
                  |
                  v
          TF-IDF Retrieval
                  |
                  v
         Retrieved Evidence
                  |
                  v
                Gemini
                  |
                  v
          Grounded Answer
```

---

## 4. Project Structure

```text
vireo-refund-intelligence/
│
├── app.py
├── README.md
├── requirements.txt
├── submission-form.md
├── .gitignore
│
├── data/
│   ├── tickets.csv
│   ├── agents.csv
│   ├── customers.csv
│   ├── orders.csv
│   ├── products.csv
│   ├── support-policy.pdf
│   ├── email-thread.txt
│   │
│   ├── clean_tickets.csv
│   ├── refund_by_reason.csv
│   ├── refund_by_agent.csv
│   ├── refund_monthly.csv
│   │
│   ├── gw_other_by_category.csv
│   ├── gw_other_by_team.csv
│   ├── gw_other_by_agent.csv
│   ├── gw_other_keyword_signals.csv
│   ├── gw_other_ai_summary.csv
│   ├── gw_other_ai_confidence.csv
│   │
│   ├── high_confidence_by_reason.csv
│   ├── high_confidence_by_team.csv
│   ├── high_confidence_by_agent.csv
│   ├── manual_review_queue.csv
│   │
│   ├── model_feature_comparison.csv
│   ├── business_opportunity_summary.csv
│   └── order_policy_exceptions.csv
│
└── src/
    ├── __init__.py
    ├── data_cleaning.py
    ├── reconciliation.py
    ├── analysis.py
    ├── ai_classifier.py
    ├── gw_other_analysis.py
    ├── model_validation.py
    ├── opportunity_analysis.py
    ├── policy_compliance.py
    ├── order_policy_check.py
    └── rag_assistant.py
```

---

## 5. Technology Stack

### Data Processing

- Python
- pandas
- NumPy

### Machine Learning

- scikit-learn
- TF-IDF
- Logistic Regression

### RAG

- TF-IDF retrieval
- Gemini API
- Google GenAI SDK

### Document Processing

- PyPDF

### Application

- Streamlit

### Development

- Git
- GitHub
- Python virtual environment
- `.env` for API credentials

---

## 6. Requirements

Recommended environment:

- Python 3.10+
- pip
- Git
- Gemini API key for the RAG assistant

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 7. Setup

### Clone the Repository

```bash
git clone https://github.com/Riyansh2409/vireo-refund-intelligence.git
cd vireo-refund-intelligence
```

### Create Virtual Environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 8. Gemini API Configuration

Create a `.env` file in the project root:

```text
GEMINI_API_KEY=your_gemini_api_key_here
```

The `.env` file should not be committed to Git.

The Gemini API is used for answer generation in the RAG assistant.

The retrieval layer runs locally and retrieves relevant evidence from the project's datasets, support policy, and email thread before the evidence is passed to Gemini.

---

## 9. Run the Data Pipeline

Run the cleaning step first:

```bash
python -m src.data_cleaning
```

This generates:

```text
data/clean_tickets.csv
data/refund_by_reason.csv
data/refund_by_agent.csv
data/refund_monthly.csv
```

Run reconciliation:

```bash
python -m src.reconciliation
```

Run the GW-OTHER investigation:

```bash
python -m src.gw_other_analysis
```

Run the AI classifier:

```bash
python -m src.ai_classifier
```

Run model validation:

```bash
python -m src.model_validation
```

Run business opportunity analysis:

```bash
python -m src.opportunity_analysis
```

Run policy compliance checks:

```bash
python -m src.policy_compliance
```

Run order-level policy investigation:

```bash
python -m src.order_policy_check
```

---

## 10. Run the Streamlit Application

After generating the analysis outputs:

```bash
streamlit run app.py
```

The dashboard will open in the browser.

---

## 11. Dashboard Sections

The Streamlit dashboard contains the following sections.

### Executive Summary

Shows:

- Total refund value
- Refund ticket count
- GW-OTHER value
- GW-OTHER share
- Policy investigation flags

### Reconciliation

Shows:

- Monthly refund totals
- Refunds by reason
- Refunds by agent
- Reconciliation checks

The monthly, reason-level, and agent-level totals reconcile to the same master refund total.

### AI Classification

Shows:

- Model accuracy
- Macro F1
- Confidence distribution
- Predicted reasons
- Feature-set validation
- High-confidence classification opportunity

### GW-OTHER Investigation

Shows:

- Category breakdown
- Team breakdown
- Agent breakdown
- Keyword signals
- AI prediction summary
- Confidence levels

### Policy Compliance

Shows:

- Order-level investigation flags
- Same-ticket flags
- Cross-ticket flags
- Refund value involved

### Review Queue

Shows lower-confidence AI predictions requiring human review.

### Ask the Evidence

Allows users to ask natural-language questions about:

- Refund analysis
- GW-OTHER
- Support policy
- Email thread
- AI classification
- Policy investigation

---

## 12. Data Cleaning Logic

The raw ticket export contains duplicate ticket IDs because some legacy Freshdesk tickets were re-imported into the current helpdesk.

The legacy system also stores monetary values differently from the current helpdesk.

The cleaning process therefore does not blindly remove duplicate rows.

Instead:

1. Legacy refund amounts are normalized.
2. The legacy/current refund amount relationship is validated.
3. Duplicate ticket IDs are identified.
4. Where the same ticket exists in both systems, the current helpdesk record is preferred.
5. The final dataset contains one row per unique ticket.
6. Refund totals are recalculated from the cleaned dataset.

### Final Cleaned Dataset

| Metric | Result |
|---|---:|
| Raw ticket rows | 12,238 |
| Unique ticket IDs | 11,600 |
| Refund tickets | 2,340 |
| Total refunds | ₹67,09,932 |

---

## 13. Refund Reconciliation

The cleaned refund total is:

**₹67,09,932**

The same total is reproduced independently by:

- Monthly refund aggregation
- Refund reason aggregation
- Agent aggregation

All three views reconcile to the master refund total.

This provides a single consistent baseline for Finance reporting.

---

## 14. GW-OTHER Investigation

`GW-OTHER` is the largest refund reason in the cleaned data.

| Metric | Result |
|---|---:|
| GW-OTHER tickets | 991 |
| GW-OTHER value | ₹29,07,036 |
| Share of refund value | 43.32% |

### Top Keyword-Based Signals

| Signal | Tickets | Refund Value |
|---|---:|---:|
| Delivery Issue | 176 | ₹5,29,389 |
| Goodwill | 121 | ₹3,91,324 |
| Cancellation | 120 | ₹3,55,854 |
| Duplicate Payment | 131 | ₹3,50,253 |
| Return | 105 | ₹3,37,283 |

These keyword signals are analytical indicators extracted from ticket content.

They should not automatically be treated as audited root causes.

---

## 15. Machine Learning Model

The classifier uses:

```text
TF-IDF
   +
Logistic Regression
```

### Input Features

```text
customer_message
agent_notes
category
```

The classifier was evaluated on a held-out test set.

### Performance

| Metric | Result |
|---|---:|
| Accuracy | 92.96% |
| Macro F1 | 86.92% |
| Weighted F1 | 92.70% |

Feature-set validation was also performed to compare different combinations of customer message, agent notes, and category.

The model is used as decision support and does not automatically approve refunds or replacements.

---

## 16. Confidence-Based Review

The model assigns a confidence score to each GW-OTHER prediction.

### High-Confidence Threshold

**Confidence ≥ 80%**

### Current Results

| Metric | Result |
|---|---:|
| High-confidence tickets | 220 |
| Classifiable refund value | ₹6,26,905 |

Lower-confidence predictions remain in the manual review queue.

### Manual Review Queue

| Metric | Result |
|---|---:|
| Tickets | 771 |
| Refund value | ₹22,80,131 |

> **Important:** ₹6,26,905 represents classifiable refund value. It is not claimed as realized or guaranteed savings.

---

## 17. Policy Compliance Investigation

The support policy states that a customer should not receive both a refund and replacement for the same order.

The tool therefore performs an order-level investigation.

### Current Results

| Metric | Result |
|---|---:|
| Orders with refund + replacement activity | 192 |
| Same-ticket investigation flags | 100 |
| Cross-ticket investigation flags | 92 |
| Refund value involved | ₹6,50,171 |

These records are investigation candidates.

They are not automatically treated as confirmed policy violations because the underlying order activity requires operational review.

---

## 18. RAG Assistant

The RAG assistant follows a two-stage architecture.

### Retrieval

The local retrieval layer:

1. Normalizes the user's question.
2. Expands relevant business terms.
3. Converts the knowledge base into TF-IDF vectors.
4. Calculates similarity between the question and available evidence.
5. Applies source-aware ranking.
6. Selects the most relevant evidence.

For GW-OTHER driver questions, the retriever prioritizes:

1. `gw_other_keyword_signals.csv`
2. `gw_other_by_category.csv`
3. `gw_other_by_team.csv`
4. `gw_other_by_agent.csv`

AI prediction files are treated as supporting model signals rather than audited root causes.

### Generation

The retrieved evidence is passed to Gemini.

The generation prompt instructs the model to:

- Use only retrieved Vireo evidence.
- Avoid outside knowledge.
- Preserve exact numbers.
- Distinguish facts from interpretation.
- Avoid unsupported causal claims.
- Treat AI predictions as model signals.
- Never describe classifiable value as guaranteed savings.
- Never describe policy flags as confirmed violations.
- Keep lower-confidence predictions subject to human review.

The application also displays the evidence sources used to generate each answer.

---

## 19. Example RAG Queries

```text
Why is GW-OTHER so high?
```

```text
What are the main drivers behind GW-OTHER refunds?
```

```text
What are the refund and replacement policy rules?
```

```text
Why was the legacy refund export overstated?
```

```text
What is the total refund value?
```

```text
How accurate is the AI classifier?
```

```text
Which orders need policy investigation?
```

---

## 20. Validation

The project includes multiple validation layers.

### Data Validation

- Duplicate ticket ID audit
- Legacy/current monetary normalization
- Unique-ticket validation
- Missing refund reason check
- Missing agent check

### Reconciliation Validation

```text
Monthly total = Master refund total
Reason total  = Master refund total
Agent total   = Master refund total
```

All reconciliation checks pass.

### Model Validation

The classifier is evaluated using a held-out test set rather than only training data.

### Feature Validation

The following feature combinations were compared:

```text
Customer Message
Customer Message + Agent Notes
Customer Message + Agent Notes + Category
```

The combined feature set produced the strongest tested performance.

### Policy Validation

Refund and replacement activity is checked at the order level instead of relying only on individual ticket-level checks.

---

## 21. Limitations

This is a decision-support prototype rather than a production support system.

Known limitations:

1. The classifier is retrospective.
2. Agent notes can contain information recorded after resolution, so offline model performance may be higher than real-time pre-resolution performance.
3. Keyword signals are analytical indicators and are not automatically audited root causes.
4. AI predictions are not ground truth.
5. Lower-confidence predictions require human review.
6. Policy investigation flags require operational review.
7. The prototype does not automatically approve refunds or replacements.
8. Production authentication, monitoring, model drift monitoring, and deployment infrastructure are not included.
9. Gemini API access is required for generated RAG answers.

---

## 22. Deliberately Excluded

The prototype deliberately does not perform:

- Automatic refund approval
- Automatic replacement approval
- Automatic customer communication
- Autonomous financial decisions
- Guaranteed savings calculations
- Automatic policy enforcement

The system is designed to surface evidence and prioritize human review.

---

## 23. Reproducibility

A clean run should follow this sequence:

```bash
python -m src.data_cleaning
python -m src.reconciliation
python -m src.gw_other_analysis
python -m src.ai_classifier
python -m src.model_validation
python -m src.opportunity_analysis
python -m src.policy_compliance
python -m src.order_policy_check
streamlit run app.py
```

The generated CSV outputs are consumed by the Streamlit application and RAG assistant.

---

## 24. Key Takeaway

The project converts an unreliable refund export into a reconciled and explainable refund intelligence workflow.

### Baseline

- **₹67.10 lakh total refund value**
- **2,340 refund tickets**

### GW-OTHER

- **₹29.07 lakh**
- **43.32% of total refund value**

### AI Classification

- **92.96% accuracy**
- **86.92% macro F1**
- **220 high-confidence GW-OTHER tickets**
- **₹6.27 lakh classifiable refund value**

The tool does not present this amount as guaranteed savings.

Instead, it provides Finance and Support with:

- A reconciled refund baseline
- Explainable GW-OTHER drivers
- AI-assisted reason classification
- Prioritized manual review
- Order-level policy investigation flags
- A grounded natural-language evidence assistant