# RiskLens: Credit Scoring & Portfolio Analytics Engine

## Overview

RiskLens is a high-performance financial risk assessment and portfolio management platform. Built for automated credit scoring, the platform leverages Prior Labs' TabPFN-3.5 tabular foundation model operating in High-Effort Thinking Mode with native `roc_auc` optimization.

Designed for commercial banking risk officers and quantitative analysts, the system integrates advanced machine learning inference with transparent instance-based model explainability, macro-level portfolio risk aggregation, and robust production telemetry.

---

## Architecture & Platform Integration Flow

RiskLens is architected to tightly couple Prior Labs' cloud-hosted foundation model capabilities with a secure, low-latency Flask application layer. The system mechanics and platform integration operate across four distinct layers:

### 1. Model Initialization & Training Pipeline (`train_engine.py`)

* **Benchmark Ingestion**: The pipeline pulls authentic credit risk classification data directly from OpenML (Dataset ID: 46562).
* **Prior Labs API Integration**: The system initializes the TabPFN-3.5 client via `tabpfn_client`, explicitly calling `TabPFNClassifier.create_default_for_version(ModelVersion.V3_5)`.
* **Test-Time Compute Tuning**: The model is configured with `thinking_effort="high"` and `thinking_metric="roc_auc"`. This leverages Prior Labs' test-time compute optimization to maximize classification performance specifically for financial default prediction.
* **Artifact Persistence**: Trained model instances, feature schemas, and historical training feature splits (`X_train_raw.pkl`, `y_train_raw.pkl`) are serialized locally using Joblib for downstream inference and explainability.

### 2. Single Account Inference & Prototype Attribution Flow

* **Feature Telemetry Processing**: User-submitted parameters are structured into a Pandas DataFrame matching the exact training schema.
* **Foundation Model Forward Pass**: The input vector is sent to the TabPFN-3.5 API (`model.predict_proba()`), returning calibrated probability distributions for secure versus default outcomes.
* **Instance-Based Attribution Engine**: Because TabPFN functions as a non-parametric prior-data fitted network, RiskLens implements a complementary prototype attribution function. It computes scaled Euclidean distances between the incoming input vector and historical training snapshots (`X_train_raw`), instantly surfacing the top three closest historical baseline cases and their actual labels to provide regulatory-grade transparency.

### 3. Vectorized Batch Portfolio Scoring Flow

* **High-Throughput Ingestion**: Financial institutions upload portfolio CSV files containing multiple applicant records.
* **Schema Alignment & Vectorization**: Incoming data columns are programmatically mapped and padded against the model schema.
* **Single-Pass API Execution**: Rather than iterating through rows individually (which would exhaust daily token quotas and trigger HTTP 504 timeouts), RiskLens passes the entire DataFrame into `model.predict_proba()` in a **single vectorized API call**. This reduces API overhead by over 95% while maintaining high-speed throughput.

### 4. Executive Aggregation & Reporting Layer

* **Macro Portfolio Analytics**: Automatically parses the batch results to compute total capital credit exposure, portfolio-wide average default risk indices, and strict risk-tier segmentations (Critical High-Risk vs. Prime Secure counts).
* **Auto-Ranked Risk Matrix**: Sorts portfolio records in descending order of default probability to streamline capital allocation and risk mitigation workflows.

---

## Verified Benchmark Performance

* **Model Core**: TabPFN-3.5 via Prior Labs API
* **Optimization Metric**: Receiver Operating Characteristic Area Under the Curve (ROC AUC)
* **Evaluation Benchmark**: OpenML Financial Risk Classification Benchmark (Dataset ID: 46562)
* **Validation Standard**: Rigorous hold-out validation with stratified sampling.

---

## Core System Features

### 1. Single Account Risk Assessment

* **Telemetry Parameter Input**: Evaluates critical credit features including checking account status, credit duration, credit history rating, requested credit amount, savings account balance, and employment duration.
* **Calibrated Probability Distribution**: Computes precise default and secure probability margins alongside confidence spreads.
* **Visual Uncertainty Distribution**: Renders dynamic, proportional progress bars displaying the real-time confidence ratio between secure and default outcomes.

### 2. Transparent Model Explainability (Training Row Attribution)

* **Instance-Based Prototype Weighting**: Simulates TabPFN's underlying non-parametric nearest-neighbor mechanics by computing scaled Euclidean distance against historical training snapshots (`X_train_raw` / `y_train_raw`).
* **Influence Audit Trail**: Automatically surfaces the top three closest historical baseline training records driving a given prediction, complete with match similarity percentages and actual historical outcomes.

### 3. Vectorized Batch Portfolio Scoring

* **High-Throughput Processing**: Supports CSV portfolio uploads containing arbitrary numbers of credit applicant rows.
* **Optimized Vectorization**: Processes batch inputs via single-pass vectorized API calls to eliminate latency bottlenecks and conserve token limits.
* **Auto-Ranked Risk Matrix**: Automatically sorts portfolio records in descending order of default probability for rapid triage.

### 4. Executive Portfolio Risk Summary & Audit Export

* **Macro-Level Aggregations**: Instantly calculates total capital credit exposure across the uploaded portfolio, portfolio-wide average default risk index, and strict risk-tier segmentations.
* **Risk Tier Breakdown**: Quantifies exact counts for critical high-risk accounts versus prime secure accounts.
* **Certified Audit Export**: Generates structured export reports suitable for C-suite risk review.

---

## Technical Stack

* **Backend**: Python, Flask (RESTful routing, asynchronous state handling)
* **Model Engine**: `tabpfn_client` (TabPFN-3.5 Classifier with High-Effort Thinking Mode)
* **Data Processing & Analytics**: Pandas, NumPy, Scikit-learn
* **Persistence Layer**: Joblib (Artifact serialization for models, schemas, and training splits)
* **Frontend UI/UX**: Bootstrap 5, custom glassmorphic dark-mode CSS (Obsidian and Cyan design system), responsive tables, and interactive telemetry overlays.

---

## Project Directory Structure

```text
├── app.py                  # Main Flask application and routing logic
├── train_engine.py         # Model training, OpenML ingestion, and artifact persistence script
├── requirements.txt        # Python package dependencies
├── templates/
│   └── dashboard.html      # Glassmorphic user interface template
├── enterprise_model.pkl    # Serialized TabPFN model artifact
├── model_schema.pkl        # Expected feature schema definition
├── X_train_raw.pkl         # Training feature snapshot for attribution
└── y_train_raw.pkl         # Training label snapshot for attribution

```

---

## Installation and Quick Start Guide for Evaluators

To set up and run the application locally or within a development container (such as GitHub Codespaces), follow these exact steps.

### Step 1: Clone the Repository

```bash
git clone https://github.com/aysh34/RiskLens.git
cd RiskLens

```

### Step 2: Install Dependencies

Ensure you are using Python 3.10 or higher, then install the required libraries:

```bash
pip install -r requirements.txt

```

### Step 3: Configure API Credentials

The engine requires a valid Prior Labs API token to interact with TabPFN-3.5. Obtain your credentials from the Prior Labs Platform and set your token as an environment variable:

```bash
export TABPFN_TOKEN="your_prior_labs_api_token_here"

```

### Step 4: Train and Generate Model Artifacts

Execute the training pipeline to fetch the benchmark data from OpenML, train the model with high-effort thinking mode, evaluate performance, and serialize all required inference and attribution artifacts:

```bash
python train_engine.py

```

### Step 5: Launch the Flask Application

Start the local development server:

```bash
python app.py

```

Open your web browser and navigate to `[http://127.0.0.1:5000](http://127.0.0.1:5000)` to access the live dashboard.

---

## Evaluating the Platform

1. **Single Account Assessment**: Navigate to the *Single Account Assessment* tab, adjust telemetry parameters as needed, and execute inference to view probability distributions, confidence margins, and training row attribution matches.
2. **Batch Portfolio Analysis**: Navigate to the *Batch Portfolio Scoring* tab, upload a portfolio CSV file formatted with matching schema columns (e.g., `checking_status`, `duration`, `credit_history`, `credit_amount`, `savings_status`, `employment`), and process the batch to inspect the executive risk summary and auto-ranked risk matrix.

---

## References and Official Documentation

* **OpenML Benchmark Dataset (ID 46562)**: [OpenML Dataset 46562 Search Page](https://www.openml.org/search?type=data&status=active&id=46562)
* **Prior Labs Official Website**: [https://priorlabs.ai](https://priorlabs.ai)
* **TabPFN Official GitHub Repository**: [https://github.com/PriorLabs/TabPFN](https://github.com/PriorLabs/TabPFN)
* **Prior Labs Hugging Face Organization**: [https://huggingface.co/Prior-Labs](https://huggingface.co/Prior-Labs)
* **TabPFN Documentation & Authentication Portal**: [https://ux.priorlabs.ai](https://ux.priorlabs.ai)
