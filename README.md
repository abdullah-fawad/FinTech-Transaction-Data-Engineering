# FinTech Data Engineering — Automated Transaction Processing & Analytics Platform

An end-to-end FinTech data engineering and analytics platform for processing, standardizing, categorizing, and analyzing financial transactions. The system combines batch ETL, incremental processing, streaming ingestion, machine learning-based transaction categorization, PostgreSQL storage, Apache Airflow orchestration, AWS S3, Apache Kafka, and Power BI.

The project uses a simulated banking transaction source to demonstrate how raw financial transaction data can be transformed into reliable, structured data for financial and cashflow analytics.

---

## 1. Project Overview

Financial transaction data is often messy, inconsistent, and difficult to use directly for analytics. Transaction descriptions may contain spelling variations, different languages, inconsistent formatting, missing values, and unreliable source category labels.

This project addresses these challenges through an end-to-end data pipeline that:

* Ingests transaction data from a simulated banking source
* Stores raw data in AWS S3
* Cleans and validates transaction records
* Standardizes transaction descriptions and fields
* Applies a canonical business taxonomy
* Categorizes transactions using a hybrid approach
* Assigns confidence levels to predictions
* Stores processed transactions in PostgreSQL
* Supports incremental ETL processing
* Supports Kafka-based streaming ingestion
* Automates workflows using Apache Airflow
* Provides financial analytics through Power BI

The project was developed with an emphasis on **data quality, reproducibility, auditability, and conservative financial categorization**.

---

# 2. Architecture

## Batch ETL Architecture

```text
                    SIMULATED BANK
                          │
                          ▼
                    AWS S3 — RAW
                          │
                          ▼
                  Apache Airflow
                          │
                          ▼
              Data Cleaning & Validation
                          │
                          ▼
               Canonical Standardization
                          │
                          ▼
              Hybrid Categorization Engine
                          │
             ┌────────────┼────────────┐
             ▼            ▼            ▼
        Approved        Rules          ML
        Mappings     / Evidence    Classifier
             └────────────┼────────────┘
                          ▼
                    Confidence
                       Routing
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
             AUTO        REVIEW      UNKNOWN
                          │
                          ▼
                   Processed Data
                          │
                          ▼
                     PostgreSQL
                          │
                          ▼
                      Power BI
```

## Streaming Architecture

```text
Simulated Transaction
        │
        ▼
Kafka Producer
        │
        ▼
Kafka Topic: transactions
        │
        ▼
Kafka Consumer
        │
        ▼
Transformation & Validation
        │
        ▼
PostgreSQL
        │
        ▼
Power BI
```

Batch and streaming processing are treated as complementary pipelines. Airflow orchestrates scheduled ETL workflows, while Kafka demonstrates event-based transaction ingestion.

---

# 3. Technology Stack

| Area                   | Technology               |
| ---------------------- | ------------------------ |
| Programming            | Python                   |
| Data Processing        | Pandas, NumPy            |
| Machine Learning       | Scikit-learn             |
| Text Representation    | TF-IDF character n-grams |
| ML Model               | Logistic Regression      |
| Data Storage           | PostgreSQL               |
| Object Storage         | AWS S3                   |
| Workflow Orchestration | Apache Airflow           |
| Streaming              | Apache Kafka             |
| Containers             | Docker                   |
| Analytics              | Microsoft Power BI       |
| Data Format            | CSV / Parquet            |
| Version Control        | Git / GitHub             |

---

# 4. Data Engineering Pipeline

## 4.1 Data Ingestion

The project uses a simulated banking transaction dataset as its source.

The raw transaction data contains fields such as transaction descriptions, amounts, transaction types, category IDs, and voucher IDs.

The raw dataset is stored locally during development and uploaded to:

```text
AWS S3
└── fintech-transactions-storage
    ├── raw/
    │   └── hk_transactions_table.csv
    └── processed/
        └── transactions_processed.csv
```

The original source data is preserved separately from processed data so that transformations remain traceable.

---

## 4.2 Data Cleaning and Validation

The ETL pipeline performs data-quality processing before loading transactions into PostgreSQL.

This includes:

* Standardizing transaction fields
* Cleaning descriptions
* Handling missing values
* Validating required columns
* Checking transaction identifiers
* Creating data-quality indicators
* Detecting invalid or corrupted category values
* Preventing duplicate transaction insertion

The production ETL successfully processed:

```text
Source records:        146,375
Transformed records:   146,375
Database records:      146,375
Record difference:     0

ETL STATUS: PASS
```

---

# 5. Canonical Business Taxonomy

One of the most important findings of the project was that the original `category_id` field did not consistently represent a stable business taxonomy.

For example, the description:

```text
petrol
```

appeared 1,025 times but was distributed across 55 different source category IDs.

The most common source category represented only approximately 28% of the occurrences.

This meant that simply training a model to reproduce the original `category_id` values would reproduce inconsistencies in the source data.

To address this, a source-independent canonical taxonomy was introduced.

## Canonical Categories

|   ID | Business Category       | Examples                              |
| ---: | ----------------------- | ------------------------------------- |
| 2001 | Food & Groceries        | groceries, dining, bakery, dairy      |
| 2002 | Fuel                    | petrol, diesel, CNG                   |
| 2003 | Transport               | ride-sharing, parking, tolls          |
| 2004 | Utilities & Bills       | electricity, water, internet, telecom |
| 2005 | Healthcare              | medicine, doctor, hospital            |
| 2006 | Shopping & Retail       | clothing, shopping, food chains       |
| 2007 | Transfers & Withdrawals | ATM, cash withdrawal, transfers       |
| 2008 | Income                  | salary, interest, deposits            |
| 2009 | Other                   | miscellaneous / fallback              |

The original `category_id` is retained for auditability rather than being deleted or globally merged.

---

# 6. Human-Approved Categorization

Canonical categories are supported by approved mappings and deterministic business rules.

The mapping workflow follows:

```text
Machine Discovery
       ↓
Candidate Mapping
       ↓
Validation
       ↓
Human Approval
       ↓
Approved Mapping
       ↓
Deterministic Application
```

This prevents semantic similarity alone from automatically assigning potentially incorrect financial categories.

For example, multilingual similarity may identify descriptions that appear related, but the mapping is not automatically accepted unless sufficient evidence exists.

Approved mappings are stored separately from the original transaction data.

---

# 7. Hybrid Transaction Categorization

The categorization system combines multiple sources of evidence.

```text
Transaction Description
        │
        ▼
Normalization
        │
        ├── Historical Evidence
        │
        ├── Semantic Similarity
        │
        ├── Category / Mapping Evidence
        │
        ├── Contextual Evidence
        │
        └── ML Prediction
                │
                ▼
        Confidence Combination
                │
        ┌───────┼────────┐
        ▼       ▼        ▼
       AUTO    REVIEW   UNKNOWN
```

## Categorization Components

### 1. Normalization

Transaction descriptions are standardized by:

* Converting text to lowercase
* Removing unnecessary punctuation
* Handling digits
* Collapsing whitespace
* Producing a normalized description for matching

### 2. Approved Mappings

Known descriptions can be assigned directly to an approved canonical category.

For example:

```text
petrol → Fuel
milk   → Food & Groceries
atm    → Transfers & Withdrawals
```

### 3. Historical Lookup

The system checks previously observed descriptions and their historical category behavior.

Historical evidence is only considered reliable when sufficient consistency exists.

### 4. Semantic Similarity

The production semantic layer uses:

```text
TF-IDF
+
Character n-grams
+
Cosine similarity
```

Character n-grams help identify variations such as:

```text
petrol
petrol 500
Bolan petrol
```

### 5. Cluster Evidence

Existing HDBSCAN cluster information is used as supporting evidence rather than as an independent categorizer.

Only sufficiently coherent clusters are considered reliable.

### 6. Contextual Evidence

Transaction type can provide additional information for specific descriptions.

For example, `"cash"` can have different meanings depending on transaction type.

Contextual rules are therefore applied selectively rather than assuming transaction type is universally useful.

### 7. Machine Learning

A TF-IDF + Logistic Regression classifier is used to predict transaction categories when sufficient training evidence exists.

---

# 8. Confidence-Based Routing

The system does not force a category for every transaction.

Predictions are routed according to confidence:

|   Confidence | Decision          |
| -----------: | ----------------- |
|       ≥ 0.90 | `AUTO_CATEGORIZE` |
| 0.70 – <0.90 | `REVIEW`          |
|        <0.70 | `UNKNOWN`         |

This conservative approach is intentional.

For financial data, an incorrect category can be more harmful than leaving a transaction temporarily uncategorized.

---

# 9. Machine Learning Model

The final semantic classification approach uses:

```text
Transaction Description
        ↓
TF-IDF Character N-grams
        ↓
Logistic Regression
        ↓
Category Prediction
        ↓
Confidence Score
```

## Training

The model uses:

* TF-IDF character n-grams
* n-gram range: 3–5 characters
* sublinear TF scaling
* Logistic Regression
* balanced class weights
* maximum iterations: 2000
* stratified train/test split

The trained model is stored under:

```text
models/
└── ml_model/
    ├── transaction_category_classifier.joblib
    ├── classification_report.txt
    ├── confusion_matrix.csv
    └── threshold_evaluation.csv
```

---

# 10. ML Experimentation and Model Freeze

Several approaches were evaluated during development.

A multilingual SentenceTransformer approach was tested using:

```text
paraphrase-multilingual-MiniLM-L12-v2
```

However, under the evaluated held-out setup, it performed worse than the TF-IDF baseline.

### Baseline

```text
AUTO:       2,281
Accuracy:   91.32%

REVIEW:     1,979
Accuracy:   71.60%

UNKNOWN:    12,986

Overall:    40.70%
```

The canonical taxonomy evaluation subsequently produced:

```text
AUTO:       3,757
Accuracy:   95.48%

REVIEW:     1,505
Accuracy:   70.10%

UNKNOWN:    11,984

Overall:    45.41%
```

This represents:

```text
AUTO volume:       +1,476
AUTO accuracy:     +4.16 percentage points
UNKNOWN:           -1,002
Overall accuracy:  +4.71 percentage points
```

The results showed that improving the underlying label taxonomy produced a more meaningful improvement than simply increasing model complexity.

TF-IDF was therefore retained as the production semantic approach for the current project freeze.

---

# 11. UNKNOWN Investigation

The system deliberately leaves low-confidence transactions as `UNKNOWN`.

A separate investigation was performed to determine whether UNKNOWN transactions could safely be recovered.

Important findings included:

* 13,305 UNKNOWN transactions in the investigated held-out evaluation
* 9,088 descriptions appeared only once
* 69 repeated descriptions accounted for 1,860 UNKNOWN rows
* 137 high-frequency, low-purity descriptions accounted for 2,246 UNKNOWN rows
* Historical purity alone did not provide sufficient accuracy for automatic categorization
* Semantic similarity alone was also insufficient
* Tested recovery strategies generally reached approximately REVIEW-level accuracy rather than AUTO-level accuracy

For example:

```text
Historical purity ≥ 0.90
support ≥ 1

Rows recovered: 1,537
Accuracy:       67.4%
```

This was not considered safe for automatic categorization.

The investigation therefore supported the decision to **keep the 0.90 AUTO threshold and avoid forcing uncertain transactions into categories**.

---

# 12. Incremental ETL

The project also implements incremental processing.

Instead of reloading every transaction, the incremental pipeline identifies new records using the transaction's `voucher_id`.

```text
Raw Transactions
       ↓
Read Existing Voucher IDs
       ↓
Compare Incoming Records
       ↓
Identify New Transactions
       ↓
Transform
       ↓
Validate
       ↓
Insert New Records
       ↓
PostgreSQL
```

PostgreSQL uses conflict handling to prevent duplicate insertion.

This makes the pipeline suitable for repeated execution as new transaction data becomes available.

---

# 13. Kafka Streaming Pipeline

Kafka was implemented as a streaming extension to demonstrate event-based transaction processing.

The flow is:

```text
Transaction
    ↓
Kafka Producer
    ↓
transactions Topic
    ↓
Kafka Consumer
    ↓
Transformation
    ↓
PostgreSQL
```

The PostgreSQL consumer uses idempotent insertion logic:

```text
ON CONFLICT (voucher_id) DO NOTHING
```

This ensures that receiving an already-processed transaction does not create a duplicate database record.

---

# 14. Apache Airflow

Apache Airflow is used to automate and schedule the data workflows.

The project contains DAGs for:

```text
airflow/
└── dags/
    ├── fintech_pipeline_dag.py
    ├── fintech_model_training_dag.py
    └── fintech_incremental_dag.py
```

### Main workflows

#### Main Pipeline DAG

Handles the scheduled transaction processing workflow.

#### Model Training DAG

Automates the model-training workflow.

#### Incremental DAG

Processes newly available transactions without reprocessing the entire dataset.

Airflow runs inside Docker and communicates with the project's data-processing components and PostgreSQL environment.

---

# 15. PostgreSQL

PostgreSQL serves as the structured database layer for processed financial transactions.

The production pipeline loads transformed transactions into:

```text
production_transactions
```

The database provides:

* Structured storage
* Primary transaction identifiers
* Duplicate prevention
* Incremental updates
* SQL-based analytics
* A reliable source for Power BI

Analytics views are also created to make downstream reporting easier.

---

# 16. Power BI

Power BI connects to the PostgreSQL database rather than directly to Airflow.

The architecture is:

```text
Airflow / ETL
      ↓
PostgreSQL
      ↓
Power BI
```

This keeps the reporting layer separate from the orchestration layer.

The Power BI dashboard provides a business-facing view of the processed transaction data and supports financial/cashflow analysis.

The dashboard file is maintained separately from the Python pipeline.

---

# 17. Data Quality and Reliability

The pipeline includes several reliability mechanisms:

* Schema validation
* Required-field validation
* Duplicate detection
* Voucher ID validation
* Data-quality flags
* PostgreSQL conflict handling
* Incremental processing
* ETL record-count verification
* Conservative ML confidence thresholds
* Human-approved canonical mappings

The goal is not simply to process transactions, but to make the resulting dataset reliable enough for downstream analytics.

---

# 18. Repository Structure

```text
hybrid_categorization/
│
├── airflow/
│   ├── dags/
│   │   ├── fintech_incremental_dag.py
│   │   ├── fintech_model_training_dag.py
│   │   └── fintech_pipeline_dag.py
│   └── docker-compose.yml
│
├── data/
│   └── raw/
│       └── hk_transactions_table.csv
│
├── models/
│   └── ml_model/
│       ├── classification_report.txt
│       ├── confusion_matrix.csv
│       ├── threshold_evaluation.csv
│       └── transaction_category_classifier.joblib
│
├── outputs/
│   ├── approved_canonical_mappings.csv
│   ├── canonical_taxonomy_review.csv
│   ├── canonical_taxonomy_review_summary.csv
│   ├── category_id_canonical_analysis.csv
│   ├── category_id_dominant_mapping.csv
│   ├── ml_training_dataset.csv
│   ├── ml_training_dataset_summary.csv
│   └── shopping_candidate_validation.csv
│
├── pipeline/
│   │   ├── 01_build_taxonomy.py
│   │   ├── 02_apply_approved_mappings.py
│   │   ├── 03_build_ml_training_dataset.py
│   │   ├── 04_train_ml_classifier.py
│   │   └── 06_run_evaluation.py
│   │   ├── 07_create_postgres_schema.py
│   │   ├── 09_validate_postgres_data.py
│   │   ├── 11_etl_production_pipeline.py
│   │   ├── 12_create_analytics_views.py
│   │   └── 14_incremental_etl.py
│   │   ├── 16_kafka_consumer.py
│   │   └── 17_kafka_to_postgres.py
│   │   ├── 13_add_test_transactions.py
│   │   └── 15_add_incremental_test.py
│       ├── 05_apply_ml_to_unresolved.py
│       ├── 10_inspect_production_data.py
│       ├── 18_diagnose_unresolved.py
│       ├── 19_analyze_unresolved_metadata.py
│       ├── 20_analyze_category_id_mapping.py
│       └── 21_audit_rule_mappings.py
│
├── src/
│   ├── canonical_taxonomy.py
│   ├── category_mapping.py
│   ├── cluster_evidence.py
│   ├── engine.py
│   ├── historical_lookup.py
│   ├── normalize.py
│   ├── similarity.py
│   └── __init__.py
│
├── taxonomy/
│   ├── category_mapping_approved.csv
│   └── contextual_rules_approved.csv
│
├── dashboard.pbix
├── requirements.txt
└── README.md
```

Large datasets, credentials, virtual environments, logs, caches, and development archives should be excluded from version control.

---

# 19. Running the Project

## Install Python dependencies

```bash
pip install -r requirements.txt
```

## Train the categorization model

```bash
python pipeline/training/01_build_taxonomy.py
python pipeline/training/02_apply_approved_mappings.py
python pipeline/training/03_build_ml_training_dataset.py
python pipeline/training/04_train_ml_classifier.py
```

## Run evaluation

```bash
python pipeline/training/06_run_evaluation.py
```

## Create PostgreSQL schema

```bash
python pipeline/batch/07_create_postgres_schema.py
```

## Run production ETL

```bash
python pipeline/batch/11_etl_production_pipeline.py
```

## Validate PostgreSQL data

```bash
python pipeline/batch/09_validate_postgres_data.py
```

## Create analytics views

```bash
python pipeline/batch/12_create_analytics_views.py
```

## Run incremental ETL

```bash
python pipeline/batch/14_incremental_etl.py
```

Kafka and Airflow workflows are run through the Docker environment provided in:

```text
airflow/docker-compose.yml
```

---

# 20. Configuration and Security

Credentials are not included in the repository.

Database and AWS configuration should be provided through environment variables.

Example:

```text
DB_HOST=localhost
DB_PORT=5432
DB_NAME=fintech_transactions
DB_USER=postgres
DB_PASSWORD=<your_password>

AWS_ACCESS_KEY_ID=<your_key>
AWS_SECRET_ACCESS_KEY=<your_secret>
AWS_DEFAULT_REGION=<your_region>
```

Actual credentials, `.env` files, virtual environments, Airflow logs, and other secrets must be excluded using `.gitignore`.

---

# 21. Key Results

The project demonstrates improvements at both the data-engineering and categorization levels.

### ETL

```text
Source records:        146,375
Transformed records:   146,375
Database records:      146,375
Difference:                  0
Status:                    PASS
```

### Categorization

```text
                       Baseline     Canonical
AUTO volume               2,281        3,757
AUTO accuracy             91.32%       95.48%
REVIEW volume             1,979        1,505
UNKNOWN volume           12,986       11,984
Overall accuracy          40.70%       45.41%
```

The canonical taxonomy resulted in:

* 1,476 additional AUTO predictions
* 4.16 percentage-point improvement in AUTO accuracy
* 1,002 fewer UNKNOWN transactions
* 4.71 percentage-point improvement in overall evaluated accuracy

---

# 22. Key Findings

### Finding 1 — Source labels were inconsistent

The original `category_id` values did not behave like a stable business taxonomy.

Repeated descriptions could appear under many different category IDs.

### Finding 2 — Better labels produced more value than more complex models

The canonical taxonomy improved categorization performance more meaningfully than the tested multilingual embedding approach.

### Finding 3 — Confidence matters

The system deliberately refuses to make low-confidence predictions instead of forcing every transaction into a category.

### Finding 4 — Historical evidence is useful only when supported by sufficient data

A single historical occurrence can produce apparently perfect purity while providing almost no statistical evidence.

### Finding 5 — Transaction type is useful selectively

Some descriptions become significantly clearer when combined with transaction type, while others remain ambiguous because their source labels are inherently inconsistent.

### Finding 6 — Canonical categories preserve auditability

The original source `category_id` remains available while business-level canonical categories provide a cleaner analytical representation.

---

# 23. Limitations

The current project has several limitations:

* The banking source is simulated rather than a live bank API.
* The transaction dataset is historical/static for the demonstration environment.
* The original source labels contain substantial inconsistency.
* Some transactions remain UNKNOWN by design.
* Canonical mappings require human validation when new transaction patterns appear.
* TF-IDF primarily captures textual similarity and may miss semantic synonyms with no shared character patterns.
* Kafka and Airflow demonstrate production-style architecture but are implemented as a project environment rather than a deployed banking production system.
* Power BI analytics depend on the PostgreSQL data available to the project.

---

# 24. Future Improvements

Potential future enhancements include:

* Integration with a real banking transaction API
* Automated data ingestion from live sources
* Expansion of the canonical taxonomy
* Monitoring for category and data drift
* Improved multilingual semantic representations
* More advanced merchant/entity normalization
* Automated data-quality monitoring
* Cloud-native deployment
* Role-based access controls
* Real-time Power BI or streaming analytics
* Automated human-review interfaces for uncertain classifications

---

# 25. Project Conclusion

This project demonstrates an end-to-end FinTech data engineering workflow rather than a standalone machine learning model.

The final system combines:

```text
Data Ingestion
      +
Cloud Storage
      +
ETL & Data Quality
      +
Canonical Standardization
      +
Hybrid ML Categorization
      +
Incremental Processing
      +
Kafka Streaming
      +
Airflow Orchestration
      +
PostgreSQL
      +
Power BI
```

The most important lesson from the project was that improving a financial ML system is not always a matter of using a more complex model. In this dataset, inconsistent source labels were a major limitation. Introducing a canonical business taxonomy and applying human-approved mappings provided a more transparent and effective solution.

The resulting system prioritizes **data quality, explainability, conservative predictions, and reliable downstream analytics** over simply maximizing the number of automatically categorized transactions.
