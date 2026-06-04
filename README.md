# Adaptive Semi-Structured Dataset Reconstruction Engine

## Overview
The Adaptive Semi-Structured Dataset Reconstruction Engine is an enterprise-ready, Python-based framework designed to automatically reconstruct fragmented, corrupted, and semi-structured datasets into optimized tabular formats. 

Equipped with a native dual-output architecture, the platform simultaneously delivers human-readable datasets tailored for Business Intelligence (BI/Analysts) alongside clean, target-aware mathematical matrices optimized for downstream Machine Learning (ML) pipelines. The engine features a premium web application canvas built with Streamlit for dynamic profiling, real-time threshold configuration, and direct asset downloads.

---

## Problem Statement

Traditional rule-based parsers fail when datasets contain structural fragmentation, severe noise, or multi-collinearity:
- Multi-row scattered headers or inconsistent row boundaries
- Shifted columns and sparse/null regions caused by corrupt software exports
- High-cardinality data noise (e.g., random row hashes, customer identifiers) that overfit ML architectures
- Severe multi-collinearity where independent features perfectly overlap, masking the true signal
- OCR-like corrupted structures and mixed-type record fragmentations

This engine establishes an autonomous pipeline to reconstruct, profile, cleanse, normalize, encode, and structurally prune these records with zero manual pipeline scripting.

---

## Features

- **Automatic Dataset Profiling & Cleaning:** Automatically dynamicizes data types and handles missing values based on statistical distributions.
- **Premium User Interface:** Clean, spacious drag-and-drop file uploader supporting CSV and Excel formats with an interactive, tooltip-supported configuration profile sidebar.
- **In-Memory Cache Guard (`st.session_state`):** Leverages a composite configuration `cache_hash` to register dataset properties and parameters, allowing users to toggle visual layouts instantly without re-triggering heavy background computations.
- **Universal Noise Filtering:** Scans all columns globally for high-cardinality structural noise (like tracking IDs or numerical customer IDs) and strips them from the ML stream based on a dynamic uniqueness ratio.
- **Target-Aware Collinear Pruning:** Identifies feature-to-feature overlaps. If two independent columns tell the same story, it calculates their absolute predictive relationship with the target feature, preserving the one with the stronger target correlation and pruning the redundant one.
- **Multi-Distribution Imputation:** Detects column lopsidedness via skewness metrics. Highly skewed columns automatically use robust median tracking to preserve structural integrity against extreme outliers.
- **Confidence-Based Dual Outputs:** Toggles seamlessly between a Human-Readable Cleaned view (optimized for BI analytics with zero columns dropped) and a Machine Learning Ready matrix (fully scaled, encoded, and pruned).

---

## Reconstruction Strategies

### Segmented Reconstruction
Handles datasets where logical records span multiple physical rows.

### Positional Reconstruction
Corrects shifted and sparse column structures.

### Streaming Reconstruction
Processes OCR-like fragmented datasets.

### Hybrid Reconstruction
Combines multiple adaptive recovery techniques dynamically.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Streamlit
- Regular Expressions (Regex)
- Object-Oriented Programming (OOP)

---

## Project Structure

```bash
adaptive-dataset-reconstruction-engine/
│
├── src/
│   ├── __init__.py
│   └── auto_prep.py            # Primary Backend Engine Logic Core
├── app.py                      # Premium Streamlit UI Canvas & View Router
├── requirements.txt            # System Dependency Tracking Manifest
└── README.md                   # System Documentation Profile

```
---

## Instalation

```Bash
# Clone the workspace repository
git clone [https://github.com/shyamganeesh/adaptive-dataset-reconstruction-engine.git](https://github.com/shyamganeesh/adaptive-dataset-reconstruction-engine.git)
cd adaptive-dataset-reconstruction-engine

# Install required libraries
pip install -r requirements.txt

```
---

## Usage

Local Network Deployment
To launch the application canvas in a development or internal local area network environment, execute:
```Bash
streamlit run app.py
```
The engine will host your application locally at http://localhost:8501 and provide your LAN Network IP.

---

## Example Use Cases
- Broken Excel exports
- Corrupted CSV files
- OCR-generated tables
- Sparse enterprise reports
- Multi-row transactional datasets
- Semi-structured business reports

---

## Configuration Guide for Users
- Target Column Name: Define your dependent target column (e.g., selling_price, is_churn). The engine uses this column as the center of mass for all downstream ML importance assessments.
- Skewness Threshold (Default 0.5): Adjusts sensitivity to extreme outliers. Lower boundaries force RobustScaler usage to neutralize lopsided distributions.
- Feature-to-Feature Correlation Limit (Default 0.85): The pruning wire threshold. If two independent variables overlap above this value, the element with the weaker predictive correlation to the target is systematically eliminated.
- Noise Threshold (Default 0.80): Checks column cardinality. If a feature contains more unique data records than this percentage parameter relative to the total rows, it is automatically discarded from the ML stream.

---

## Author
Shyam Ganeesh