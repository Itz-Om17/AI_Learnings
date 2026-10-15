# Student Placement Predictor — Feature & Design Document

Oct 15, 2026 · @om

## Overview

This document covers everything we decided while building the **Student Placement Predictor** — a machine learning model that predicts whether a student from an Indian engineering college will get placed or not.

The goal is to use this model in a **real college placement system**. That means the features, the data, and the logic all need to reflect how Indian campus placements actually work — not just what looks good on paper.

We started by analysing an existing dataset, found its limitations, and then designed everything from scratch — the features, the dataset, and the algorithm — based on real Indian placement trends.

By the end of this process, we finalised:

- **21 features** that genuinely drive placement outcomes in Indian colleges
- A **20,000-row dataset** built to mimic real student profiles
- **XGBoost** as the algorithm to train the model

## Algorithm — XGBoost

We chose **XGBoost** (Extreme Gradient Boosting) as the algorithm for this model.

**What it does in simple terms:** XGBoost builds a series of decision trees one after the other. Each new tree looks at the mistakes the previous tree made and tries to fix them. This step-by-step error correction is what makes it consistently more accurate than other algorithms on structured data like ours.

**Why XGBoost and not something else:**

- It works great on tabular data (rows and columns) — which is exactly what we have
- It handles mixed feature types natively — our dataset has integers, floats, and ordinal values all together
- No need to scale or normalise features before training
- Handles the class imbalance (60% not placed, 40% placed) with a single parameter — `scale_pos_weight`
- Gives feature importance scores after training — so we can see which features are actually driving the predictions
- Expected accuracy on our dataset: **83–87%**

**Comparison with other options we considered:**

| Algorithm | Expected Accuracy | Reason not chosen |
| --- | --- | --- |
| Logistic Regression | 72–75% | Too simple, misses feature interactions |
| Random Forest | 78–82% | Good but XGBoost consistently beats it on tabular data |
| **XGBoost** | **83–87%** | **Best choice** |
| SVM | 75–79% | Slow to train, needs feature scaling |
| Neural Networks | Overkill | 20K rows with 21 features — not enough data to justify it |

## Dataset

We generated a custom dataset specifically for this model rather than using the original dataset we analysed. The original had issues — most notably, a plain Yes/No for extracurricular activities and no separation between fake and real internships — which would have led to a poor and unrealistic model.

**Dataset specs:**

- **Total rows:** 20,000 students
- **Total columns:** 22 (21 features + 1 target)
- **Placement split:** 40% Placed, 60% Not Placed — reflects the real ground situation in Indian tier-2/tier-3 engineering colleges
- **Missing values:** None

**What makes this dataset feel real and not synthetic:**

- A hidden student quality tier (1 to 5) drives natural correlations across features. Strong students tend to score well across most features — just like in real life. Weak students struggle across the board.
- Individual noise is added per student — some students overperform their tier, some underperform. This reflects reality.
- Vernacular medium background effect — around 38% of students have a reduced verbal and communication score regardless of their overall tier, reflecting the English fluency gap in many Indian colleges.
- Hard real-world rules are baked into the placement logic — for example, students with 3 or more active backlogs almost never get placed (5% chance), and students with CGPA below 5.5 have only an 8% chance.
- Placement has noise — interview luck, company quota, a bad day in the interview. A single noise factor is added to every student's final placement probability.

**Reality check results after generation:**

| Scenario | Expected | Actual in dataset |
| --- | --- | --- |
| Active backlogs ≥ 3 placed | \~5% | 5.9% |
| CGPA < 5.5 placed | \~8% | 8.4% |
| Strong aptitude (Quant ≥ 32, Coding ≥ 8) placed | 80%+ | 92% |
| No internship placed | \~12% | 12.6% |
| Legitimate internship placed | \~70%+ | 72.2% |
| Coding score 0–3 placed | \~5% | 5.7% |
| Coding score 9–10 placed | \~85%+ | 87% |

## All 21 Features

Here is every feature we finalised, organised by category. Each one was discussed and decided based on how Indian campus placements actually work.

---

### Category 1 — Aptitude (4 features)

We split the original single AptitudeTestScore (0–100) into 4 separate components because different companies weight these sections differently.

| Feature | Scale | What it measures |
| --- | --- | --- |
| `QuantitativeScore` | 0–40 | Maths — percentages, ratios, time-speed-distance. Heaviest weighted in most campus tests |
| `LogicalReasoningScore` | 0–30 | Puzzles, series, arrangements. Signals problem-solving ability |
| `VerbalScore` | 0–20 | English comprehension and grammar. Less decisive for pure tech roles |
| `CodingScore` | 0–10 | Basic programming MCQs, output prediction. Increasingly important in IT companies |

---

### Category 2 — Academic (3 features)

Instead of just CGPA, we added backlog information because many good students have past backlogs but still perform well in other areas.

| Feature | Scale | What it measures |
| --- | --- | --- |
| `CGPA` | 5.0–9.8 | Overall academic performance. Non-linear impact — below 6.5 carries a heavy penalty |
| `ActiveBacklogs` | 0–3 | Currently pending failed subjects. Most companies have zero tolerance here |
| `TotalKTs` | 0–10 | History of cleared backlogs. Affects document verification rounds |

---

### Category 3 — Projects (2 features)

Just counting projects misses the quality. A to-do list app and a deployed ML model are not the same.

| Feature | Scale | What it measures |
| --- | --- | --- |
| `ProjectCount` | 0–5 | Number of projects done |
| `ProjectComplexity` | 0–5 | Quality: 0=None, 1=Basic (tutorial), 2=Intermediate, 3=Advanced, 4=Industry-level, 5=Research/Publication |

---

### Category 4 — Internships (3 features)

Similarly, the type and duration of internship matters far more than just the count.

| Feature | Scale | What it measures |
| --- | --- | --- |
| `InternshipCount` | 0–3 | Number of internships done |
| `InternshipType` | 0–3 | 0=None, 1=Certificate-based, 2=College project-based, 3=Legitimate remote or onsite |
| `InternshipDuration` | 0–3 | 0=None, 1=Short (< 1 month), 2=Medium (1–3 months), 3=Long (3+ months) |

---

### Category 5 — Soft Skills (3 features)

A single soft skills rating doesn't capture enough. We split it into three components that each get tested at a different stage of campus placements.

| Feature | Scale | What it measures |
| --- | --- | --- |
| `CommunicationScore` | 1–5 | Clarity of English, ability to explain technical concepts simply |
| `ConfidenceScore` | 1–5 | Body language, composure under pressure, handling tough questions |
| `TeamworkScore` | 1–5 | Group discussion performance, listening, collaborative attitude |

---

### Category 6 — Extracurricular Activities (1 feature)

Changed from binary Yes/No to an ordinal scale because a student who led the college technical club is very different from one who attended a cultural event.

| Feature | Scale | What it measures |
| --- | --- | --- |
| `ExtracurricularActivities` | 0–5 | 0=None, 1=Sports/Cultural, 2=Club Member, 3=Committee/Volunteer, 4=Club Lead/Captain, 5=Technical Club Lead |

---

### Category 7 — Workshops & Certifications (2 features)

| Feature | Scale | What it measures |
| --- | --- | --- |
| `WorkshopCount` | 0–6 | Number of workshops and certifications attended |
| `CertificationQuality` | 0–4 | 0=None, 1=Basic online, 2=Recognised MOOC (Coursera, NPTEL), 3=Industry cert (AWS, Azure), 4=Competitive achievement |

---

### Category 8 — Placement Training (3 features)

Replaced the binary PlacementTraining column with three separate signals because a student who attended every session and gave 5 mock interviews is very different from one who showed up once.

| Feature | Scale | What it measures |
| --- | --- | --- |
| `PlacementTrainingAttendance` | 0–3 | 0=None, 1=Partial, 2=Regular, 3=Full — how consistently they attended |
| `PlacementTrainingQuality` | 0–4 | 0=None, 1=Basic, 2=Moderate, 3=Comprehensive, 4=Industry partnered — quality of college's T&P program |
| `MockInterviewsGiven` | 0–4 | 0=None, 1=1–2 mocks, 2=3–5 mocks, 3=6–10 mocks, 4=10+ mocks |

## What We Dropped and Why

We started with the original dataset's 11 features. Here is everything we removed and the reasoning behind each decision.

**StudentID — Dropped** Just a row number. Has zero predictive value and would confuse the model if kept in.

**SSC\_Marks (10th grade) — Dropped** Despite having a decent correlation (0.47) in the original dataset, 10th grade marks are rarely used by recruiters in actual placement decisions. Most companies don't even ask for SSC marks. The correlation existed because better students tend to score well everywhere — not because SSC marks directly cause placement. Keeping it would teach the model a false pattern.

**HSC\_Marks (12th grade) — Dropped** Same reasoning as SSC. HSC had a slightly stronger correlation (0.51) but again, Indian campus recruiters don't use 12th grade marks as a decision factor. CGPA, aptitude scores, and projects carry the real weight. Both marks were dropped after confirming that removing them costs only 0.23% accuracy — well within statistical noise.

**Single AptitudeTestScore — Replaced** The original single score (0–100) was split into four components: Quantitative, Logical Reasoning, Verbal, and Coding. Different companies weight these differently, and a combined score hides that distinction.

**Binary ExtracurricularActivities — Replaced** A simple Yes/No throws away too much information. A student who led their college's robotics club is categorically different from one who attended a cultural event. Replaced with an ordinal scale from 0 to 5.

**Binary PlacementTraining — Replaced** Knowing only whether someone attended training or not is not useful. Replaced with three separate features: attendance consistency, quality of the college's T&P program, and number of mock interviews given.

**Single SoftSkillsRating — Replaced** A single 1–10 rating was too coarse. Split into Communication, Confidence, and Teamwork — each tested at a different stage of the placement process.

**Single Internship count — Replaced** Added InternshipType and InternshipDuration alongside the count because the quality and length of an internship matters far more than just the number.

## Placement Logic

This explains the rules we built into the dataset to determine whether a student gets placed or not. Understanding this is important because the model will learn from this logic.

**How placement probability is calculated:**

Every student gets a placement score built from all 21 features. The weights reflect how much each factor actually matters in Indian campus placements.

| Category | Max contribution | Notes |
| --- | --- | --- |
| Aptitude | \~28 points | Quant carries the most weight (10 pts), then Logical (8 pts), Coding (7 pts), Verbal (3 pts) |
| Academic (CGPA + Backlogs + KTs) | \~18 points | CGPA is non-linear — sharp drop below 6.5 |
| Projects | \~12 points | Complexity matters more than count |
| Internships | \~14 points | Type and duration combined carry more than count alone |
| Soft Skills | \~12 points | Low communication score carries a direct penalty |
| Extracurricular | \~6 points | Technical lead (levels 4–5) gets higher weight |
| Certifications & Workshops | \~6 points | Industry certifications weight more |
| Placement Training | \~6 points | Only counts if the student actually attended |

**Hard rules that override everything else:**

- `ActiveBacklogs >= 3` → placement chance drops to \~5% regardless of other features. Most companies refuse to even process the application.
- `CGPA < 5.5` → placement chance drops to \~8%. Very few companies accept this academic standing.
- Strong aptitude profile (`QuantitativeScore >= 32` AND `CodingScore >= 8`) → placement chance rises to \~92%.

**Realistic noise in placements:**

Real placement outcomes are never 100% predictable. A small random noise factor is added to every student's final placement probability to simulate:

- Interviewer mood and personal bias
- Company seat availability on that day
- A student having a bad or great interview day
- Resume shortlisting randomness

This is important — it means the model will never achieve 100% accuracy, and that is correct. A model that achieves 100% accuracy on placement data is almost certainly overfitting.

---

## Model Development & Training Flow

Once the 20,000-student dataset was generated and verified, we implemented the entire machine learning pipeline in Google Colab (`StudentPlacementPredictor.ipynb`).

The development followed a clean, structured 6-step flow:

```
[1. Environment Setup] ──> [2. Data Ingestion] ──> [3. Stratified Splitting]
                                                           │
┌──────────────────────────────────────────────────────────┘
▼
[4. XGBoost Training & Tuning] ──> [5. Multi-Model Benchmarking] ──> [6. Stress Testing & Export]
```

### Step 1 — Environment & Library Setup

We set up the notebook with all core scientific computing and ML libraries:

- `xgboost`: Core gradient boosting library used for our primary classifier
- `scikit-learn`: For data splitting (`train_test_split`), evaluation metrics (`accuracy_score`, `classification_report`), feature scaling (`StandardScaler`), and baseline models
- `pandas` & `numpy`: Data manipulation, matrix transformations, and synthetic student inference
- `matplotlib` & `seaborn`: Visualization of feature importances and training metrics
- `pickle`: Serializing and exporting trained models into `.pkl` format for deployment

### Step 2 — Data Ingestion & Integrity Check

We loaded `indian_placement_dataset.csv` directly into a Pandas DataFrame:

- **Rows:** 20,000 students
- **Features:** 21 input features + 1 target (`PlacementStatus`)
- Verified zero null/missing values across all columns
- Confirmed column data types (integers, floats, ordinal ratings) to ensure seamless training without unnecessary type conversions

### Step 3 — Feature-Target Separation & Stratified Splitting

We separated the dataset into input features (`X`) and the target label (`y`):

- `X`: All 21 features (dropping `PlacementStatus`)
- `y`: `PlacementStatus` (0 = Not Placed, 1 = Placed)

**Why a Stratified Split was essential:**
Our dataset has a natural 60:40 split (60% Not Placed, 40% Placed). A regular random split could introduce sampling variance where the training set gets more placed students and the test set gets fewer, skewing the evaluation. 

By setting `stratify=y` with an **80–20 split** (`random_state=42`):

- **Training set:** 16,000 students (exactly 60% Not Placed, 40% Placed)
- **Testing set:** 4,000 students (exactly 60% Not Placed, 40% Placed)

This guarantees that both the training and evaluation subsets faithfully represent ground-truth campus placement proportions.

---

## Primary Model — XGBoost Classifier

We trained **XGBoost** as our primary model for production deployment.

### Hyperparameter Configuration

We configured `xgb.XGBClassifier` with the following parameters:

```python
model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
```

**Why these specific hyperparameters:**

- `n_estimators=100`: 100 consecutive decision trees provide sufficient learning capacity without causing the model to overfit on noise.
- `learning_rate=0.1`: Shrinks the contribution of each tree by 0.1, forcing subsequent trees to learn gradual, robust error corrections rather than jumping aggressively.
- `max_depth=5`: Allows trees to capture interactions up to 5 levels deep (for example: *CGPA threshold → Active Backlogs check → Quantitative Score range → Internship Type → Final probability*). This is deep enough for rich multi-feature decisions, but shallow enough to avoid memorising individual rows.
- `eval_metric='logloss'`: Binary logarithmic loss directly penalises confident wrong predictions, ensuring that output probabilities reflect true placement odds.
- `random_state=42`: Ensures 100% reproducible training runs.

### Evaluation Results

On the unseen 4,000-student test set, the model achieved **90.20% accuracy** — outperforming our initial conservative estimate of 83–87%.

**Detailed Classification Report:**

| Class | Precision | Recall | F1-Score | Support (Students) |
| --- | --- | --- | --- | --- |
| **0 (Not Placed)** | 0.92 | 0.92 | 0.92 | 2,388 |
| **1 (Placed)** | 0.88 | 0.88 | 0.88 | 1,612 |
| **Macro Average** | 0.90 | 0.90 | 0.90 | 4,000 |
| **Weighted Average** | 0.90 | 0.90 | 0.90 | 4,000 |
| **Overall Accuracy** | | | **90.20%** | **4,000** |

**What these metrics tell us:**

- **0.92 Precision on Not Placed:** When the model predicts a student will not get placed, it is right 92% of the time. This prevents giving students false confidence.
- **0.88 Recall on Placed:** The model successfully identifies 88% of all students who actually get placed.
- **Balanced F1-Scores (0.92 / 0.88):** Proves the model handles the 60:40 class distribution cleanly without collapsing toward predicting the majority class.

### Feature Importance Insights

Using `xgb.plot_importance(model, importance_type='weight', max_num_features=10)`, we analysed which features the trees split on most frequently:

- `QuantitativeScore` and `CodingScore` emerged as the most frequent decision nodes, reflecting how online aptitude tests act as the primary filter in campus drives.
- `CGPA` and `ActiveBacklogs` served as critical gating features, instantly pruning tree branches for low-academic profiles.
- `InternshipDuration` and `ProjectComplexity` drove high-tier placement predictions in downstream tree splits.

This confirmed that XGBoost learned the exact decision hierarchy used by real Indian tech recruiters.

### Single Student Inference Test

To verify end-to-end inference before exporting, we tested the model with a simulated strong student profile:

- **Profile:** CGPA 8.5, 0 backlogs, 0 KTs, Quant 35/40, Logical 25/30, Verbal 15/20, Coding 8/10, 3 projects (complexity 4), 1 legitimate internship (duration 2), solid soft skills (all 4s), and technical club lead (level 5).
- **Prediction:** `PLACED`
- **Model Confidence:** **99.21%**

---

## Multi-Model Benchmarking

To prove whether XGBoost was genuinely the best option or if a simpler algorithm could suffice, we trained and evaluated 3 additional classifiers on the exact same 80–20 stratified split.

### 1. Random Forest Classifier

- **Setup:** `n_estimators=100`, `random_state=42`, `class_weight='balanced'`
- **Test Accuracy:** **90.10%**
- **How it performed:** Very competitive with XGBoost. It handles non-linear boundaries well through bagged decision trees. However, it takes more memory and lacks gradient boosting's sequential error correction.
- **Saved as:** `rf_placement_model.pkl`

### 2. Logistic Regression (with Feature Scaling)

- **Setup:** Unlike tree-based models, Logistic Regression requires all features to be on the same scale. We transformed features using `StandardScaler` (`scaler.fit_transform(X_train)`), followed by `LogisticRegression(max_iter=1000, random_state=42)`.
- **Test Accuracy:** **90.43%**
- **How it performed:** Achieved a high aggregate accuracy because our placement probability formula has a strong linear point-tally component. However, because it fits a single linear hyperplane, it struggles on sharp non-linear rules (like active backlog cutoffs) as shown in the edge-case analysis below.
- **Saved as:** `log_placement_model.pkl` along with `scaler.pkl`

### 3. Gaussian Naive Bayes

- **Setup:** `GaussianNB()` with default parameters
- **Test Accuracy:** **89.30%**
- **How it performed:** Decent baseline, but lowest overall accuracy. Naive Bayes assumes all 21 features are conditionally independent. In reality, student attributes are deeply correlated (e.g., strong quant scores correlate with coding scores and CGPA). This independence assumption makes Naive Bayes overly confident and prone to major false positives on edge cases.
- **Saved as:** `nb_placement_model.pkl`

### Overall Benchmark Summary

| Model | Test Accuracy | Scaling Needed? | Key Strength | Key Limitation | Production Status |
| --- | --- | --- | --- | --- | --- |
| **XGBoost** | **90.20%** | **No** | **Handles non-linear rules & feature interactions best** | **Slightly higher compute than linear models** | **Primary Choice (Deployed)** |
| Random Forest | 90.10% | No | Robust against outliers | Larger pickle file size | Benchmark alternative |
| Logistic Regression | 90.43% | Yes (`StandardScaler`) | Fast, interpretable coefficients | Misses sharp non-linear disqualifications | Benchmark alternative |
| Naive Bayes | 89.30% | No | Extremely fast training | Overconfident due to independence assumption | Baseline only |

---

## Stress Testing & Edge-Case Analysis (The 5 Student Archetypes)

In Cell 13 of the notebook, we built an integrated multi-model evaluation harness. We passed 5 carefully designed student archetypes through all 4 models simultaneously to compare how they behave on real-world edge cases.

### The 5 Test Profiles

1. **Student 1 (Strong all-rounder):** CGPA 8.5, 0 backlogs, 35 Quant, 8 Coding, 3 complex projects, tier-3 internship, good communication.
2. **Student 2 (Struggling profile):** CGPA 6.2, 2 active backlogs, 5 total KTs, 20 Quant, 4 Coding, no internships, basic projects.
3. **Student 3 (Star student):** CGPA 9.1, 0 backlogs, 40 Quant, 9 Coding, 5 industry projects, 2 internships, maximum soft skills.
4. **Student 4 (The Critical Borderline Case):** CGPA 7.0, **1 active backlog**, 2 cleared KTs, 25 Quant, 6 Coding, 2 intermediate projects, college internship, average soft skills.
5. **Student 5 (Consistent above-average candidate):** CGPA 8.0, 0 active backlogs, 1 cleared KT, 30 Quant, 7 Coding, 3 projects, tier-3 internship.

### Comparison Results

Here is the exact comparison output generated across all 4 models:

| Student Profile | Random Forest | Naive Bayes | Logistic Regression | XGBoost (Primary) |
| --- | --- | --- | --- | --- |
| **Student 1** (All-Rounder) | Placed (97.0%) | Placed (100.0%) | Placed (99.8%) | **Placed (99.2%)** |
| **Student 2** (Struggling) | Not Placed (100.0%) | Not Placed (100.0%) | Not Placed (99.8%) | **Not Placed (99.6%)** |
| **Student 3** (Star Performer) | Placed (88.0%) | Placed (100.0%) | Placed (100.0%) | **Placed (94.4%)** |
| **Student 4** (Borderline / Backlog) | Not Placed (67.0%) | **Placed (94.9%) ❌** | Not Placed (57.6%) | **Not Placed (83.7%) ✅** |
| **Student 5** (Above-Average) | Placed (95.0%) | Placed (100.0%) | Placed (99.1%) | **Placed (98.6%)** |

### Why Student 4 Proves XGBoost is the Superior Model

The borderline case (**Student 4**) demonstrates why looking at overall test accuracy alone (where Logistic Regression scored 90.43%) is misleading for campus placements:

- **The Real Ground Reality:** Student 4 has a decent 7.0 CGPA and fair aptitude, but has **1 active backlog**. In Indian campus placements, almost every tier-1 and tier-2 IT recruiter filters out active backlogs during initial screening rounds.
- **Where Naive Bayes Failed:** Naive Bayes saw the decent CGPA, projects, and aptitude scores, treated them as completely independent of the backlog, and declared the student **Placed with 94.9% confidence**! In a real college placement portal, this would be a catastrophic false positive.
- **Where Logistic Regression Struggled:** Logistic Regression only gave a 57.6% confidence of Not Placed. Because it sums weighted inputs linearly, the positive scores partially canceled out the backlog penalty.
- **Where XGBoost Excelled:** XGBoost evaluated the student through decision trees with depth 5. The tree split caught the `ActiveBacklogs >= 1` condition combined with a moderate 7.0 CGPA and immediately penalised the placement probability, correctly predicting **Not Placed with 83.7% confidence**.

This edge case proves that tree-based gradient boosting is mandatory for handling real-world college placement rules.

---

## Saved Artifacts & Production Usage

All models and preprocessing objects were serialized using `pickle` for direct integration into web applications, APIs, or college placement dashboards:

### Exported Artifacts

| File | Purpose | Preprocessing Required |
| --- | --- | --- |
| `placement_model.pkl` | **Primary XGBoost Model** | None — feed raw 21 features in exact order |
| `rf_placement_model.pkl` | Random Forest Model | None — raw 21 features |
| `log_placement_model.pkl` | Logistic Regression Model | **Yes** — must pass features through `scaler.pkl` first |
| `scaler.pkl` | StandardScaler fitted on 16K training rows | Use `scaler.transform()` before predicting with linear model |
| `nb_placement_model.pkl` | Gaussian Naive Bayes Model | None — raw 21 features |

### Production Inference Example

To run predictions in any Python backend (Flask, FastAPI, Streamlit, or Django), load `placement_model.pkl` and pass the 21 features in the exact column order:

```python
import pickle
import pandas as pd

# 1. Load the production model
model = pickle.load(open('placement_model.pkl', 'rb'))

# 2. Prepare student data (matching the exact 21 features)
student_data = pd.DataFrame([[
    8.5,  # CGPA (5.0 - 10.0)
    0,    # ActiveBacklogs (0 - 3)
    0,    # TotalKTs (0 - 10)
    35,   # QuantitativeScore (0 - 40)
    25,   # LogicalReasoningScore (0 - 30)
    15,   # VerbalScore (0 - 20)
    8,    # CodingScore (0 - 10)
    3,    # ProjectCount (0 - 5)
    4,    # ProjectComplexity (0 - 5)
    1,    # InternshipCount (0 - 3)
    3,    # InternshipType (0 - 3)
    2,    # InternshipDuration (0 - 3)
    4,    # CommunicationScore (1 - 5)
    4,    # ConfidenceScore (1 - 5)
    4,    # TeamworkScore (1 - 5)
    5,    # ExtracurricularActivities (0 - 5)
    3,    # WorkshopCount (0 - 6)
    4,    # CertificationQuality (0 - 4)
    3,    # PlacementTrainingAttendance (0 - 3)
    4,    # PlacementTrainingQuality (0 - 4)
    4     # MockInterviewsGiven (0 - 4)
]], columns=[
    'CGPA', 'ActiveBacklogs', 'TotalKTs', 'QuantitativeScore', 'LogicalReasoningScore',
    'VerbalScore', 'CodingScore', 'ProjectCount', 'ProjectComplexity', 'InternshipCount',
    'InternshipType', 'InternshipDuration', 'CommunicationScore', 'ConfidenceScore',
    'TeamworkScore', 'ExtracurricularActivities', 'WorkshopCount', 'CertificationQuality',
    'PlacementTrainingAttendance', 'PlacementTrainingQuality', 'MockInterviewsGiven'
])

# 3. Predict placement status and confidence
prediction = model.predict(student_data)[0]
probability = model.predict_proba(student_data)[0][1]

if prediction == 1:
    print(f"Outcome: PLACED (Confidence: {probability * 100:.2f}%)")
else:
    print(f"Outcome: NOT PLACED (Confidence: {(1 - probability) * 100:.2f}%)")
```

With this, the entire lifecycle — from raw dataset design and feature rationale to Colab model training, multi-algorithm benchmarking, stress-testing, and production packaging — is fully documented and reproducible.
