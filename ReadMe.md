# Himalayan Climbing Expeditions — ML Interpretability Analysis

Predicting summit success in the Himalayas using Random Forest and SHAP, 
with a critical reading of what the model actually learns.

**SC398 — Statistical and Machine Learning Interpretability**  
Colby College · Co-authored with Anthony Jordan and Bhuwan Bhandari

---

## The core argument

The model flags `hired` — whether a climber is paid Nepali staff — as one 
of the strongest predictors of summit success. Hired workers show both higher 
success rates *and* higher death rates than foreign clients. Their "success" 
is getting someone else to the summit. The model learns a real pattern; what 
it cannot see is that the pattern encodes labor structure, not climbing ability.

This is the central interpretability problem the project investigates: a 
deterministic rules engine can achieve high accuracy while its most important 
feature is a proxy for colonial labor hierarchy rather than the outcome it 
claims to predict.

---

## Dataset

[HimEx — Himalayan Expeditions Database](https://www.himalayandatabase.com/)  
73,007 individual climber records · 1905–2019  
Three source files: `expeditions.csv`, `peaks.csv`, `members.csv`

**Features used**

| Feature | Description |
|---|---|
| `success` | Binary — did the climber summit? (target) |
| `hired` | Boolean — paid Nepali staff or foreign client |
| `age` | Climber age at time of expedition |
| `sex` | Climber gender |
| `season` | Spring / Autumn / Winter / Summer |
| `year` | Expedition year |
| `oxygen_used` | Supplemental oxygen used |
| `solo` | Solo attempt |
| `height_metres` | Peak elevation |

---

## What the code does

### 1. Data preparation
Merges member, expedition, and peak tables. Filters incomplete records. 
Encodes categoricals as factors.

### 2. Classification model — predicting summit success
Random Forest (500 trees, 80/20 train-test split, 5-fold stratified 
cross-validation). Class weights applied to handle imbalance between 
successful and failed summits.

### 3. Global interpretability — SHAP
SHAP values computed via `fastshap` with `nsim = 50` Monte Carlo samples. 
Outputs: beeswarm importance plot, bar importance plot, dependence plots 
for `height_metres × oxygen_used` and `age × hired`.

### 4. Uncertainty quantification — conformal prediction
Split-conformal approach. Calibration set used to compute non-conformity 
scores. Prediction sets generated at α = 0.10 (target coverage ≥ 90%). 
High-uncertainty cases (both classes in prediction set) flagged for local 
analysis.

### 5. Local interpretability — LIME
LIME explanations generated for three case types: most uncertain predictions, 
high-confidence successes, high-confidence failures. 5 features per 
explanation, 500 permutations.

### 6. Failure depth analysis — regression
Secondary model predicting *how far* failed climbers got (`percent_climbed = 
highpoint / peak height`). Same SHAP pipeline applied to the regression 
output. Investigates whether the same features that predict binary success 
also predict depth of failure.

---

## Key findings

- `hired` dominates feature importance in both classification and regression 
  models — higher than age, oxygen use, or peak height
- Hired Nepali staff succeed at higher rates but also die at higher rates, 
  because their role is summit support, not personal ascent
- Oxygen use interacts with peak height in a threshold pattern — above ~7,500m, 
  supplemental oxygen becomes a near-requirement for success
- Autumn and Spring show significantly higher success rates than Winter; 
  Summer is negligible volume
- Success rates trend upward over the 20th century, then plateau — likely 
  reflecting improved gear and logistics hitting diminishing returns

---

## Running the analysis

### Prerequisites

```r
install.packages(c(
  "randomForest", "modeldata", "lime", "fastshap",
  "shapviz", "iml", "ggplot2", "dplyr", "tidyr",
  "scales", "caret"
))
```

### Data setup

Download the HimEx dataset and place the three CSV files at the link given.

Or update the file paths at the top of `Final_Script_SC398.R` to match 
your local directory.

### Run

Open `Final_Script_SC398.R` in RStudio and run the full script. Plots 
render sequentially in the Plots pane.

---

## File

| File | Description |
|---|---|
| `Final_Script_SC398.R` | Complete analysis — data prep, models, SHAP, LIME, conformal prediction |

---

## Methods

Follows the Molnar interpretable ML pipeline:  
global importance (SHAP) → uncertainty quantification (conformal prediction) 
→ local explanation (LIME). Random Forest chosen for its non-linearity and 
compatibility with model-agnostic explanation methods.
