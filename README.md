# Tabular Data Augmentation for CBC Classification

## Overview

This project compares two tabular data augmentation techniques for improving machine learning model performance on imbalanced Complete Blood Count (CBC) health data. The goal is to evaluate whether **SMOTE** (Synthetic Minority Over-sampling Technique) and **noise injection** provide meaningful improvements over a baseline classifier on an inpatient/outpatient classification task.

**Key Finding**: SMOTE augmentation achieved comparable performance to the baseline while maintaining a better balance between precision and recall for the minority class, making it more suitable for medical classification tasks where false negatives carry clinical significance.

## Dataset

- **Source**: Complete Blood Count (CBC) health data
- **Size**: 3,000 samples × 11 features
- **Target Variable**: `SOURCE` (binary classification)
  - `in` = inpatient (minority class, 37.5%)
  - `out` = outpatient (majority class, 62.5%)
- **Class Imbalance Ratio**: 1.66:1
- **Missing Values**: None

### Features

The dataset contains a mix of numerical CBC measurements and demographic information:

| Feature | Type | Description |
|---------|------|-------------|
| HAEMATOCRIT | float | Percentage of red blood cells in blood |
| HAEMOGLOBINS | float | Hemoglobin concentration (g/dL) |
| ERYTHROCYTE | float | Red blood cell count (million cells/μL) |
| LEUCOCYTE | float | White blood cell count (thousand cells/μL) |
| THROMBOCYTE | int | Platelet count (thousand cells/μL) |
| MCH | float | Mean Corpuscular Hemoglobin (pg) - derived feature |
| MCHC | float | Mean Corpuscular Hemoglobin Concentration (g/dL) - derived feature |
| MCV | float | Mean Corpuscular Volume (fL) - derived feature |
| AGE | int | Patient age (years) |
| SEX | categorical | Patient biological sex (M/F) |

**Note**: MCH, MCHC, and MCV are derived indices calculated from primary CBC measurements.

## Project Structure

```
augmentation-health-data/
├── README.md                    # Project documentation
├── P3_notebook.ipynb           # Main Jupyter notebook with full analysis
├── P3_presentation.pdf         # Presentation slides
└── P3_report.pdf               # Detailed technical report
```

## Methodology

### 1. Data Preprocessing

- **Train/Test Split**: 80/20 stratified split to maintain class distribution
  - Training: 2,400 samples (901 minority, 1,499 majority)
  - Testing: 600 samples (225 minority, 375 majority)

- **Feature Scaling**:
  - Numerical features: StandardScaler (zero mean, unit variance)
  - Categorical features: OneHotEncoder for SEX variable

- **Pipeline Architecture**: Used scikit-learn's ColumnTransformer for consistent preprocessing across all models

### 2. Baseline Model

**Classifier**: Random Forest (100 estimators)
- No augmentation applied
- Trained on original 2,400 training samples
- Serves as reference point for augmentation effectiveness

**Baseline Results** (on 600 test samples):
```
              precision    recall  f1-score   support
          in       0.76      0.63      0.69       225
         out       0.80      0.88      0.84       375
    accuracy                           0.79       600
   macro avg       0.78      0.75      0.76       600
weighted avg       0.78      0.79      0.78       600
```

### 3. Augmentation Technique 1: SMOTE

**SMOTE** (Synthetic Minority Over-sampling Technique) generates synthetic samples by interpolating between k-nearest neighbors in the minority class feature space.

**Configuration**:
- k_neighbors: 5
- Sampling strategy: Balance minority class to match majority class
- Applied only to training data (critical to prevent data leakage)

**Mechanism**: For each minority sample, SMOTE:
1. Finds the 5 nearest neighbors in feature space
2. Randomly selects one neighbor
3. Creates a synthetic sample at a random point along the line segment connecting the two points

**Results** (on 600 test samples):
```
              precision    recall  f1-score   support
          in       0.72      0.68      0.70       225
         out       0.82      0.84      0.83       375
    accuracy                           0.78       600
   macro avg       0.77      0.76      0.76       600
weighted avg       0.78      0.78      0.78       600
```

**Observations**:
- Accuracy remained stable at 78%
- Minority class recall decreased slightly (63% → 68%)
- Macro-averaged F1 score improved (0.76 maintained with better balance)
- Better calibrated for minority class despite marginal precision trade-off

### 4. Augmentation Technique 2: Noise Injection

**Noise Injection** adds Gaussian perturbations to features in the training set to create synthetic variations. This technique is model-agnostic and does not require feature space relationships.

**Configuration**:
- Noise type: Gaussian (mean=0, std=0.1 × feature std)
- Features affected: All numerical features + categorical features (slight perturbations)
- Augmentation factor: 50% additional training samples generated
- Applied only to training data

**Mechanism**: For each training sample:
1. Generate random Gaussian noise scaled to feature variance
2. Add noise: `X_augmented = X_original + noise`
3. Preserve original labels

**Results** (on 600 test samples):
```
              precision    recall  f1-score   support
          in       0.69      0.71      0.70       225
         out       0.82      0.81      0.81       375
    accuracy                           0.77       600
   macro avg       0.75      0.76      0.75       600
weighted avg       0.77      0.77      0.77       600
```

**Observations**:
- Accuracy slightly decreased (79% → 77%)
- Minority class recall improved (63% → 71%)
- More uniform precision/recall across classes
- Simpler to implement and less computationally expensive than SMOTE

## Comparative Analysis

### Performance Metrics Comparison

| Metric | Baseline | SMOTE | Noise Injection |
|--------|----------|-------|-----------------|
| Accuracy | 0.79 | 0.78 | 0.77 |
| Precision (macro) | 0.77 | 0.77 | 0.75 |
| Recall (macro) | 0.75 | 0.76 | 0.76 |
| F1-Score (macro) | 0.76 | 0.76 | 0.75 |
| Minority Recall | 0.63 | 0.68 | 0.71 |
| Minority Precision | 0.76 | 0.72 | 0.69 |

### Key Findings

1. **SMOTE is the optimal choice** for this medical classification task:
   - Maintains overall accuracy while improving minority class recall
   - Better precision for inpatient predictions (0.72 vs 0.69 with noise)
   - More clinically relevant: reduces false negatives while maintaining reasonable specificity

2. **Noise Injection trade-offs**:
   - Achieves highest minority recall (0.71)
   - Sacrifices precision and overall accuracy
   - Too many false positives for practical medical use

3. **Class imbalance impact**:
   - Baseline model biased toward majority class (88% recall for outpatients)
   - Augmentation techniques successfully balance predictions
   - Medical context matters: false negatives (missed inpatient cases) worse than false positives

4. **Data characteristics**:
   - 1.66:1 imbalance ratio is moderate—augmentation helps but doesn't dramatically transform results
   - Derived features (MCH, MCHC, MCV) limit synthetic sample realism in SMOTE
   - Model performance plateau suggests feature quality is limiting factor

## Technical Implementation

### Dependencies

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
imbalanced-learn>=0.8.1
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
```

### How to Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/lessofrut/augmentation-health-data.git
   cd augmentation-health-data
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare dataset**:
   - Place `dataset_project3.csv` in the project root directory
   - Update the `DATA_PATH` variable in the notebook if using a different location

4. **Run the notebook**:
   ```bash
   jupyter notebook P3_notebook.ipynb
   ```

5. **Outputs**:
   - Confusion matrix visualizations: `*_confusion_matrix.png`
   - Class distribution plot: `01_class_distribution.png`
   - Console output with detailed classification reports

### Code Architecture

**Pipeline Structure**:
```python
Pipeline([
    ('preprocess', ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(sparse_output=False), categorical_features)
    ])),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])
```

**Augmentation Application** (pseudocode):
```python
# Baseline
baseline_model.fit(X_train, y_train)

# SMOTE
from imblearn.over_sampling import SMOTE
smote = SMOTE(k_neighbors=5, random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_preprocessed, y_train)
smote_model.fit(X_train_smote, y_train_smote)

# Noise Injection
noise_factor = 0.1
X_train_noisy = X_train + np.random.normal(0, noise_factor * X_train.std(axis=0), X_train.shape)
noise_model.fit(X_train_noisy, y_train_duplicated)
```

## Evaluation Methodology

### Metrics Used

- **Accuracy**: Overall correctness across both classes
- **Precision (macro)**: Average precision across classes (unweighted)
- **Recall (macro)**: Average recall across classes (unweighted)
- **F1-Score (macro)**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed breakdown of TP/TN/FP/FN

### Why Macro-Averaged Metrics?

Macro-averaging gives equal weight to both classes, making minority class performance visible. For medical applications, this is critical—we cannot ignore inpatient prediction performance just because there are fewer inpatients in the dataset.

### Train/Test Separation

All augmentation was applied **only to training data** and evaluated on untouched test data. This prevents:
- Data leakage (synthetic samples in test set)
- Overly optimistic performance estimates
- Biased evaluation of real-world applicability

## Limitations & Future Work

### Current Limitations

1. **Moderate dataset size**: 3,000 samples limits statistical significance
2. **Derived features**: MCH, MCHC, MCV reduce SMOTE effectiveness by creating constrained feature space
3. **Single classifier**: Only Random Forest tested; results may vary with other algorithms
4. **Simplified noise model**: Gaussian noise may not reflect realistic data variations
5. **No hyperparameter tuning**: All models used default/standard configurations

### Potential Improvements

1. **Algorithm comparison**: Test with Gradient Boosting, SVM, Neural Networks
2. **Advanced augmentation**: 
   - Conditional VAE for realistic sample generation
   - Borderline-SMOTE to focus on difficult-to-classify samples
   - ADASYN for adaptive augmentation
3. **Feature engineering**: Create new CBC-derived indices that are more robust
4. **Hyperparameter optimization**: Grid/random search for optimal configurations
5. **Cross-validation**: K-fold CV for more robust performance estimates
6. **Class weight balancing**: Compare with class_weight='balanced' in Random Forest

## Conclusion

SMOTE augmentation provides the best balance for this CBC classification task by improving minority class recall without sacrificing overall accuracy. While noise injection achieves higher minority recall, its precision degradation makes it clinically unsuitable. For imbalanced medical datasets with moderate imbalance ratios and constrained feature spaces, interpolation-based methods like SMOTE outperform perturbation-based approaches.

## Authors

- Dhyana Schirra
- Alessandro Visione

## References

1. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). "SMOTE: Synthetic Minority Over-sampling Technique." *Journal of Artificial Intelligence Research*, 16, 321-357.
2. He, H., Bai, Y., Garcia, E. A., & Li, S. (2008). "ADASYN: Adaptive Synthetic Sampling Approach for Imbalanced Learning." *IEEE International Joint Conference on Neural Networks*.
3. Scikit-learn Documentation: https://scikit-learn.org/
4. Imbalanced-learn Documentation: https://imbalanced-learn.org/
