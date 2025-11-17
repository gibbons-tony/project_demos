# When Deep Learning Fails: Classical Features Beat Transfer Learning for Medical X-Ray Classification

Modern computer vision wisdom says deep learning always wins. Pre-trained ImageNet models transfer to new domains. Fine-tuning DenseNet121 or ResNet50 on medical images should outperform hand-crafted features. This project proves the opposite: **carefully engineered classical features (HOG, Fourier transforms, Local Binary Patterns) achieved 80.1% F1-score while state-of-the-art transfer learning models barely exceeded random guessing at 40.1%.**

The task: classify chest X-rays from Vietnamese hospitals into five conditions (Aortic Enlargement, Cardiomegaly, Pleural Effusion, Pulmonary Fibrosis, or No Finding). The dataset: 15,000+ images with 50,000+ radiologist annotations. The surprising result: a logistic regression model trained on 410 PCA-reduced classical features in 2 seconds outperformed a DenseNet121 model that required 400 seconds and hyperparameter tuning across learning rates, dropout, and fine-tuning layers.

This isn't a condemnation of deep learning—it's a case study in domain adaptation failure and the enduring value of domain-specific feature engineering when training data is scarce and the distribution shift is severe.

## The Clinical Motivation: AI That Catches What Humans Miss

Radiologists examine thousands of chest X-rays annually, searching for subtle patterns that indicate disease. Aortic enlargement appears as a widened mediastinum. Cardiomegaly shows as an enlarged cardiac silhouette. Pleural effusion creates fluid accumulation visible in costophrenic angles. Pulmonary fibrosis manifests as diffuse interstitial patterns—textural changes that even experienced doctors can miss in low-quality or ambiguous images.

**AI diagnostic assistance addresses three clinical needs**:

1. **Reduce misdiagnosis**: Models detect features invisible to the human eye, especially in fuzzy or low-contrast images where pathology blends with background noise.

2. **Augment junior radiologists**: Less experienced doctors gain a "second opinion" that highlights potential abnormalities, accelerating learning and reducing errors.

3. **Triage workload**: Experienced radiologists focus on complex cases flagged by the model, while routine "No Finding" cases get expedited clearance.

The critical requirement: models must be **interpretable** and **computationally efficient** for hospital deployment. A 400-second DenseNet121 inference time per batch is unacceptable for real-time clinical workflows. Classical feature extraction (2 seconds) with transparent decision boundaries (logistic regression coefficients) provides both speed and interpretability that deep learning cannot match in this constrained setting.

## Dataset: 50,000 Expert Annotations Across 14 Thoracic Conditions

**Vingroup Big Data Institute Chest X-Ray Abnormalities Detection** (Kaggle)

- **15,000+ images** from two Vietnamese hospitals (2018-2020)
- **50,000+ bounding box annotations** by approximately 20 radiologists
- **14 medical conditions** with co-occurrence patterns (multiple pathologies per image)
- **Imbalanced distribution**: "No Finding" dominates at 30,000+ instances, rare conditions <2,000

### Preprocessing: Selecting Learnable Classes

To create a balanced experimental design, we selected the **5 most abundant classes**:

| Class | Count | Clinical Description |
|-------|-------|---------------------|
| No Finding | ~1,000 | Healthy chest X-ray, no abnormalities detected |
| Aortic Enlargement | ~1,000 | Widened aorta, visible in upper mediastinum |
| Pulmonary Fibrosis | ~1,000 | Lung scarring with reticular/nodular patterns |
| Cardiomegaly | ~1,000 | Enlarged heart (cardiothoracic ratio >0.5) |
| Pleural Effusion | ~1,000 | Fluid accumulation in pleural space |

**Train/Validation/Test split**: 60:20:20 ratio, stratified to maintain class balance across splits.

This curation ensures each model sees sufficient examples per class while avoiding the extreme imbalance that would cause models to default to majority-class predictions.

## Feature Engineering: Why X-Ray Textures Require Classical Computer Vision

Medical images differ fundamentally from ImageNet's natural photos. A cat vs. dog classifier learns fur texture, ear shape, facial geometry—high-level semantic features abundant in ImageNet's 1.2 million labeled images. Chest X-rays present the opposite challenge: **subtle textural variations in grayscale intensity** that indicate disease, with limited training data (1,000 examples per class) and extreme within-class variability (patient anatomy, imaging equipment, positioning).

We implemented four complementary feature families, each capturing a distinct aspect of radiological pathology:

### 1. Histogram of Oriented Gradients (HOG)

**Purpose**: Detect structural changes in lung tissue (fibrosis, opacities)

**Method**: Divide image into small cells, compute gradient orientations in each cell, create histograms of edge directions

**Why it works for X-rays**:
- Pulmonary fibrosis creates reticular (net-like) patterns with strong directional edges
- Pleural thickening shows up as localized edge gradients along the pleural boundary
- Cardiomegaly alters the gradient distribution around cardiac silhouette borders

**Implementation**: 8×8 pixel cells, 9 orientation bins, block normalization

### 2. Fourier Transform

**Purpose**: Identify global frequency structures (calcifications, distributed patterns)

**Method**: Convert spatial image to frequency domain via Fast Fourier Transform, extract magnitude spectrum

**Why it works for X-rays**:
- Calcifications appear as high-frequency components (sharp intensity transitions)
- Distributed interstitial disease shows up as mid-frequency patterns
- Normal lung tissue has characteristic low-frequency structure

**Implementation**: 2D FFT, magnitude spectrum, log-transform for visualization

### 3. Local Binary Pattern (LBP)

**Purpose**: Capture fine-grained texture inconsistencies caused by disease

**Method**: For each pixel, compare to 8 neighbors, create binary code representing local texture pattern

**Why it works for X-rays**:
- Diffuse interstitial disease creates subtle texture variations invisible to HOG
- Ground-glass opacity (early fibrosis) changes local pixel relationships
- Pleural thickening introduces texture discontinuities at boundaries

**Implementation**: Radius=1, 8 neighbors, uniform patterns

### 4. Spatial Features

**Purpose**: Encode positional relationships between pathologies

**Method**: Extract bounding box coordinates from radiologist annotations, compute:
- Euclidean distances between pathology centers
- Angular relationships (upper lobe vs. lower lobe)
- Overlap percentages (co-occurring conditions)

**Why it works for X-rays**:
- Certain conditions co-occur spatially (e.g., pleural effusion often accompanies cardiomegaly)
- Anatomical constraints limit where pathologies appear (aortic enlargement always in mediastinum)
- Spatial context disambiguates visually similar textures

**Implementation**: Pairwise distance matrices, angle features, overlap ratios as numerical feature vector

### 5. Image Pyramid Decomposition

**Purpose**: Preserve both fine details and coarse structural features across scales

**Method**: Build Gaussian pyramid (successive downsampling with smoothing), extract features at multiple scales

**Why it works for X-rays**:
- Macro-level: Cardiac silhouette size (cardiomegaly)
- Meso-level: Regional opacity patterns (consolidation)
- Micro-level: Fine reticulations (early fibrosis)

**Implementation**: 3-level pyramid, combine features from all scales

### Combined Feature Vector

After extracting all features, we concatenated them into a single high-dimensional vector:

- HOG: 1,764 dimensions
- Fourier: 512 dimensions (magnitude spectrum)
- LBP: 256 dimensions (histogram of patterns)
- Spatial: 42 dimensions (pairwise metrics)
- Pyramid: 3× feature multiplier

**Total raw dimensionality**: ~7,500 features → **PCA reduction to 410 components** (retaining 95% variance)

This dimensionality reduction served dual purposes:
1. Computational efficiency for logistic regression and SVM
2. Noise reduction by eliminating low-variance components

## Transfer Learning: DenseNet121, ResNet50, EfficientNetB0

As a baseline comparison, we fine-tuned three state-of-the-art convolutional neural networks pre-trained on ImageNet (1.2M natural images):

**DenseNet121**: Dense connectivity pattern where each layer connects to every other layer
- 121 layers with dense blocks
- Effective feature reuse, reduced parameter count
- Strong performance on medical imaging benchmarks in literature

**ResNet50**: Deep residual network with skip connections
- 50 layers with residual blocks
- Solves vanishing gradient problem
- Enables training of very deep networks

**EfficientNetB0**: Optimized architecture with compound scaling
- Balanced depth, width, and resolution
- State-of-the-art ImageNet performance with fewer parameters
- Efficient inference for deployment

### Fine-Tuning Strategy

We used Keras Tuner for systematic hyperparameter optimization:

**Hyperparameters searched**:
- Learning rate: {1e-5, 3e-5, 5e-5, 1e-4}
- Dropout rate: {0.1, 0.3, 0.5}
- Number of fine-tuning layers: {10, 20, 50, all}
- Dense units in classification head: {64, 128, 256, 512}

**Training protocol**:
- Freeze ImageNet pre-trained weights initially
- Unfreeze top N layers for domain adaptation
- Multi-label binary classification (5 classes, sigmoid activation)
- Binary cross-entropy loss per class
- Adam optimizer with weight decay

**Expected outcome**: Transfer learning should leverage ImageNet's hierarchical features (edges → textures → parts → objects) and adapt them to medical imaging with minimal fine-tuning.

**Actual outcome**: Catastrophic failure (see Results section).

## Classical ML Models: Logistic Regression and SVM

Both models operated on the 410 PCA-reduced engineered features, using one-vs-rest classification (5 binary classifiers, one per disease).

### Logistic Regression

**Architecture**: Linear decision boundary per class with L2 regularization

**Hyperparameter tuning**: GridSearchCV with 5-fold cross-validation
- Regularization (C): {0.001, 0.01, 0.1, 1, 10, 100}
- Solver: {lbfgs, liblinear, saga}
- Max iterations: {1000, 5000}

**Training time**: 2 seconds (CPU)

**Advantages**:
- Interpretable coefficients (which features matter most)
- Fast inference (<1ms per image)
- No GPU required

### Support Vector Machine

**Architecture**: RBF kernel with soft margin

**Hyperparameter tuning**: GridSearchCV with 5-fold cross-validation
- Regularization (C): {0.1, 1, 10, 100}
- Kernel coefficient (gamma): {0.001, 0.01, 0.1, 1, 'scale', 'auto'}

**Training time**: 8 seconds (CPU)

**Advantages**:
- Non-linear decision boundaries via kernel trick
- Robust to outliers with soft margin
- Better handling of class overlap than linear models

## Results: Classical Features Dominate Transfer Learning

### Overall Performance

| Model | Total F1 | AUC | Accuracy | Training Time |
|-------|----------|-----|----------|---------------|
| **Logistic Regression** | **0.801** | **0.908** | 0.840 | 2 seconds |
| SVM | 0.769 | 0.902 | 0.807 | 8 seconds |
| CNN (DenseNet121) | 0.401 | 0.467 | 0.472 | 400 seconds |

**Key takeaway**: The simple logistic regression model outperformed the sophisticated DenseNet121 transfer learning approach by **100% in F1-score** (0.801 vs. 0.401) while training **200× faster** (2s vs. 400s).

### Per-Class Performance (Logistic Regression)

| Condition | F1-Score | AUC | Precision | Recall |
|-----------|----------|-----|-----------|--------|
| Aortic Enlargement | 0.896 | 0.960 | 0.89 | 0.90 |
| No Finding | 0.889 | 0.967 | 0.91 | 0.87 |
| Cardiomegaly | 0.845 | 0.915 | 0.87 | 0.82 |
| Pleural Thickening | 0.698 | 0.855 | 0.71 | 0.69 |
| Pulmonary Fibrosis | 0.679 | 0.846 | 0.72 | 0.64 |

**Clinical interpretation**:

- **"No Finding" (AUC 0.967)**: Exceptional performance. Model reliably identifies healthy X-rays, enabling fast-track clearance and reducing radiologist workload on routine cases.

- **"Aortic Enlargement" (AUC 0.960)**: Excellent detection. The widened aorta creates strong HOG gradients and distinct spatial features that the model captures effectively.

- **"Cardiomegaly" (AUC 0.915)**: Strong performance. Cardiac silhouette size is well-represented by spatial features and HOG boundaries.

- **"Pleural Thickening" (AUC 0.855)**: Good but lower. Subtle texture changes require finer-grained LBP features; some cases remain ambiguous even to radiologists.

- **"Pulmonary Fibrosis" (AUC 0.846)**: Acceptable but challenging. Diffuse interstitial patterns vary widely between patients; early-stage fibrosis has minimal visual signature.

### SVM Performance Insights

SVM achieved slightly lower overall F1 (0.769 vs. 0.801) but showed **improved detection of Pleural Thickening** (AUC 0.877 vs. 0.855 for logistic regression).

**Why SVM excels at this condition**: The RBF kernel captures non-linear texture patterns in pleural regions that linear logistic regression misses. However, this advantage doesn't outweigh the computational cost (4× slower) and reduced interpretability for clinical deployment.

### CNN Transfer Learning Failure

**Performance**: AUC values 0.413-0.536 (barely above random guessing at 0.50)

**Why it failed**:

1. **Domain shift**: ImageNet features (fur, leaves, cars, faces) don't transfer to grayscale medical textures (lung parenchyma, pleural boundaries, cardiac silhouettes). Pre-trained filters detect edges of everyday objects, not subtle radiological findings.

2. **Insufficient data for fine-tuning**: 1,000 examples per class is tiny for deep learning. Medical imaging benchmarks like ChestX-ray14 use 100,000+ images. Our dataset lacks the scale needed to retrain high-dimensional networks (DenseNet121 has 8M parameters).

3. **Subtle visual patterns**: Radiological findings often differ by grayscale intensity variations of <10% from normal tissue. ImageNet models learn high-contrast semantic features (cat vs. dog), not low-contrast texture nuances.

4. **Class similarity**: Different thoracic pathologies produce similar visual patterns. Cardiomegaly vs. pericardial effusion both enlarge the cardiac silhouette. Pulmonary fibrosis vs. pneumonia both create interstitial opacity. Without domain-specific inductive bias, CNNs confuse these overlapping presentations.

5. **Mutual exclusivity constraints**: "No Finding" is mutually exclusive with pathological classes, but the model treats all 5 classes as independent binary decisions. This architectural mismatch introduces logical inconsistencies (predicting both "No Finding" and "Cardiomegaly" for the same image).

**Attempted fixes** (none successful):
- Custom loss functions weighting rare classes higher
- Focal loss to address class imbalance
- Different fine-tuning strategies (freeze fewer/more layers)
- Data augmentation (rotation, translation, intensity shifts)
- Ensemble of DenseNet121 + ResNet50 + EfficientNetB0

## Feature Space Visualization: Why Classes Overlap

We applied t-SNE (t-distributed Stochastic Neighbor Embedding) to visualize the 410-dimensional PCA feature space in 2D:

**Observations**:
- **No clear clusters**: Unlike ImageNet data where "cat" and "airplane" form distinct clusters, our five thoracic conditions show **extensive overlap** in feature space.

- **Similar distribution patterns**: Aortic Enlargement and Cardiomegaly scatter across similar regions, reflecting their shared characteristic (both involve mediastinal/cardiac enlargement).

- **Intermingled rare classes**: Pleural Thickening and Pulmonary Fibrosis (both texture-based diagnoses) intermix, suggesting their engineered features capture similar patterns.

**Clinical relevance**: This overlap mirrors radiological reality. Experienced doctors frequently disagree on ambiguous cases (inter-rater agreement ~80% for some conditions). The feature space visualization reveals why:  pathologies exist on a continuum, not in discrete categories.

## Misclassification Analysis: Where Models Fail

### Confusion Matrix (Logistic Regression)

**Most common error**: Cardiomegaly misclassified as Aortic Enlargement (67 instances)

**Why this happens**:
- Both conditions involve enlargement in the mediastinal region
- Cardiomegaly often co-occurs with aortic changes in elderly patients
- Spatial features (size, position) are similar
- HOG gradients along enlarged cardiac border resemble aortic boundaries

**Clinical impact**: This error is **clinically acceptable**—both conditions require cardiac workup. Flagging either prompts appropriate follow-up (echocardiogram, CT angiography).

**Second most common error**: Pleural Thickening misclassified as No Finding (46 instances)

**Why this happens**:
- Mild pleural thickening is visually subtle, even to radiologists
- LBP texture features may not capture minimal thickening (<2mm)
- Image quality (noise, contrast) obscures findings

**Clinical impact**: This error is **potentially harmful**—missing pleural pathology could delay diagnosis of underlying conditions (TB, asbestos exposure, malignancy).

## Computational Efficiency: Deployment Feasibility

For hospital integration, training time and inference latency matter:

| Model | Training Time | Inference (per image) | Hardware |
|-------|--------------|----------------------|----------|
| Logistic Regression | 2 seconds | <1 millisecond | CPU |
| SVM | 8 seconds | ~5 milliseconds | CPU |
| DenseNet121 CNN | 400 seconds | ~50 milliseconds | GPU (T4) |

**Hospital deployment scenario**: Radiology department processing 500 X-rays/day

- **Logistic Regression**: 500ms total inference time, runs on hospital workstation CPU, instant results
- **CNN**: 25 seconds total inference time, requires dedicated GPU server, introduces lag in clinical workflow

**Retraining frequency**: Medical protocols evolve; models require periodic retraining on new data

- **Logistic Regression**: Retrain in 2 seconds during nightly maintenance window
- **CNN**: Retrain in 400 seconds, requires scheduling and GPU allocation

**Winner for production**: Logistic Regression on classical features provides the best balance of accuracy, speed, and interpretability for resource-constrained hospital environments.

## Lessons Learned: Domain Knowledge Beats Generic Features in Low-Data Regimes

### 1. Domain Adaptation Requires Domain-Aligned Data

ImageNet's 1.2M natural images don't prepare CNNs for medical X-rays. The distribution shift is too severe: color → grayscale, high-contrast objects → subtle textures, semantic categories → continuous pathological spectra.

**Actionable insight**: For medical imaging, pre-train on medical datasets (ChestX-ray14, MIMIC-CXR) or use self-supervised learning on unlabeled X-rays to learn radiological representations.

### 2. Sample Efficiency: Classical Features Win with <10K Examples

Logistic regression on engineered features achieves 80% F1 with 600 training images per class. DenseNet121 fails with the same data. Deep learning's data hunger makes it unsuitable for rare diseases or small hospitals with limited labeled datasets.

**Actionable insight**: For low-resource scenarios, invest effort in feature engineering rather than chasing deep learning state-of-the-art benchmarks from different domains.

### 3. Interpretability Matters for Clinical Adoption

Radiologists won't trust a "black box" CNN that can't explain its predictions. Logistic regression coefficients show which features (HOG gradients in upper mediastinum, LBP texture in lung bases) drive decisions, enabling clinical validation.

**Actionable insight**: Build hybrid systems where CNNs extract features and interpretable classifiers make final decisions, providing both performance and explainability.

### 4. Computational Constraints Are Real

Academic papers benchmark on high-end GPUs with unlimited time. Hospitals run on constrained budgets with CPU-only workstations. A 2-second model that runs anywhere beats a 400-second model requiring specialized hardware.

**Actionable insight**: Optimize for deployment constraints from the start, not as an afterthought. Real-world impact requires real-world feasibility.

### 5. Feature Engineering Isn't Dead

Modern ML culture dismisses hand-crafted features as "obsolete" compared to learned representations. This project proves otherwise: **domain expertise encoded as features (HOG for edges, Fourier for frequency, LBP for texture, spatial for anatomy) outperforms generic learned features when training data is scarce.**

**Actionable insight**: Combine classical computer vision with modern ML. Use domain knowledge to constrain the hypothesis space, not as a fallback when deep learning fails.

## Future Directions

**1. Hybrid Architecture**: Use CNNs as feature extractors (final conv layer activations) combined with classical features (HOG, Fourier, LBP), then train interpretable classifier on merged feature set.

**2. Medical Pre-training**: Fine-tune on ChestX-ray14 (100K+ images) first, then transfer to our Vietnamese hospital dataset. This aligns source and target domains better than ImageNet.

**3. Multi-task Learning**: Train joint model for all 14 original conditions, not just top 5. Shared representations may improve rare class detection through transfer within the medical domain.

**4. Active Learning**: Identify ambiguous cases where the model is uncertain (prediction probability 0.4-0.6) and request additional radiologist annotations for those specific examples.

**5. Explainable AI Integration**: Implement GradCAM or attention maps to visualize which image regions drive CNN predictions, enabling clinical validation even for deep models.

## Repository Structure

```
computer_vision_demo/
├── 281_project_notebook_4_17_25.ipynb   # Full implementation (21MB - includes outputs)
├── Spring2025 w281 Final - Xray.pdf     # 25-page presentation with visuals
├── README.md                             # This file
├── data/                                 # Vingroup Chest X-ray dataset
│   ├── train/                           # 15,000+ X-ray images
│   ├── test/
│   └── annotations.csv                  # 50,000+ bounding boxes
└── models/                               # Saved model weights
    ├── logistic_regression.pkl
    ├── svm.pkl
    └── densenet121_finetuned.h5
```

## Running the Code

### Prerequisites
```bash
# Python 3.9+
# GPU recommended for CNN experiments (but not needed for best results)

pip install -r requirements.txt
```

### Dataset
Download from [Kaggle: VinBigData Chest X-ray Abnormalities Detection](https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection)

### Training Pipeline
```python
# Extract classical features
from feature_extraction import extract_hog, extract_fourier, extract_lbp, extract_spatial, build_pyramid

features_hog = extract_hog(images)
features_fourier = extract_fourier(images)
features_lbp = extract_lbp(images)
features_spatial = extract_spatial(bounding_boxes)
features_pyramid = build_pyramid(images)

# Concatenate and reduce dimensionality
features_combined = np.concatenate([features_hog, features_fourier, features_lbp, features_spatial, features_pyramid], axis=1)

from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)  # Retain 95% variance → 410 components
features_pca = pca.fit_transform(features_combined)

# Train logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
model = GridSearchCV(LogisticRegression(max_iter=5000), param_grid, cv=5)
model.fit(X_train_pca, y_train)

# Evaluate
from sklearn.metrics import classification_report, roc_auc_score
y_pred = model.predict(X_test_pca)
print(classification_report(y_test, y_pred))
```

### Reproducing Results
```bash
# Run full experiment notebook
jupyter notebook 281_project_notebook_4_17_25.ipynb

# Key cells:
# - Cell 15-20: Feature extraction
# - Cell 25: PCA dimensionality reduction
# - Cell 30-35: Logistic Regression training
# - Cell 40-45: SVM training
# - Cell 50-60: CNN transfer learning (DenseNet121)
# - Cell 65-70: Results visualization
```

## Key Takeaways: Transfer Learning Fails Without Domain Alignment

**1. Transfer learning is not a silver bullet**: DenseNet121 pre-trained on ImageNet failed completely (40% F1) on medical X-rays due to domain mismatch. Pre-training source must align with target domain.

**2. Sample efficiency matters**: With only 1,000 examples per class, classical engineered features (80% F1) vastly outperformed deep learning (40% F1). Data-hungry models need data-rich scenarios.

**3. Feature engineering encodes domain expertise**: HOG captures structural lung changes, Fourier identifies calcifications, LBP detects texture pathology, spatial features encode anatomical constraints. This domain knowledge beats generic learned features in low-data regimes.

**4. Computational efficiency enables deployment**: 2-second training, <1ms inference, CPU-only execution makes logistic regression deployable in resource-constrained hospitals. 400-second GPU-dependent CNNs are not.

**5. Interpretability drives clinical adoption**: Radiologists can validate logistic regression coefficients (which features drive decisions) but not DenseNet121's 8M parameter black box. Explainability builds trust.

---

**Built for**: UC Berkeley MIDS W281 (Computer Vision)
**Team**: Tony Gibbons, Meric, Mohak, Alexander
**Dataset**: VinBigData Chest X-ray Abnormalities (15,000+ images, 50,000+ annotations)
**Task**: 5-class thoracic disease classification
**Best Result**: Logistic Regression on classical features - 80.1% F1, 90.8% AUC (2s training time, CPU-only)
**Surprising Finding**: Classical computer vision features outperformed ImageNet transfer learning by 100% in F1-score
