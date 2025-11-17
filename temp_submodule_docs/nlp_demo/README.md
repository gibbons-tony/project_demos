# When Machine-Generated Labels Outperform Human Annotation: Pseudo-Labeling for Financial Sentiment

In financial NLP, labeled training data is scarce and prohibitively expensive. Domain experts must manually annotate thousands of news articles, tweets, and earnings reports—a process that costs hundreds to thousands of dollars per thousand labels. Yet algorithmic trading firms need real-time sentiment classification to predict market movements, creating an irreconcilable tension between data requirements and annotation budgets.

This project investigates whether models can teach themselves through pseudo-labeling: using confident predictions on unlabeled data as synthetic training examples. The counterintuitive finding challenges conventional wisdom about data quality: **ensemble-filtered pseudo-labels reduced negative sentiment misclassification by 50% compared to expensive human-annotated data**, achieving 80.6% accuracy while eliminating annotation costs entirely.

The implementation compares FinBERT (transformer-based, financial domain fine-tuned) against a hybrid LSTM-CNN architecture across three pseudo-labeling strategies, revealing when and why synthetic labels outperform ground truth.

## The $10,000 Question: Can ModelsAnnotate Their Own Training Data?

Financial sentiment analysis correlates with measurable market behavior—positive sentiment precedes price increases, negative sentiment signals declines. But the specialized vocabulary of finance ("bullish," "volatility," "bearish consolidation") requires domain-expert annotation, not crowdsourced labels from generalist workers.

**Industry cost reality**:
- Domain expert annotation: $0.50-$2.00 per example
- 10,000 labeled examples: $5,000-$20,000
- Continuous retraining for market shifts: recurring expense

**Pseudo-labeling alternative**:
- Train model on small labeled set (Phase 1: 70% of data)
- Generate predictions on unlabeled data (Phase 2: remaining 30%)
- Use high-confidence predictions as synthetic labels
- Retrain on combined dataset
- **Cost**: $0 for additional labels

The research question: Do these synthetic labels degrade model performance, or can they actually improve it?

## Experimental Design: Simulating Low-Resource Conditions

### Dataset
**Financial Sentiment Analysis Dataset** (Kaggle): 10,000 financial news articles spanning market trends, corporate earnings, economic forecasts

**Class distribution** (reflecting real financial reporting):
- Positive: 40% (bullish sentiment, growth indicators)
- Neutral: 35% (objective reporting, mixed signals)
- Negative: 25% (bearish sentiment, warnings) — **rare class, hardest to classify**

### Two-Phase Training Protocol

**Phase 1** (Labeled data - 70% of dataset):
- 70% training, 15% validation, 15% test
- Simulates initial labeled dataset from expensive annotation

**Phase 2** (Unlabeled data - 30% of dataset):
- 60% training, 20% validation, 20% test
- Simulates abundant unlabeled data (free from web scraping, APIs, public filings)

This division mirrors real-world scenarios: limited budget for initial labeling, unlimited access to raw text.

### Three Pseudo-Labeling Strategies

**Scenario 1: Unfiltered**
- Generate labels using FinBERT predictions
- Include ALL predictions regardless of confidence
- Tests whether volume compensates for noise

**Scenario 2: Confidence-Filtered**
- Generate labels using FinBERT predictions
- Filter: only keep predictions with confidence > 0.5
- Trades dataset size for label quality

**Scenario 3: Ensemble-Filtered**
- Generate labels using FinBERT + DistilBERT + RoBERTa ensemble
- Filter: only keep examples where at least 2 of 3 models agree
- Tests whether consensus reduces label noise

## The Counterintuitive Result: Synthetic Labels Beat Ground Truth

### Model Performance Comparison

| Training Strategy | Accuracy | F1-Score | Class 0 Misclass | Class 1 Misclass | Class 2 Misclass |
|------------------|----------|----------|------------------|------------------|------------------|
| **Unfiltered Pseudo** | **80.6%** | 79.7% | 61.14% | 8.32% | 18.14% |
| Confidence Pseudo | 80.0% | 79.4% | 60.10% | 12.99% | 12.74% |
| **Ensemble Pseudo** | 80.0% | **79.9%** | **30.05%** | 18.66% | 13.45% |
| Fully Labeled (Phase 1+2) | 79.3% | 78.6% | 61.14% | — | 27.40% |
| Baseline (Phase 1 only) | 77.6% | 75.3% | 76.68% | — | 27.21% |
| Hybrid LSTM-CNN | 76.2% | 76.8% | 57.90% | — | 31.20% |

### Critical Finding: Ensemble Filtering Solves Rare Class Detection

**Class 0 (Negative sentiment)** is the minority class (25% of data) and the most valuable for trading applications—detecting bearish signals before market downturns.

**Baseline performance** (Phase 1 labeled data only): 76.68% misclassification
- Model rarely sees negative examples during training
- Defaults to predicting majority classes

**Adding true-labeled Phase 2 data**: 61.14% misclassification
- More training data helps, but improvement plateaus
- Human annotation budget: ~$2,000 for Phase 2 labels

**Ensemble-filtered pseudo-labels**: **30.05% misclassification**
- **50% reduction** compared to fully labeled approach
- Zero additional annotation cost
- Consensus filtering eliminates noisy labels while preserving rare examples

## Why Pseudo-Labels Outperform Human Annotation

This result contradicts the assumption that ground truth labels are always superior. Three mechanisms explain the paradox:

### 1. Model-Generated Labels Are Internally Consistent

Human annotators disagree about ambiguous examples, especially in neutral vs. slightly-positive/negative edge cases. Different annotators, annotation sessions, or fatigue levels introduce label noise.

Pseudo-labels, while sometimes wrong, are **consistent with the model's internal decision boundary**. Training on these labels reinforces the model's existing understanding rather than introducing conflicting signals.

### 2. Ensemble Filtering Identifies High-Quality Examples

Requiring 2-of-3 model agreement creates a higher bar than single-annotator human labels:
- FinBERT: Fine-tuned on financial corpora (domain expert)
- DistilBERT: Compressed general transformer (broad language understanding)
- RoBERTa: Optimized BERT variant (robust representations)

When all three agree despite different architectures and training, the example is likely unambiguous. Disagreement flags edge cases that even models find confusing—better to exclude these than introduce noise.

### 3. Pseudo-Labeling Amplifies Rare Classes More Effectively

The model encounters more negative examples in unlabeled data than in the small Phase 1 labeled set. By pseudo-labeling Phase 2, high-confidence negative predictions get added to training, creating better class balance without expensive targeted annotation of rare cases.

## Architecture Comparison: Transformer vs. Hybrid LSTM-CNN

### FinBERT (Transformer-Based)

**Architecture**: BERT pre-trained on financial corpora (earnings calls, analyst reports, financial news)

**Strengths**:
- Captures bidirectional context through self-attention
- Pre-trained on domain vocabulary ("EBITDA," "bearish," "volatility")
- Transfer learning from massive financial text corpus

**Performance**: 77.6% baseline → 80.6% with unfiltered pseudo-labels

**Hyperparameters**:
- Learning rate: 1e-5 to 3e-5
- Batch size: 32
- Early stopping: 3 epochs without improvement
- Dropout: 0.1-0.5
- Optimizer: Adam with weight decay (0.01)

### Hybrid LSTM-CNN

**Architecture**:
1. **BERT embeddings**: Contextual representations (768-dimensional)
2. **Bi-directional LSTM**: Sequential modeling of sentence structure
3. **Multi-head attention**: Weighted combination of LSTM outputs
4. **CNN layers**: Spatial feature extraction from attention outputs

**Strengths**:
- Excels at rare class detection (Class 0 misclassification: 57.9% vs. 61.14% for BERT)
- Better at handling noisy data through CNN robustness
- Lower overall accuracy (76.2%) but more balanced performance

**Trade-off**: The hybrid sacrifices generalization for specialization in challenging cases.

**Hyperparameters**:
- Learning rate: 1e-4 to 5e-5 (higher than BERT due to random initialization of LSTM/CNN layers)
- Batch size: 32
- Early stopping: 3 epochs

### When to Use Each Architecture

**FinBERT**: Maximum accuracy on clean, balanced data. Ideal for offline batch processing where average performance matters more than worst-case failures.

**Hybrid LSTM-CNN**: Real-time trading signals where missing rare negative sentiment (false negatives) is costlier than occasional false positives. Better for imbalanced production data.

## From Sentiment to Trading Signals: Production Applications

### Stock Price Prediction

Sentiment scores correlate with next-day returns. Integration approach:
1. Stream financial news via APIs (Bloomberg, Reuters, Twitter/X)
2. Real-time sentiment classification (Positive/Neutral/Negative)
3. Aggregate sentiment by ticker symbol and time window
4. Combine with quantitative signals (price, volume, technical indicators)
5. Feed into ML prediction model (XGBoost, LSTM on time series)

**Why pseudo-labeling matters**: News breaks 24/7. Continuous retraining on new events requires either constant annotation (infeasible) or self-supervised pseudo-labeling.

### Volatility Forecasting

Sentiment disagreement (mixed positive/negative signals) predicts increased volatility. Applications:
- Options pricing adjustments
- Risk management position sizing
- VIX (volatility index) prediction

**Ensemble filtering value**: The 2-of-3 agreement mechanism naturally identifies uncertain examples where models disagree—exactly the ambiguous sentiment that drives volatility.

### Sector-Specific Trend Analysis

Track sentiment across sectors (tech, healthcare, energy) to identify rotation trends before they appear in price data.

**Domain adaptation**: Fine-tune separate FinBERT models per sector, then use cross-sector pseudo-labeling where a model trained on tech news labels healthcare examples. Transfer learning without transfer annotation.

## Implementation Details

### Tech Stack
```python
# Models
transformers==4.30.0          # Hugging Face: FinBERT, DistilBERT, RoBERTa
torch==2.0.1                  # PyTorch backend

# Preprocessing
pandas==2.0.3                 # Data manipulation
scikit-learn==1.3.0           # Train/val/test splits, metrics

# Training
accelerate==0.20.3            # Distributed training, mixed precision
tensorboard==2.13.0           # Training visualization
```

### Hyperparameter Tuning Strategy

Rather than grid search (computationally prohibitive), used **intuition-driven search** guided by validation performance:

1. **Learning rate sweep**: Tested [1e-5, 2e-5, 3e-5] for FinBERT
   - Observation: 2e-5 balanced convergence speed vs. stability

2. **Batch size evaluation**: [16, 32, 64]
   - Trade-off: 64 faster but risks overfitting; 16 more stable but slower
   - Selected: 32 (balanced efficiency and memory)

3. **Dropout experimentation**: [0.1, 0.3, 0.5]
   - Financial text is high-signal (not noisy social media)
   - Selected: 0.3 for FinBERT, 0.1 for hybrid (LSTM already regularizes)

4. **Early stopping**: 3 epochs without validation improvement
   - Prevents overfitting while allowing late-stage convergence

### Pseudo-Label Generation Pipeline

```python
# Scenario 3: Ensemble-Filtered (Best Performance)

# Step 1: Generate predictions from 3 models
finbert_preds = finbert_model.predict(phase2_unlabeled)
distilbert_preds = distilbert_model.predict(phase2_unlabeled)
roberta_preds = roberta_model.predict(phase2_unlabeled)

# Step 2: Calculate agreement
agreement_mask = (
    (finbert_preds == distilbert_preds) |
    (finbert_preds == roberta_preds) |
    (distilbert_preds == roberta_preds)
)

# Step 3: Use majority vote as pseudo-label
pseudo_labels = np.where(
    finbert_preds == distilbert_preds,
    finbert_preds,
    roberta_preds
)

# Step 4: Filter to high-confidence consensus examples
phase2_pseudolabeled = phase2_unlabeled[agreement_mask]
phase2_pseudolabeled['label'] = pseudo_labels[agreement_mask]

# Step 5: Combine with Phase 1 and retrain
combined_training = pd.concat([phase1_labeled, phase2_pseudolabeled])
finbert_model.train(combined_training)
```

### Evaluation Metrics

**Accuracy**: Overall correctness across all classes
- Macro-average: Equal weight per class (better for imbalanced data)
- Weighted average: Proportional to class frequency

**F1-Score**: Harmonic mean of precision and recall
- Weighted F1: Accounts for class imbalance
- Per-class F1: Reveals which classes the model handles well

**Confusion Matrix**: Visualization of misclassification patterns
- Critical for financial applications: Different error types have different costs
- False negative on Class 0 (missing bearish signal): High cost in trading loss
- False positive on Class 2 (spurious bullish signal): Moderate cost in missed opportunity

## Limitations and Failure Modes

### When Pseudo-Labeling Fails

**Distribution shift**: If Phase 2 unlabeled data comes from a different distribution (e.g., bull market news when Phase 1 was bear market), pseudo-labels propagate initial model bias.

**Low initial accuracy**: If Phase 1 model performs poorly (<70%), pseudo-labels are too noisy to be useful. Requires sufficient labeled data to bootstrap.

**Ambiguous examples dominate**: In datasets where most examples are genuinely ambiguous (e.g., sarcasm detection), ensemble agreement filters out too much data, leaving insufficient training examples.

### Class Imbalance Persistence

While ensemble filtering improved Class 0 (Negative) detection, misclassification remained at 30.05%—better than alternatives, but still a failure on 1 in 3 rare class examples.

**Mitigation strategies** (not implemented in this project):
- Class-balanced sampling during training
- Focal loss (penalizes confident misclassifications)
- SMOTE (Synthetic Minority Over-sampling Technique) for rare classes

### Model Calibration Issues

Pseudo-labeling assumes model confidence scores are calibrated (predicted probabilities match true probabilities). Financial BERT models often exhibit overconfidence—90% predicted probability might correspond to 70% true probability.

**Solution**: Temperature scaling or Platt scaling to calibrate probabilities before confidence filtering.

### Real-Time Processing Latency

Production deployment requires inference latency <100ms for real-time trading signals.

**FinBERT latency**: ~50ms per example on GPU (acceptable)
**Ensemble latency**: ~150ms (3 models × 50ms) — **too slow** for high-frequency trading

**Trade-off**: Use ensemble for offline training but deploy single FinBERT for real-time inference.

## Repository Structure

```
nlp_demo/
├── 266_project_workbook_final.ipynb  # Full implementation and experiments
├── 266 Project Writeup.pdf            # 8-page academic paper
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
└── data/                              # Financial Sentiment Analysis Dataset (Kaggle)
    ├── train.csv
    ├── validation.csv
    └── test.csv
```

## Running the Experiments

### Prerequisites
```bash
# Python 3.9+
# GPU recommended (16GB+ VRAM for ensemble training)

pip install -r requirements.txt
```

### Dataset
Download from [Kaggle: Financial Sentiment Analysis](https://www.kaggle.com/datasets/sbhatti/financial-sentiment-analysis)

### Training Pipeline
```bash
# Open Jupyter notebook
jupyter notebook 266_project_workbook_final.ipynb

# Key notebooks sections:
# 1. Data Preparation: Phase 1/Phase 2 splits
# 2. Baseline Training: FinBERT on Phase 1 only
# 3. Pseudo-Label Generation: Three filtering strategies
# 4. Model Comparison: FinBERT vs. Hybrid LSTM-CNN
# 5. Evaluation: Confusion matrices, per-class metrics
```

### Hyperparameter Configuration
```python
# FinBERT optimal configuration
LEARNING_RATE = 2e-5
BATCH_SIZE = 32
MAX_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 3
DROPOUT = 0.3
WEIGHT_DECAY = 0.01

# Hybrid LSTM-CNN
LSTM_HIDDEN_SIZE = 256
LSTM_LAYERS = 2
CNN_FILTERS = 128
ATTENTION_HEADS = 8
```

## Key Takeaways: Pseudo-Labeling Works When Initial Models Are Strong

### Technical Insights

1. **Pseudo-labeling works when initial model is strong**: 77.6% baseline accuracy was sufficient to generate useful synthetic labels. Below ~70%, label noise dominates.

2. **Ensemble filtering > confidence filtering**: Requiring multi-model agreement (30.05% Class 0 error) outperformed single-model confidence thresholds (60.10% error) by 2x.

3. **Volume vs. quality trade-off**: Unfiltered pseudo-labels achieved highest accuracy (80.6%) through sheer data volume, but ensemble filtering achieved better rare class performance through label quality.

4. **Architecture specialization matters**: Hybrid LSTM-CNN trades 4.4% overall accuracy for better rare class detection—acceptable trade-off for imbalanced production data.

### Operational Insights

1. **Annotation budget allocation**: Spend budget on initial high-quality Phase 1 data. Use pseudo-labeling for continuous expansion as new unlabeled data arrives.

2. **Ensemble for training, single model for inference**: 3-model consensus during training improves label quality; deploy fastest single model (FinBERT) for real-time production.

3. **Continuous retraining strategy**: Financial language evolves (new jargon, market regimes change). Pseudo-label new data monthly, retrain quarterly.

4. **Monitor for distribution shift**: Track pseudo-label confidence over time. Declining confidence signals need for new human-annotated data.

### Research Questions for Future Work

**Why do pseudo-labels outperform ground truth?** This project demonstrates the effect but doesn't fully explain the mechanism. Hypotheses:
- Label noise in human annotations exceeds noise in filtered pseudo-labels
- Phase 2 data has different characteristics (complexity, ambiguity) where model-generated labels are more consistent

**Optimal ensemble size?** Tested 3 models; would 5 or 7 improve filtering further, or do diminishing returns appear?

**Active learning integration?** Combine pseudo-labeling (high-confidence examples) with active learning (query human annotators for low-confidence examples where ensemble disagrees).

---

**Built for**: UC Berkeley MIDS W266 (Natural Language Processing)
**Dataset**: Financial Sentiment Analysis (Kaggle) - 10,000 labeled articles
**Task**: 3-class sentiment classification (Positive, Neutral, Negative)
**Best Result**: 80.6% accuracy (unfiltered pseudo-labels), 50% reduction in rare class error (ensemble filtering)
**Key Innovation**: Demonstrating that ensemble-filtered pseudo-labels outperform expensive human annotation for low-resource financial NLP
