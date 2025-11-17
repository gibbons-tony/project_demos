# Machine Learning Engineering Portfolio: When One Size Never Fits All

The most dangerous assumption in machine learning is that a single approach works everywhere. Deep learning dominates headlines. LLMs promise to solve every NLP task. Transfer learning should always beat classical features. Yet real-world ML engineering demands a different skill: **knowing when conventional wisdom fails and which alternative approach will actually work.**

This portfolio demonstrates that versatility through four production-oriented projects spanning NLP, computer vision, generative AI, and cloud deployment—each revealing a critical lesson about when to break the rules:

- **NLP**: Ensemble-filtered pseudo-labels reduced misclassification by 50% compared to expensive human annotation
- **Computer Vision**: Classical features (HOG, Fourier, LBP) achieved 80% F1 while state-of-the-art transfer learning failed at 40%
- **Generative AI**: 256-token chunk size mattered more than LLM choice for RAG system performance
- **Cloud Deployment**: Production-ready ML API serving 10 requests/second with Kubernetes autoscaling and Redis caching

Rather than demonstrating mastery of a single framework or methodology, these projects collectively show the engineering judgment required to navigate real-world constraints: limited training data, domain mismatch, compliance requirements, computational budgets, and production latency targets.

## The Four Projects: Different Domains, Same Discipline

### 1. NLP Demo: When Machine-Generated Labels Outperform Human Annotation

**The counterintuitive finding**: Ensemble-filtered pseudo-labels eliminated annotation costs entirely while reducing negative sentiment misclassification by 50% compared to expensive human-labeled data.

**The constraint**: Financial sentiment analysis requires domain-expert annotation at $0.50-$2.00 per example. A 10,000-example dataset costs $5,000-$20,000—prohibitive for continuous retraining as markets evolve.

**The solution**: Train FinBERT on a small labeled set, generate pseudo-labels on unlabeled data, filter using 3-model ensemble consensus (FinBERT + DistilBERT + RoBERTa), retrain on combined dataset.

**The lesson**: Pseudo-labeling works when initial models are strong (>75% baseline). Ensemble filtering creates higher-quality synthetic labels than single-annotator human labels by requiring multi-model agreement—effectively multiple "expert" validators.

**Key technologies**: HuggingFace Transformers, PyTorch, FinBERT, DistilBERT, RoBERTa, scikit-learn

**Course**: UC Berkeley MIDS W266 (Natural Language Processing)

[→ Explore the NLP project](./nlp_demo/)

---

### 2. Computer Vision Demo: When Deep Learning Fails and Classical Features Win

**The surprising result**: Logistic regression on hand-crafted features (HOG, Fourier transforms, Local Binary Patterns) achieved 80.1% F1-score while DenseNet121 transfer learning barely exceeded random guessing at 40.1%.

**The constraint**: Medical chest X-ray classification with only 1,000 training examples per class—insufficient for deep learning but common in clinical settings with rare diseases or small hospital datasets.

**The solution**: Engineer domain-specific features capturing radiological patterns (HOG for structural changes, Fourier for calcifications, LBP for texture pathology, spatial features for anatomical constraints), reduce to 410 PCA components, train interpretable classifier.

**The lesson**: Transfer learning fails when source domain (ImageNet: color photos of everyday objects) misaligns with target domain (medical: grayscale tissue textures). With <10K examples, invest effort in feature engineering, not hyperparameter tuning of data-hungry deep models.

**Key technologies**: Scikit-learn, OpenCV, DenseNet121, ResNet50, EfficientNetB0, Keras Tuner, Classical CV features

**Course**: UC Berkeley MIDS W281 (Computer Vision)

[→ Explore the Computer Vision project](./computer_vision_demo/)

---

### 3. RAG Demo: Building Enterprise Q&A When External LLMs Violate Compliance

**The operational requirement**: Organizations handling HIPAA, GDPR, or FedRAMP data cannot send queries to external LLM APIs—yet still need AI-powered search over internal documentation.

**The constraint**: Deploy Retrieval-Augmented Generation entirely within enterprise security boundaries while serving distinct user populations (technical engineers vs. marketing teams) with different information needs.

**The solution**: Build production RAG pipeline with local Mistral-7B (4-bit quantized for resource efficiency) and Cohere API comparison, Qdrant vector store, multi-qa-mpnet embeddings, persona-based prompting for audience adaptation.

**The lesson**: Chunk size (256 tokens, k=8 retrieval) dominated system performance more than LLM choice. Simpler personas that stay close to source material outperformed elaborate prompts attempting to add context. Multi-metric evaluation (ROUGE-L, Semantic Similarity, BLEU, BERTScore) essential—no single metric captures answer quality.

**Key technologies**: LangChain, Qdrant, Mistral-7B, Cohere API, Sentence Transformers, PEFT/LoRA

**Course**: UC Berkeley MIDS W266 (Generative AI)

[→ Explore the RAG project](./rag_demo/)

---

### 4. Cloud App Demo: Full End-to-End ML API Deployment at Production Scale

**The production target**: Deploy sentiment analysis API on AWS Kubernetes (EKS) serving 10 requests/second with p(99) latency <2 seconds under load testing, autoscaling during traffic spikes, Redis caching to prevent abuse.

**The constraint**: Real production systems require more than model training—containerization, orchestration, caching, monitoring, load testing, multi-service routing, resource optimization.

**The solution**: Package DistilBERT model with FastAPI into Docker container baked with model weights (no runtime downloads), deploy to EKS with Horizontal Pod Autoscaling, implement Redis caching layer, expose via Istio virtual service with path-based routing, validate with k6 load testing and Grafana monitoring.

**The lesson**: Production ML is 10% model, 90% infrastructure. Model size determines resource requirements (1GB DistilBERT model → adjust pod limits accordingly). Baking models into images critical for fast autoscaling (vs. pulling from HuggingFace at runtime). Pay-per-use deployment aligns with spiky workloads better than reserved capacity.

**Key technologies**: FastAPI, Docker, Kubernetes, AWS EKS/ECR, Istio, Redis, k6, Grafana, Poetry

**Course**: UC Berkeley MIDS W255 (MLOps)

[→ Explore the Cloud Deployment project](./cloud_app_demo/)

---

## Cross-Project Insights: Engineering Judgment Over Trend-Following

### 1. Sample Efficiency Determines Architecture Choice

**Pattern observed**: Classical ML on engineered features outperformed deep learning with <10K examples across both CV and NLP projects.

**Computer Vision**: 1,000 examples/class → Logistic regression (80% F1) beat DenseNet121 (40% F1)

**NLP**: 7,000 labeled examples → Pseudo-labeling necessary to scale; pure supervised learning plateaued

**Actionable principle**: Don't default to deep learning. Evaluate training data size first. Below 10K examples per class, consider classical ML + feature engineering or data augmentation strategies (pseudo-labeling, augmentation, transfer from aligned domains).

### 2. Domain Alignment Trumps Model Sophistication

**Pattern observed**: Mismatched source-target domains cause catastrophic transfer learning failure, regardless of source model quality.

**Computer Vision**: ImageNet (color objects) → Medical X-rays (grayscale textures) = 40% F1 failure

**NLP**: Generic BERT → Financial language = poor baseline; FinBERT (financial pre-training) = strong baseline enabling pseudo-labeling

**RAG**: General embeddings → Specialized QA task = chose multi-qa-mpnet (QA-optimized) over generic sentence transformers

**Actionable principle**: Prioritize domain-aligned pre-training over state-of-the-art generic models. FinBERT (smaller, specialized) beats BERT-large (bigger, generic) for financial text. Medical-specific pre-training essential before transfer learning on X-rays.

### 3. Production Constraints Drive Real Decisions

**Pattern observed**: Academic benchmarks optimize for accuracy; production systems balance accuracy, latency, cost, interpretability, compliance.

**Computer Vision**: 2-second CPU training + <1ms inference (Logistic Regression) beats 400-second GPU training + 50ms inference (DenseNet121) for hospital deployment—despite similar accuracy

**RAG**: Pay-per-use (on-demand) LLM calls aligned with spiky workload (0.9 QPM average, 25-40 QPM quarterly peaks) better than reserved capacity (97% idle time)

**Cloud App**: Baking 1GB model into container image (longer build, faster autoscaling) beats runtime downloads (fast build, slow pod startup during traffic spikes)

**Actionable principle**: Define deployment constraints upfront (latency SLA, hardware availability, cost budget, interpretability requirements). Optimize for production reality, not leaderboard rankings.

### 4. Multi-Metric Evaluation Reveals Hidden Failures

**Pattern observed**: Single metrics miss critical quality dimensions—especially for generation tasks.

**RAG**: ROUGE-L rewards fidelity but misses paraphrasing; Semantic Similarity rewards meaning but overlooks factual precision; BLEU detects verbatim copying; BERTScore balances both. Weighted composite score (0.35 semantic + 0.30 BERT + 0.25 ROUGE + 0.10 BLEU) aligned better with human judgment.

**NLP**: Class-level metrics revealed ensemble pseudo-labeling reduced negative sentiment misclassification by 50%—invisible in overall accuracy (80.6% unfiltered vs. 80.0% ensemble).

**Computer Vision**: Confusion matrix showed Cardiomegaly→Aortic Enlargement errors (clinically acceptable, both require cardiac workup) vs. Pleural Thickening→No Finding errors (clinically harmful, delays diagnosis).

**Actionable principle**: Use multiple complementary metrics calibrated to task requirements. For imbalanced classification, report per-class F1 and confusion matrices. For generation, combine n-gram overlap + semantic similarity + domain-specific validators.

### 5. Interpretability Enables Validation and Trust

**Pattern observed**: Stakeholders (clinicians, compliance officers, end users) won't adopt black-box models they can't validate.

**Computer Vision**: Logistic regression coefficients showed which features (HOG gradients in upper mediastinum, LBP texture in lung bases) drove decisions—enabling radiologist validation. DenseNet121's 8M parameter black box failed to gain trust despite (hypothetically) comparable accuracy.

**NLP**: Tracking which specific training examples influenced pseudo-labels allowed identifying when ensemble filtering removed noisy examples vs. useful rare-class instances.

**RAG**: Citation mechanism (return source documents with answers) critical for detecting when LLM answered from training data vs. retrieved context—hallucination detection requirement for compliance environments.

**Actionable principle**: Build hybrid systems where complex models extract features and interpretable classifiers make final decisions. Implement attribution mechanisms (attention weights, SHAP values, source citations) for auditing.

## Tech Stack Breadth: Full-Stack ML Engineering

### Languages & Core Frameworks
- **Python 3.9+**: Primary language across all projects
- **PyTorch**: Deep learning framework (NLP sentiment classification)
- **TensorFlow/Keras**: Computer vision transfer learning
- **Scikit-learn**: Classical ML, evaluation metrics, pipelines

### NLP & Language Models
- **HuggingFace Transformers**: Model hub and training pipelines
- **FinBERT, DistilBERT, RoBERTa**: Ensemble pseudo-labeling
- **Mistral-7B**: Local LLM for enterprise RAG
- **Cohere API**: Proprietary LLM comparison
- **LangChain**: RAG orchestration framework
- **Sentence Transformers**: Embedding models (multi-qa-mpnet)

### Computer Vision
- **OpenCV**: Image preprocessing and manipulation
- **Classical features**: HOG, LBP, Fourier transforms, image pyramids
- **Transfer learning**: DenseNet121, ResNet50, EfficientNetB0
- **Keras Tuner**: Hyperparameter optimization

### Cloud & Deployment Infrastructure
- **Docker**: Containerization with multi-stage builds
- **Kubernetes**: Orchestration (EKS deployment)
- **AWS**: EKS (managed Kubernetes), ECR (container registry)
- **Istio**: Service mesh, virtual service routing
- **Redis**: Caching layer for API abuse prevention
- **FastAPI**: High-performance API framework
- **Poetry**: Dependency management

### Evaluation & Monitoring
- **k6**: Load testing (sustained 10 req/sec validation)
- **Grafana**: Real-time system monitoring
- **pytest**: Unit and integration testing
- **ROUGE, BLEU, BERTScore**: NLP evaluation metrics
- **scikit-learn metrics**: Classification reports, confusion matrices, ROC-AUC

### Development Tools
- **Git + Git LFS**: Version control for large model files
- **Jupyter**: Exploratory analysis and experimentation
- **TensorBoard**: Training visualization
- **VS Code**: Primary IDE

## Repository Structure

This repository uses **git submodules** to organize independent demonstration projects while maintaining a cohesive portfolio.

```
project_demos/
├── README.md                    # This file - narrative overview
├── PROJECT_STRUCTURE.md         # Technical reference and submodule management
├── .gitmodules                  # Submodule configuration
├── nlp_demo/                    # Financial sentiment with pseudo-labeling
├── computer_vision_demo/        # Medical X-ray classification
├── rag_demo/                    # Enterprise RAG system
└── cloud_app_demo/              # Production ML API deployment
```

Each submodule contains:
- Complete standalone project with dedicated repository
- Comprehensive README with narrative structure
- Jupyter notebook with full implementation
- Requirements/dependencies specification
- Project-specific documentation (PDF reports, presentations)

**Getting started**:
```bash
# Clone with all submodules
git clone --recursive https://github.com/gibbons-tony/project_demos.git

# Or initialize submodules after cloning
git submodule update --init --recursive

# Navigate into any project
cd nlp_demo
# Follow project-specific README instructions
```

For detailed submodule management, branch strategy, and maintenance procedures, see [PROJECT_STRUCTURE.md](./PROJECT_STRUCTURE.md).

## Key Takeaways: Versatility Over Specialization

### Technical Principles

**1. No universal architecture**: Deep learning excels with abundant data and aligned domains. Classical ML wins in low-data, high-interpretability, resource-constrained scenarios. Choosing correctly requires understanding constraints, not following trends.

**2. Domain knowledge is a feature, not a crutch**: Engineered features encoding expert knowledge (radiological patterns, financial sentiment cues, anatomical constraints) outperform generic learned representations when training data is scarce.

**3. Evaluation drives improvement**: Multi-metric assessment reveals failure modes invisible to single metrics. Per-class analysis, confusion matrices, and error categorization guide targeted improvements.

**4. Production is the real test**: Academic benchmarks optimize for accuracy on curated test sets. Production systems balance accuracy, latency, cost, interpretability, scalability, and compliance—often requiring architectural trade-offs invisible in research.

**5. Infrastructure is half the battle**: Model training is 10-20% of ML engineering effort. Deployment, monitoring, caching, testing, containerization, orchestration, and scaling consume the rest. Production-ready ML requires full-stack engineering.

### Operational Insights

**1. Start with constraints, not algorithms**: Define latency requirements, hardware availability, interpretability needs, and cost budgets before selecting architectures. Constraints eliminate 80% of options immediately.

**2. Prototype fast, validate rigorously**: Jupyter notebooks enable rapid experimentation. But production deployment requires pytest coverage, load testing, monitoring, and failure mode analysis—validate early.

**3. Domain alignment > model size**: A smaller model pre-trained on domain-specific data (FinBERT for finance, medical pre-training for X-rays) outperforms a larger generic model. Prioritize source-target alignment over parameter count.

**4. Build hybrid systems**: Combine strengths of different approaches—classical features + deep learning features, ensemble filtering for pseudo-labels, interpretable classifiers on neural network embeddings.

**5. Monitor what matters**: Track per-class metrics for imbalanced data, p(99) latency not average latency, cache hit rates not just throughput, hallucination detection not just fluency.

## Future Directions

### Immediate Extensions
- **Medical CV**: Implement GradCAM attention visualization for DenseNet121 to achieve interpretability while preserving deep learning's representational capacity
- **NLP Pseudo-labeling**: Explore active learning integration—query human annotators specifically for low-confidence examples where ensemble disagrees
- **RAG**: Fine-tune Mistral-7B with LoRA adapters for engineering vs. marketing personas instead of prompt-based adaptation
- **Cloud App**: Implement A/B testing framework to evaluate model updates with real user traffic before full rollout

### Long-Term Research Questions
- **Cross-domain transfer**: Can models trained on one MIDS project transfer to another? (e.g., does medical image feature extraction help with financial document analysis?)
- **Meta-learning for architecture selection**: Train meta-model to predict which approach (deep learning vs. classical ML) will succeed given dataset characteristics
- **Automated feature engineering**: Explore neural architecture search for domain-specific feature extractors that match hand-crafted performance
- **Federated learning**: Enable model training across distributed datasets (multiple hospitals, financial institutions) without centralized data sharing

---

**Built by**: Tony Gibbons
**Program**: UC Berkeley, Master of Information and Data Science (MIDS)
**Focus**: Full-stack machine learning engineering—from research to production deployment
**Projects**: 4 production-oriented demonstrations spanning NLP, Computer Vision, Generative AI, and Cloud MLOps
**Key Philosophy**: Engineering judgment over trend-following. Match architectures to constraints, not headlines.

**Contact**: [GitHub](https://github.com/gibbons-tony)
