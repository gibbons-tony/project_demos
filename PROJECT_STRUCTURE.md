# Project Structure

This repository is organized as a collection of git submodules, each representing a distinct machine learning demonstration project from UC Berkeley MIDS coursework.

## Repository Organization

```
project_demos/
├── README.md                    # Main narrative documentation
├── PROJECT_STRUCTURE.md         # This file - technical structure reference
├── .gitmodules                  # Git submodule configuration
├── cloud_app_demo/              # Full-stack ML deployment (submodule)
├── nlp_demo/                    # NLP with pseudo-labeling (submodule)
├── computer_vision_demo/        # Medical image classification (submodule)
└── rag_demo/                    # Enterprise RAG system (submodule)
```

## Submodule Details

### cloud_app_demo
- **Repository**: https://github.com/gibbons-tony/cloud_app_demo.git
- **Focus**: Full end-to-end machine learning API deployment
- **Key Technologies**: FastAPI, Kubernetes, Docker, Redis, HuggingFace, AWS EKS
- **Course**: UC Berkeley MIDS W255 (MLOps)

### nlp_demo
- **Repository**: https://github.com/gibbons-tony/nlp_demo.git
- **Focus**: Financial sentiment analysis with pseudo-labeling
- **Key Technologies**: BERT, DistilBERT, RoBERTa, PyTorch, HuggingFace Transformers
- **Course**: UC Berkeley MIDS W266 (Natural Language Processing)

### computer_vision_demo
- **Repository**: https://github.com/gibbons-tony/computer_vision_demo.git
- **Focus**: Medical X-ray classification using classical features vs. transfer learning
- **Key Technologies**: DenseNet121, ResNet50, scikit-learn, OpenCV, HOG/LBP/Fourier features
- **Course**: UC Berkeley MIDS W281 (Computer Vision)

### rag_demo
- **Repository**: https://github.com/gibbons-tony/rag_demo.git
- **Focus**: Enterprise RAG system for internal Q&A
- **Key Technologies**: LangChain, Qdrant, Mistral-7B, Cohere API, Sentence Transformers
- **Course**: UC Berkeley MIDS W266 (Generative AI)

## Working with Submodules

### Initial Clone
```bash
# Clone with all submodules
git clone --recursive https://github.com/gibbons-tony/project_demos.git

# Or if already cloned, initialize submodules
git submodule update --init --recursive
```

### Updating Submodules
```bash
# Update all submodules to latest commit
git submodule update --remote --merge

# Update specific submodule
git submodule update --remote cloud_app_demo
```

### Making Changes to Submodules
```bash
# Navigate into submodule
cd cloud_app_demo

# Make changes, commit, push as normal
git add .
git commit -m "Update documentation"
git push origin main

# Return to parent repo and commit submodule pointer update
cd ..
git add cloud_app_demo
git commit -m "Update cloud_app_demo submodule"
git push
```

## Branch Strategy

### Parent Repository (project_demos)
- **main**: Stable release branch with finalized documentation
- **claude/*****: Development branches for documentation and structure updates

### Submodules
Each submodule maintains its own branching strategy. The parent repository references specific commits from the submodule's main/master branch.

## Documentation Standards

Each submodule contains its own comprehensive README.md following a narrative structure:

1. **Compelling title** - Highlighting key finding or question
2. **Opening hook** - Core insight and context
3. **Problem statement** - Why the work matters
4. **Methodology** - Technical approach and implementation
5. **Results** - Findings and performance metrics
6. **Lessons learned** - Key takeaways and insights
7. **Repository structure** - Code organization
8. **Running instructions** - How to reproduce results
9. **Tech stack** - Tools and dependencies

## Tech Stack Overview

### Languages
- Python 3.9+

### ML Frameworks
- PyTorch
- TensorFlow/Keras
- Scikit-learn
- HuggingFace Transformers

### Deployment & Infrastructure
- Docker
- Kubernetes
- AWS (EKS, ECR)
- Istio
- Redis

### NLP & LLMs
- BERT variants (FinBERT, DistilBERT, RoBERTa)
- Mistral-7B
- Cohere API
- LangChain
- Sentence Transformers

### Computer Vision
- OpenCV
- Classical features (HOG, LBP, Fourier)
- Transfer learning (DenseNet121, ResNet50, EfficientNetB0)

### Development Tools
- Poetry (dependency management)
- pytest (testing)
- Jupyter notebooks
- Git LFS (large file storage)
- k6 (load testing)
- Grafana (monitoring)

## Maintenance

### Adding New Submodules
```bash
# Add new demo project as submodule
git submodule add https://github.com/gibbons-tony/new_demo.git new_demo

# Commit the addition
git add .gitmodules new_demo
git commit -m "Add new_demo submodule"
git push
```

### Removing Submodules
```bash
# Remove submodule
git submodule deinit -f path/to/submodule
git rm -f path/to/submodule
rm -rf .git/modules/path/to/submodule

# Commit the removal
git commit -m "Remove submodule"
git push
```

## Contact

**Author**: Tony Gibbons
**Institution**: UC Berkeley, Master of Information and Data Science (MIDS)
**GitHub**: https://github.com/gibbons-tony
