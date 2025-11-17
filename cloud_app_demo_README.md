# Production ML is 10% Model, 90% Infrastructure: Deploying Sentiment Analysis at Scale

**The counterintuitive reality**: The model trained in minutes—5 minutes on dual A4000 GPUs for transfer learning. The production system took 10 hours to build: containerization, orchestration, caching, monitoring, load testing, autoscaling, and multi-service routing. This project reveals that deploying real ML APIs requires full-stack engineering skills far beyond model training.

## The Production Challenge

Academic ML ends with model accuracy metrics on test sets. Production ML begins there: How do you serve a 1GB model at 10 requests/second with <2s p(99) latency? How do you autoscale during traffic spikes without 60-second pod startup delays? How do you prevent API abuse while maintaining cache hit rates above 95%? How do you route multiple services (lab + project) through a single gateway?

This project demonstrates end-to-end ML deployment on AWS Kubernetes (EKS), exposing a DistilBERT sentiment analysis API that meets production SLAs while handling the operational complexity invisible in research papers.

## System Architecture

**The production stack**: FastAPI application serving DistilBERT sentiment predictions, packaged in Docker containers with baked-in model weights (no runtime downloads), deployed to AWS EKS with Horizontal Pod Autoscaling, Redis caching layer for abuse prevention, Istio service mesh for path-based routing, k6 load testing for validation, and Grafana dashboards for real-time monitoring.

**Key design decisions**:

- **Model baking vs. runtime loading**: Baking the 1GB DistilBERT model into the Docker image adds 2 minutes to build time but enables <10s pod startup during autoscaling. Alternative (pull from HuggingFace at runtime) creates 60+ second delays when new pods spawn under load—unacceptable when k6 is hammering the endpoint.

- **Init containers for dependency verification**: Before the FastAPI container starts, init containers verify Redis DNS resolution and readiness. This prevents race conditions where the API tries to cache results before Redis is available, causing cascading failures.

- **Kustomize overlays for environment management**: Base Kubernetes manifests define core resources (deployments, services). Overlays add environment-specific configurations—LoadBalancer service for local Minikube development, HorizontalPodAutoscaler + VirtualService for AWS EKS production. Single source of truth, no duplicate YAML.

- **Path-based routing with Istio**: A single VirtualService routes `/lab` traffic to the lab prediction service and `/project` traffic to the project prediction service. This demonstrates multi-service architectures where different models/versions coexist behind one gateway—critical for A/B testing and gradual rollouts.

## The Model: DistilBERT for Efficient Sentiment Analysis

**Why DistilBERT**: DistilBERT retains 97% of BERT's language understanding with 40% fewer parameters (66M vs 110M) and 60% faster inference. For CPU-based deployment (no GPU on EKS pods), this difference determines whether you hit latency SLAs or burn budget on oversized instances.

**Model specifics**:
- **Base architecture**: [DistilBERT base uncased](https://arxiv.org/abs/1910.01108) fine-tuned on SST-2 (Stanford Sentiment Treebank)
- **Training**: 5 minutes on 2x NVIDIA A4000 GPUs (batch size 256, 15GB memory per GPU)
- **Model size**: ~500MB (model weights) + ~500MB (PyTorch dependencies) = 1GB total image overhead
- **Inference**: Handles sequences up to 512 tokens, returns POSITIVE/NEGATIVE labels with confidence scores
- **Source**: Hosted on [HuggingFace](https://huggingface.co/winegarj/distilbert-base-uncased-finetuned-sst2) for versioning and artifact storage

**API endpoints**:
- `GET /project/health` - Readiness/liveness probe for Kubernetes
- `POST /project/predict` - Single/batch sentiment prediction with Redis caching
- `POST /project/bulk-predict` - Optimized batch prediction endpoint

**Input format**:
```json
{
    "text": ["example 1", "example 2"]
}
```

**Output format**:
```json
{
    "predictions": [
        [
            {"label": "POSITIVE", "score": 0.7127904295921326},
            {"label": "NEGATIVE", "score": 0.2872096002101898}
        ],
        [
            {"label": "POSITIVE", "score": 0.7186233401298523},
            {"label": "NEGATIVE", "score": 0.2813767194747925}
        ]
    ]
}
```

## Production Requirements and Validation

**Performance SLAs**:
- **Throughput**: 10 requests/second sustained for 10 minutes
- **Latency**: p(99) < 2 seconds under load (10 virtual users)
- **Cache hit rate**: >95% (demonstrates Redis effectiveness)
- **Autoscaling**: Pods must scale horizontally based on CPU/memory thresholds

**Load testing with k6**: Simulates realistic user traffic patterns (ramp-up, sustained load, ramp-down) while measuring percentile latencies (p90, p95, p99, p99.99). The p(99) metric matters more than average latency—it captures the worst user experience for 1% of requests, critical for SLA compliance.

**Monitoring with Grafana**: Real-time dashboards track pod CPU/memory utilization, request rates, cache hit ratios, and response times. During k6 execution, you observe autoscaling behavior: pod count increasing as CPU hits thresholds, latency spikes during new pod startup, then stabilization as load distributes.

## Critical Lessons: Infrastructure Complexity

### 1. Model Size Determines Resource Requirements

**The problem**: Lab model (1MB) vs Project model (1GB)—a 1000x difference. Initial pod resource limits (512MB memory) caused OOMKilled errors when loading DistilBERT + PyTorch. Overcompensating with 4GB limits wasted money on unused capacity.

**The solution**: Profile actual memory usage (`docker stats` during local testing), then set limits with 20% headroom. For this project: ~1.2GB total → set 1.5GB limit. Right-sizing prevents both failures and waste.

### 2. Autoscaling Speed Requires Model Baking

**The problem**: If the model downloads from HuggingFace at pod startup (best practice for model versioning), each new pod takes 60+ seconds to become ready—500MB download + decompression + model initialization. During k6 load spikes, existing pods max out while new pods are still initializing, causing latency violations.

**The solution**: Bake the model into the Docker image during build. Trade-off: Longer build times (2 extra minutes) and larger images (1GB vs 100MB), but pods start in <10s. For production with autoscaling, startup speed > build efficiency.

**Better long-term solution**: Mount model from shared persistent storage (EFS on AWS). All pods share one copy, model updates don't require image rebuilds, and scaling is fast. But this adds complexity (PersistentVolumeClaims, storage costs, permissions) beyond the scope of this project.

### 3. Caching Prevents Abuse and Reduces Latency

**The problem**: Inference costs ~200ms CPU time per request. Without caching, duplicate requests (users retrying, testing, or malicious abuse) waste compute. At 10 req/sec with 50% duplicates, you're burning 5x unnecessary resources.

**The solution**: Redis caching with hash-based keys (`hash(text_input) → prediction`). Cache hits return in <5ms vs 200ms for model inference. At >95% cache hit rate (achievable after warm-up period), effective throughput increases 20x for the same hardware.

**Design choice**: Cache predictions, not input embeddings. Embeddings are model-specific and large (768 dimensions for DistilBERT). Predictions are 2 labels + scores (~100 bytes). Storage efficiency matters when caching millions of requests.

### 4. Multi-Service Routing Enables Gradual Rollout

**The problem**: You can't take down production to deploy a new model. You need blue-green deployments, canary releases, or A/B testing—all require multiple service versions coexisting.

**The solution**: Istio VirtualService with path-based routing. `/lab` routes to the lab service, `/project` routes to the project service. Extend this pattern:
- `/project/v1` → old model (90% traffic)
- `/project/v2` → new model (10% traffic)
Monitor metrics, shift traffic percentages gradually, rollback if errors spike.

**Production benefit**: Deploy with confidence. No "hope and pray" full releases. Controlled experiments with real user traffic inform whether model changes actually improve production outcomes.

### 5. Init Containers Prevent Race Conditions

**The problem**: Kubernetes starts all containers in a pod simultaneously. If the FastAPI container initializes before Redis is ready, cache writes fail, errors propagate, and the pod crashes. Manually restarting pods is not scalable.

**The solution**: Init containers run sequentially before main containers. First init container verifies Redis DNS resolution (`nslookup redis-service`). Second init container polls Redis readiness (`redis-cli ping`). Only when both succeed does the FastAPI container start. Guaranteed safe startup order.

## Repository Structure

```
cloud_app_demo/
├── ASSIGNMENT_SPEC.md          # Original project requirements and grading rubric
├── README.md                   # This narrative documentation
├── mlapi/                      # FastAPI application code
│   ├── main.py                 # API endpoints and Redis caching logic
│   ├── models.py               # Pydantic input/output models
│   ├── predictor.py            # DistilBERT model loading and inference
│   ├── example.py              # Model usage examples
│   └── test_mlapi.py           # Pytest test suite
├── Dockerfile                  # Multi-stage build with model baking
├── pyproject.toml              # Poetry dependency specification
├── poetry.lock                 # Locked dependency versions
├── .k8s/                       # Kubernetes manifests
│   ├── base/                   # Base resources (deployments, services)
│   │   ├── deployment-mlapi.yaml
│   │   ├── deployment-redis.yaml
│   │   ├── service-mlapi.yaml
│   │   ├── service-redis.yaml
│   │   └── kustomization.yaml
│   └── overlays/
│       ├── dev/                # Minikube configuration
│       │   ├── kustomization.yaml
│       │   ├── namespace.yaml
│       │   └── service-mlapi-lb.yaml
│       └── prod/               # AWS EKS configuration
│           ├── kustomization.yaml
│           ├── hpa-mlapi.yaml          # Horizontal Pod Autoscaler
│           └── virtual-service.yaml    # Istio routing rules
├── build-push.sh               # Build Docker image and push to ECR
├── deploy.sh                   # Deploy to Minikube (dev) or EKS (prod)
├── load.js                     # k6 load testing script
└── distilbert-base-uncased-finetuned-sst2/  # Model files (gitignored)
```

## Running the Project

### Prerequisites

**Required tools**:
- Docker Desktop
- Kubernetes CLI (`kubectl`)
- Minikube (local testing)
- AWS CLI configured with `ucberkeley-student` profile
- k6 (load testing)
- Git LFS (for model download)

**System requirements**:
- ARM64 architecture (PyTorch no longer supports Intel Macs; Windows ARM also unsupported)
- 8GB+ RAM (Docker needs 4GB, Minikube needs 4GB)
- 10GB disk space (model + images + dependencies)

### Step 1: Clone and Set Up

```bash
# Clone repository
git clone https://github.com/gibbons-tony/cloud_app_demo.git
cd cloud_app_demo

# Install dependencies
poetry install  # Do NOT run `poetry update` - torch resolution takes hours

# Download model from HuggingFace (requires git-lfs)
git lfs install
git clone https://huggingface.co/winegarj/distilbert-base-uncased-finetuned-sst2

# Verify model downloaded (should show ~500MB)
du -sh distilbert-base-uncased-finetuned-sst2/
```

### Step 2: Local Development Testing

```bash
# Run pytest tests
poetry run pytest mlapi/test_mlapi.py -v

# Start local API server
poetry run uvicorn mlapi.main:app --reload

# Test endpoints (in separate terminal)
curl http://localhost:8000/project/health

curl -X POST http://localhost:8000/project/predict \
  -H "Content-Type: application/json" \
  -d '{"text": ["This movie was fantastic!", "Terrible waste of time"]}'
```

### Step 3: Deploy to Minikube (Local Kubernetes)

```bash
# Start Minikube with sufficient resources
minikube start --cpus 4 --memory 4096

# Deploy application
./deploy.sh

# The script will:
# 1. Switch to Minikube context
# 2. Build Docker image locally
# 3. Apply Kubernetes manifests
# 4. Start Minikube tunnel (keep terminal open)

# Get service URL (in new terminal)
minikube service mlapi-service -n w255 --url

# Test deployed endpoint
curl <service-url>/project/health
curl -X POST <service-url>/project/predict \
  -H "Content-Type: application/json" \
  -d '{"text": ["Great product!"]}'
```

### Step 4: Deploy to AWS EKS (Production)

```bash
# Authenticate to AWS and EKS
aws sso login --profile ucberkeley-sso
aws eks update-kubeconfig --name eks-datasci255-students --profile ucberkeley-student

# Authenticate to ECR (Elastic Container Registry)
aws ecr get-login-password --region us-west-2 --profile ucberkeley-student | \
  docker login --username AWS --password-stdin 650251712107.dkr.ecr.us-west-2.amazonaws.com

# Build and push image to ECR
./build-push.sh  # Tags image with git commit hash, pushes to ECR

# Deploy to EKS
./deploy.sh prod

# Verify deployment
kubectl get pods -n <your-namespace>
kubectl get svc -n <your-namespace>
kubectl get hpa -n <your-namespace>  # Check autoscaler
```

### Step 5: Load Testing and Monitoring

```bash
# Set your EKS namespace
export NAMESPACE=gibbons-tony  # Replace with your namespace

# Run k6 load test (10 minute sustained load)
k6 run -e NAMESPACE=${NAMESPACE} \
  --summary-trend-stats "min,avg,med,max,p(90),p(95),p(99),p(99.99)" \
  load.js

# Monitor in Grafana (separate terminal)
kubectl port-forward -n prometheus svc/grafana 3000:3000
# Open browser to http://localhost:3000
# Navigate to dashboards → Kubernetes → Workload metrics
# Watch pod scaling, CPU/memory usage, request rates during load test
```

**What to observe during k6 testing**:
- Initial latency spike as cache warms up
- CPU utilization increasing on initial pod
- HorizontalPodAutoscaler triggering (new pods spawning at CPU >70%)
- Brief latency increase as new pods initialize
- Latency stabilization as load distributes across multiple pods
- Cache hit rate climbing to >95%
- p(99) latency dropping below 2s threshold

### Cleaning Up

```bash
# Local Minikube
kubectl delete namespace w255
minikube stop

# AWS EKS
kubectl delete -k .k8s/overlays/prod
```

## Tech Stack

**Application Framework**:
- **FastAPI**: Async Python web framework (automatic OpenAPI docs, type validation)
- **Pydantic**: Data validation using Python type hints
- **Poetry**: Dependency management and packaging

**ML Model**:
- **HuggingFace Transformers**: Model loading and inference pipelines
- **PyTorch**: Deep learning framework (CPU inference)
- **DistilBERT**: Efficient BERT variant (66M parameters, 40% faster)

**Infrastructure**:
- **Docker**: Containerization with multi-stage builds
- **Kubernetes**: Orchestration (local Minikube + AWS EKS)
- **Kustomize**: Environment-specific configuration management
- **Istio**: Service mesh for routing and traffic management
- **Redis**: In-memory caching for prediction results

**AWS Services**:
- **EKS**: Managed Kubernetes control plane
- **ECR**: Docker container registry
- **IAM**: Authentication and authorization via SSO

**Testing & Monitoring**:
- **pytest**: Unit and integration testing
- **k6**: Load testing and performance validation
- **Grafana**: Real-time metrics dashboards
- **Prometheus**: Metrics collection and storage

## Key Takeaways: Bridging Research and Production

**1. Infrastructure skills matter as much as model skills**: You can train the best sentiment classifier, but if you can't deploy it with acceptable latency, cache hit rates, and autoscaling behavior, it's useless in production. ML engineering requires full-stack capabilities.

**2. Production constraints drive architectural decisions**: Model baking vs. runtime loading, resource limit tuning, caching strategies, init container ordering—none of these appear in model training notebooks. They emerge from real SLAs, cost budgets, and operational requirements.

**3. Observability enables iteration**: Without Grafana dashboards showing pod scaling, cache hit rates, and percentile latencies, you're flying blind. You can't optimize what you can't measure. Instrumentation is not optional.

**4. Kubernetes is complex but necessary at scale**: Managing deployments, services, autoscalers, and routing manually is error-prone. Kubernetes abstracts infrastructure concerns (self-healing, scaling, load balancing) but requires significant learning investment. For production ML, it's worth it.

**5. Academic metrics ≠ production metrics**: Test set accuracy doesn't predict p(99) latency under load. Training loss doesn't correlate with cache hit rates. F1-score doesn't measure pod resource efficiency. Production requires an entirely different evaluation framework.

---

**Project**: UC Berkeley MIDS W255 (Machine Learning Operations)
**Author**: Tony Gibbons
**Focus**: End-to-end ML deployment—containerization, Kubernetes orchestration, autoscaling, caching, monitoring
**Key Technologies**: FastAPI, DistilBERT, Docker, Kubernetes, Redis, Istio, AWS EKS, k6, Grafana

For technical implementation details, see [ASSIGNMENT_SPEC.md](./ASSIGNMENT_SPEC.md).
For deployment configurations, see [mlapi/README.md](./mlapi/README.md) (if available).
