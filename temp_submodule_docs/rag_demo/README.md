# Enterprise RAG: Building Internal Q&A Systems When External LLMs Aren't an Option

When your company handles HIPAA-protected patient records, GDPR-regulated customer data, or FedRAMP-certified government information, sending queries to external LLM APIs isn't just inadvisable—it's often illegal. Yet these organizations still need AI-powered search capabilities. This project builds a production-grade Retrieval-Augmented Generation (RAG) system from scratch, designed to operate entirely within enterprise security boundaries while serving distinct user populations with different information needs.

The system addresses a real scenario: enabling both technical engineers (300 users) and marketing teams (40 users) to query internal AI/ML documentation through a custom RAG pipeline that adapts its responses based on audience. Rather than simply demonstrating "RAG works," this implementation rigorously evaluates the engineering decisions that matter in production: which LLM performs better for internal deployment, how chunking strategy affects answer quality, and whether persona-based prompting can replace costly model fine-tuning.

## Why 256-Token Chunks Beat Your LLM Choice

The most consequential finding from this implementation wasn't about model selection—it was about information architecture. After systematic evaluation across 78 validation questions using four complementary metrics (ROUGE-L, Semantic Similarity, BLEU, BERTScore), chunk size emerged as the dominant factor in system performance.

**Optimal Configuration**: chunk_size=256 tokens, k=8 retrieved documents

This specific configuration creates a "Goldilocks zone" for semantic search: chunks large enough to preserve complete conceptual units (typically 1-2 paragraphs), yet small enough to maintain retrieval precision. The alternative approaches reveal why this balance matters:

- **50-token chunks**: High precision for simple factoid questions, but severe context fragmentation. The LLM receives disjointed sentence fragments, leading to incoherent answers for complex "how" and "why" queries. Increased hallucination risk as the model attempts to bridge gaps between isolated snippets.

- **5000-token chunks**: Maximum context preservation, but the embedding represents averaged meaning across several pages, drastically reducing retrieval precision. The LLM faces a "needle in haystack" problem with 40,000-token contexts (8 chunks × 5000 tokens), often exceeding model limits entirely.

The 256-token sweet spot enabled both tested LLMs to perform competently, while deviations in either direction degraded results regardless of which language model was used.

## Cohere vs. Mistral-7B: Persona Adaptation Matters More Than Model Size

### Performance Results

Evaluated on 78 validation questions spanning both technical (engineering) and accessible (marketing) gold-standard answers:

| Configuration | Weighted Score | ROUGE-L | Semantic Sim | BLEU | BERTScore |
|--------------|----------------|---------|--------------|------|-----------|
| **Cohere + Marketing** | **0.571** | 0.389 | 0.811 | 0.127 | 0.857 |
| Mistral-7B + Marketing | 0.554 | 0.390 | 0.795 | 0.133 | 0.854 |
| Cohere + Engineering | 0.528 | 0.371 | 0.771 | 0.118 | 0.852 |
| Mistral-7B + Engineering | 0.523 | 0.375 | 0.765 | 0.126 | 0.848 |

*Metric weights calibrated to test set: Semantic Similarity (0.35), BERTScore (0.30), ROUGE-L (0.25), BLEU (0.10)*

### Why Marketing Personas Outperformed Engineering

Counter-intuitively, both models achieved higher scores when prompted to generate marketing-oriented responses, even when evaluated against technical gold answers. The marketing persona prompt explicitly instructed the model to:

```
Provide clear, concise answers focusing on key concepts and business
implications. Avoid excessive technical jargon. Make content accessible
while maintaining accuracy.
```

This constraint toward accessibility and conciseness aligned better with the evaluation metrics, which reward semantic overlap and faithful reproduction. The engineering persona, instructed to provide "comprehensive answers with specific implementation details," often generated longer, more elaborate responses that introduced additional technical context not present in the gold answers—reducing metric scores despite being technically correct.

**Key insight**: For internal RAG systems, simpler personas that stay close to source material outperform complex personas that attempt to elaborate or contextualize.

## Production RAG: Beyond the "Store, Retrieve, Generate" Tutorial

### Architecture

```python
# Embeddings: multi-qa-mpnet-base-dot-v1
# Specialized for question-answering tasks, dot-product optimized
embedding_model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')

# Vector Store: Qdrant with in-memory mode for POC
vectorstore = Qdrant.from_documents(
    splits,
    embedding_model,
    location=":memory:",
    collection_name="enterprise_docs"
)

# LLMs: Mistral-7B-Instruct (local) vs Cohere Command (API)
local_llm = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    load_in_4bit=True,  # Quantization for resource efficiency
    device_map="auto"
)

# Retrieval: Top-k similarity search
retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

# Chain: RetrievalQA with persona-injected prompts
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": persona_prompt}
)
```

### Evaluation Framework

The system implements a composite metric strategy combining four complementary dimensions:

1. **ROUGE-L (0.25 weight)**: Measures longest common subsequence between generated and gold answers, capturing answer structure and key phrase inclusion

2. **Semantic Similarity (0.35 weight, highest)**: Cosine similarity of sentence embeddings, evaluating whether the generated answer conveys the same meaning even with different phrasing

3. **BLEU (0.10 weight)**: N-gram precision, traditionally from machine translation, useful for detecting verbatim copying of source material

4. **BERTScore (0.30 weight)**: Contextual embedding similarity at the token level, balancing between strict matching and pure semantic equivalence

This multi-metric approach addresses the fundamental challenge in RAG evaluation: no single metric captures all dimensions of answer quality. ROUGE rewards fidelity to gold answers but misses paraphrasing; semantic similarity rewards meaning but can overlook factual precision; BERTScore balances both but requires calibration.

## Scaling to Production: Cost, Latency, and the Pay-Per-Use Case

### Load Profile Analysis

For the target organization (300 engineers, 40 marketing staff, quarterly product releases):

**Average daily load**: ~0.9 queries per minute (QPM)
- 25% of users active (85 users)
- ~5 queries each over 8-hour workday
- 425 total queries / 480 minutes

**Quarterly peak load**: ~8.0 sustained QPM, bursts to 25-40 QPM
- 75% of users active during release periods (255 users)
- ~15 queries each over 8-hour workday
- 3,825 total queries / 480 minutes

### Deployment Recommendation: Pay-Per-Use

**Reserved LLM instance**: Provisioning for 40 QPM peak capacity results in 97% idle time during average periods—extremely cost-inefficient for this load profile.

**Pay-per-use (on-demand)**: Directly ties cost to consumption. Ideal for spiky, variable workloads with predictable peaks. Minimal cost during low-usage periods, automatic scaling for quarterly bursts.

**Cohere Trial API constraints**: 1,000 calls/month limit requires careful budgeting during POC phase. Production deployment would require upgrading to paid tier or deploying Mistral-7B locally (higher infrastructure cost, but no per-query fees).

## What Fine-Tuning Would Actually Fix

The current system relies on prompt engineering to achieve persona adaptation. Parameter-Efficient Fine-Tuning (PEFT) would address this through specialized model weights rather than runtime instructions.

### Proposed Approach: LoRA (Low-Rank Adaptation)

LoRA, extensively documented in the system's own knowledge base, enables efficient adaptation without full fine-tuning costs or catastrophic forgetting risks. The approach would involve:

**Training data**: Curated question-context-answer triplets
- Question: User query tailored to persona (engineering or marketing)
- Context: Relevant chunks from source documents
- Answer: Expert-crafted gold response in appropriate style

**Implementation**: Create small adapter layers for each persona, loaded on-demand over the base LLM. This enables:
- **Improved persona adherence**: Native generation in required styles without complex prompts
- **Enhanced domain acumen**: Specialized understanding of AI/ML jargon and concepts
- **Increased faithfulness**: Better adherence to source material, reduced hallucination

**Trade-off**: Requires upfront investment in expert-labeled training data and additional model artifacts to maintain, versus the current zero-cost prompt-based approach.

## Limitations and Operational Risks

### Content and Retrieval Constraints

**Knowledge boundary**: The system cannot answer out-of-domain questions. Test question 109 (about a Marvel actor) retrieved completely irrelevant context, demonstrating hard limits of the knowledge base.

**Retrieval failure**: Ambiguous phrasing or embedding model limitations can result in wrong context being retrieved, almost certainly producing incorrect answers.

**Information decay**: Static knowledge base becomes outdated as new research publishes. No mechanism for automatic updates or detecting staleness.

### LLM Generation Risks

**Hallucination with confidence**: The Marvel actor question (109) elicited the correct answer—but this information was *not* in the retrieved context. The model drew from internal training data, creating a false sense of reliability. In a compliance-critical environment, this is unacceptable.

**Faithfulness drift**: The model may subtly misinterpret or distort source material, producing answers that are nuancedly incorrect.

**Bias amplification**: Inherits and potentially amplifies biases present in source documents.

### User and Operational Risks

**Over-reliance**: Users may accept confident-sounding but incorrect answers without critical evaluation, leading to misinformed engineering or marketing decisions.

**Cost management**: Pay-per-use model requires monitoring to prevent unexpected usage spikes from causing budget overruns.

**Model safety**: Mistral-7B-Instruct is not trained for safety—unsafe answers may be generated without filtering.

## Implementation Details

### Tech Stack
- **LangChain**: RAG orchestration framework
- **Qdrant**: Vector database for semantic search
- **Sentence Transformers**: Embedding model library (multi-qa-mpnet-base-dot-v1)
- **Hugging Face Transformers**: Local LLM inference (Mistral-7B)
- **Cohere API**: Proprietary LLM comparison
- **Evaluation**: rouge-score, bert-score, nltk, scikit-learn

### Hyperparameter Search Strategy

Rather than grid search, the evaluation followed an intuition-driven approach:

1. **Chunk size exploration**: Tested [128, 256, 512, 1024] tokens to establish retrieval precision vs. context completeness trade-off
2. **Retrieval depth**: Evaluated k=[4, 8, 12] documents to balance context richness with noise
3. **Persona iteration**: Developed marketing and engineering prompts through iterative refinement based on qualitative answer review
4. **Metric calibration**: Weighted metrics based on observed correlation with human judgment on test subset

This approach prioritizes developing engineering intuition about parameter impact over exhaustive search.

### Repository Structure

```
rag_demo/
├── Assignment_5.ipynb              # Full implementation and experimentation
├── Assignment 5_ RAG System POC.pdf # Detailed findings and recommendations
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── .env.example                     # Configuration template
└── .gitignore                      # Excludes models, data, secrets
```

## Running the System

### Prerequisites
- Python 3.9+
- GPU recommended for local Mistral-7B inference (T4 or better)
- Cohere API key for proprietary model comparison

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env to add COHERE_API_KEY

# Run notebook
jupyter notebook Assignment_5.ipynb
```

### Key Configuration Points
```python
# Optimal configuration from evaluation
CHUNK_SIZE = 256
CHUNK_OVERLAP = 50  # 20% overlap to prevent concept boundary splits
TOP_K = 8
EMBEDDING_MODEL = "multi-qa-mpnet-base-dot-v1"
```

## Lessons Learned: Chunk Size Trumps Model Choice, Simpler Personas Win

### Technical Insights

1. **Chunk size dominates performance**: More impactful than LLM choice within tested range. Architecture decisions matter more than model selection for RAG.

2. **Simpler personas win**: Constrained, concise instructions outperform elaborate personas that attempt to add context or detail beyond source material.

3. **Multi-metric evaluation is essential**: Single metrics miss critical quality dimensions. Weighted composite scores aligned better with human judgment.

4. **Quantization enables local deployment**: 4-bit quantization makes Mistral-7B viable on consumer GPUs without dramatic quality loss.

### Operational Insights

1. **POC constraints drive real decisions**: API rate limits (1,000 Cohere calls/month) forced strategic test allocation, mirroring production budget constraints.

2. **Hallucination detection requires source tracking**: Must implement context citation mechanism to flag when models answer from training data vs. retrieved context.

3. **Persona validation is manual-intensive**: No automated metric reliably detects whether generated tone matches target audience—requires human review.

4. **Pay-per-use aligns with usage patterns**: For spiky workloads with predictable peaks, on-demand pricing beats reserved capacity.

---

**Built for**: UC Berkeley MIDS W266 (Generative AI)
**Scenario**: Enterprise RAG POC for internal Q&A supporting engineering and marketing teams
**Evaluation**: 78 validation questions, 4-metric composite scoring, persona-based prompting
**Key Result**: Cohere + Marketing persona achieved 0.571 weighted score with chunk_size=256, k=8
