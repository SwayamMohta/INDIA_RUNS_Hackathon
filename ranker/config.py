"""
config.py — All JD-specific constants, keyword lists, weights, and thresholds.

This is the single place to tune the scoring system.
Everything is derived from reading the Job Description for:
  Senior AI Engineer @ Redrob (Pune/Noida)
"""

from datetime import date

# ─────────────────────────────────────────────
# EVALUATION DATE
# ─────────────────────────────────────────────
EVAL_DATE = date(2026, 6, 26)

# ─────────────────────────────────────────────
# BM25 RETRIEVAL
# ─────────────────────────────────────────────
BM25_TOP_K = 100000          # score all 100k candidates
FINAL_TOP_N = 100          # final output

# ─────────────────────────────────────────────
# JD QUERY — what the Senior AI Engineer role cares about
# (used for BM25 query)
# ─────────────────────────────────────────────
JD_QUERY = """
machine learning embedding retrieval vector_search ranking recommendation_system
LLM RAG fine_tuning FAISS Milvus Weaviate vector_database approximate_nearest_neighbor
LambdaMART learning_to_rank BM25 Elasticsearch information_retrieval
production deployment inference MLOps evaluation NDCG
Python PyTorch transformer NLP sentence_transformers
"""

# ─────────────────────────────────────────────
# SKILL TAXONOMY — JD-aligned skills with weights
# ─────────────────────────────────────────────

# Core must-have skills (high weight)
CORE_SKILLS = {
    # Embeddings & Semantic Search
    "embedding": 2.0, "embeddings": 2.0, "vector search": 2.5, "vector database": 2.5,
    "semantic search": 2.0, "dense retrieval": 2.0, "ann": 1.5, "approximate nearest neighbor": 2.0,
    "faiss": 2.5, "milvus": 2.5, "weaviate": 2.5, "pinecone": 2.5, "qdrant": 2.0,
    "chroma": 1.5, "chromadb": 1.5,

    # Ranking & IR
    "learning to rank": 3.0, "ltr": 3.0, "lambdamart": 3.0, "ranknet": 2.0, "lambdarank": 2.5,
    "bm25": 2.5, "information retrieval": 2.5, "ranking": 2.0, "reranking": 2.5,
    "elasticsearch": 2.0, "opensearch": 2.0, "solr": 1.5,
    "ndcg": 2.0, "map": 1.5, "mrr": 1.5,  # evaluation metrics knowledge

    # LLMs & RAG
    "llm": 2.0, "large language model": 2.0, "rag": 2.5, "retrieval augmented generation": 2.5,
    "fine-tuning": 2.0, "fine tuning": 2.0, "finetuning": 2.0,
    "lora": 1.5, "rlhf": 2.0, "instruction tuning": 2.0,
    "transformer": 2.0, "bert": 1.5, "gpt": 1.5, "llama": 1.5,

    # ML Foundations
    "machine learning": 1.5, "deep learning": 1.5, "neural network": 1.5,
    "nlp": 2.0, "natural language processing": 2.0,
    "pytorch": 2.0, "tensorflow": 1.5, "hugging face": 2.0, "huggingface": 2.0,

    # Production ML
    "mlops": 2.5, "model serving": 2.5, "model deployment": 2.5, "inference": 2.0,
    "triton": 2.0, "torchserve": 2.0, "bentoml": 1.5, "ray serve": 2.0,
    "feature store": 2.0, "feature engineering": 1.5,
    "a/b testing": 2.0, "ab testing": 2.0, "online evaluation": 2.0,
}

# Shipper verbs — evidence of building/deploying (in career descriptions)
SHIPPER_VERBS = [
    "deployed", "shipped", "built", "architected", "designed", "scaled",
    "optimized", "launched", "productionized", "implemented", "developed",
    "led", "owned", "migrated", "integrated", "automated",
]

# Retrieval-specific signals (for retrieval_experience_score)
RETRIEVAL_KEYWORDS = [
    "faiss", "milvus", "weaviate", "pinecone", "qdrant", "chroma",
    "elasticsearch", "opensearch", "solr", "vector search", "semantic search",
    "ann", "hnsw", "ivf", "approximate nearest neighbor", "dense retrieval",
    "sparse retrieval", "hybrid search", "bm25", "tf-idf", "inverted index",
]

# Ranking-specific signals
RANKING_KEYWORDS = [
    "learning to rank", "ltr", "lambdamart", "ranknet", "lambdarank",
    "xgboost rank", "lightgbm rank", "ranking model", "reranking",
    "ndcg", "map", "mrr", "precision@", "recall@", "rank fusion",
    "reciprocal rank", "cross-encoder", "colbert", "bge reranker",
]

# LLM/RAG signals
LLM_KEYWORDS = [
    "llm", "large language model", "gpt", "claude", "llama", "mistral",
    "rag", "retrieval augmented", "fine-tuning", "finetuning", "lora", "rlhf",
    "instruction tuning", "prompt engineering", "chain of thought",
    "hugging face", "huggingface", "transformers library",
]

# Production/MLOps signals
PRODUCTION_KEYWORDS = [
    "mlops", "model serving", "model deployment", "inference", "latency",
    "throughput", "triton", "torchserve", "bentoml", "ray serve",
    "feature store", "feast", "mlflow", "kubeflow", "airflow",
    "ab testing", "online evaluation", "canary", "shadow deployment",
    "monitoring", "data drift", "model drift",
]

# ─────────────────────────────────────────────
# DOMAIN-MISMATCH SIGNALS
# JD explicitly does NOT want candidates whose primary expertise is computer
# vision / speech / robotics WITHOUT significant NLP/IR exposure. We measure the
# balance of evidence and apply a graded penalty (not a hard zero).
# ─────────────────────────────────────────────
# NOTE: keyword matching is substring-based, so short/common tokens that collide with
# unrelated words (e.g. "gan" in "began", "visual" in "visualization", "ros" in "across")
# are deliberately excluded — only specific, low-collision CV/speech/robotics terms.
CV_KEYWORDS = [
    "computer vision", "image classification", "object detection", "image segmentation",
    "semantic segmentation", "opencv", "resnet", "yolo", "image moderation",
    "face recognition", "facial recognition", "pose estimation",
    "generative adversarial", "diffusion model", "diffusion models", "optical character",
    "image generation", "imagenet", "u-net", "medical imaging", "image recognition",
]
SPEECH_KEYWORDS = [
    "speech recognition", "speech-to-text", "text-to-speech", "speech synthesis",
    "wav2vec", "acoustic model", "speaker recognition", "audio classification", "phoneme",
]
ROBOTICS_KEYWORDS = [
    "robotics", "robot operating system", "slam", "motion planning", "lidar",
    "autonomous navigation", "kinematics",
]
# Positive in-domain evidence (NLP / information retrieval / ranking / LLM / recsys)
NLP_IR_KEYWORDS = [
    "nlp", "natural language", "text classification", "named entity",
    "sentiment", "question answering", "summarization", "machine translation",
    "language model", "llm", "retrieval augmented", "transformer", "bert",
    "information retrieval", "semantic search", "ranking", "reranking",
    "learning to rank", "recommendation", "recommender", "embedding", "embeddings",
    "vector search", "faiss", "milvus", "weaviate", "pinecone", "qdrant",
    "elasticsearch", "opensearch", "bm25", "dense retrieval", "fine-tuning",
    "fine tuning", "sentence transformers", "document classification",
]

# ─────────────────────────────────────────────
# COMPANY TIER CLASSIFICATION
# ─────────────────────────────────────────────

# Tier 1 — FAANG/Top Product Companies (score: 1.0)
TIER_1_COMPANIES = {
    "google", "meta", "facebook", "apple", "amazon", "microsoft", "netflix",
    "openai", "anthropic", "deepmind", "google deepmind",
    "linkedin", "twitter", "x corp", "uber", "airbnb", "stripe", "square",
    "salesforce", "adobe", "nvidia", "amd", "intel",
}

# Tier 2 — Strong Indian Product Companies + Global Mid-tier (score: 0.85)
TIER_2_COMPANIES = {
    "razorpay", "zepto", "swiggy", "zomato", "flipkart", "meesho", "cred",
    "phonepe", "paytm", "nykaa", "urban company", "urbanclap", "ola", "rapido",
    "slice", "groww", "zerodha", "sharechat", "dailyhunt", "moj", "josh",
    "freshworks", "zoho", "chargebee", "leadsquared", "clevertap",
    "unacademy", "byju", "byjus", "vedantu", "upgrad",
    "atlassian", "shopify", "twilio", "datadog", "cloudflare",
    "cohere", "mistral", "hugging face", "huggingface",
    "redrob", "springworks",
}

# Tier 3 — Decent but not top-tier product/startup (score: 0.65)
TIER_3_COMPANIES = {
    "startup", "scale",  # generic but positive signals
    "yahoo", "ebay", "paypal", "oracle", "sap", "ibm",
}

# Service/Consulting firms — PENALTY (score: 0.3)
SERVICE_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "hcl", "hcl technologies",
    "tech mahindra", "cognizant", "capgemini", "accenture", "atos",
    "mphasis", "hexaware", "mindtree", "l&t infotech", "ltimindtree",
    "niit technologies", "zensar", "mssl", "birlasoft", "cyient",
    "persistent systems",  # borderline — keep low
    "deloitte", "pwc", "kpmg", "ey", "ernst young", "mckinsey", "bain", "bcg",
    # BPO / managed-services / staffing shops (often dress up as "AI" arms)
    "genpact", "wns", "exlservice", "exl service", "firstsource", "teleperformance",
    "conduent", "sutherland", "wipro digital", "infosys bpm", "concentrix",
}

# ─────────────────────────────────────────────
# DISQUALIFYING TITLE PATTERNS
# (current_title that strongly suggests non-ML background)
# ─────────────────────────────────────────────
DISQUALIFYING_TITLES = [
    "hr manager", "human resources", "recruiter", "talent acquisition",
    "marketing manager", "digital marketing", "seo", "content writer",
    "accountant", "finance", "chartered accountant", "auditor",
    "sales executive", "business development", "account manager", "sales manager",
    "operations manager", "supply chain", "logistics", "warehouse",
    "mechanical engineer", "civil engineer", "electrical engineer",
    "chemical engineer", "structural engineer", "aerospace engineer",
    "graphic designer", "ui designer", "ux designer", "interior designer",
    "customer support", "customer success", "customer service",
    "product manager",  # softer penalty — some PMs are technical
    "project manager", "business analyst", "program manager",
    "scrum master", "agile coach",
    "content writer", "copywriter", "editor", "journalist", "blogger",
    "teacher", "professor", "lecturer", "instructor", "tutor",
    "nurse", "doctor", "physician", "pharmacist", "pharmacy",
    "hr manager", "human resources", "recruiter", "talent acquisition",
    "receptionist", "admin", "administration", "office manager",
    "marketing manager", "marketing specialist", "brand manager", "seo",
    "legal", "lawyer", "attorney", "paralegal", "compliance",
    "chef", "cook", "hospitality", "hotel",
    "real estate", "insurance", "banking officer", "loan officer",
]

# JD-Aligned titles (positive signal)
ALIGNED_TITLES = [
    "machine learning engineer", "ml engineer", "ai engineer",
    "research engineer", "applied scientist", "applied ml",
    "nlp engineer", "data scientist", "senior data scientist",
    "search engineer", "ranking engineer", "recommendation engineer",
    "retrieval engineer", "mlops engineer", "platform engineer",
    "staff engineer", "principal engineer", "tech lead",
    "software engineer", "sde",  # neutral-positive
]

# ─────────────────────────────────────────────
# LOCATION SCORING
# ─────────────────────────────────────────────
PREFERRED_LOCATIONS = {"pune", "noida", "delhi", "ncr", "gurugram", "gurgaon", "greater noida"}
INDIA_LOCATIONS = {"india", "bangalore", "bengaluru", "mumbai", "hyderabad", "chennai",
                   "kolkata", "ahmedabad", "jaipur", "kochi"}

# ─────────────────────────────────────────────
# BEHAVIORAL SIGNAL THRESHOLDS
# ─────────────────────────────────────────────
INACTIVITY_HALF_LIFE_DAYS = 90      # exponential decay for last_active_date
MAX_NOTICE_PERIOD_DAYS = 90         # candidates beyond this get heavy penalty
IDEAL_NOTICE_PERIOD_DAYS = 30

# ─────────────────────────────────────────────
# FINAL SCORING WEIGHTS
# ─────────────────────────────────────────────
WEIGHTS = {
    "career_trajectory":  0.38,   # product_company_years, pedigree, domain shipping
    "technical_fit":      0.27,   # core skill alignment, retrieval/ranking/LLM depth
    "behavioral":         0.08,   # soft behavioral score
    "availability":       0.12,   # notice period, open_to_work, relocation, location
    "bm25_score":         0.10,   # Stage 2 retrieval score (normalized)
    "education":          0.05,   # institution tier bonus
}

# Multiplier caps (applied after weighted sum)
HONEYPOT_MULTIPLIER = 0.0          # guaranteed lowest tier
KEYWORD_STUFFER_MULTIPLIER = 0.3   # strong penalty
CONSULTING_ONLY_MULTIPLIER = 0.4   # career entirely in service firms (floor of the graded penalty)
DISQUALIFYING_TITLE_MULTIPLIER = 0.0  # wrong domain entirely → hard filter
SUSPICIOUS_HONEYPOT_THRESHOLD = 0.5   # candidates with hp_mult below this are dropped

# Domain-mismatch (CV / speech / robotics primary, thin NLP/IR) — graded penalty.
# domain_mismatch_score in [0,1]; final multiplier interpolates between these bounds.
CV_MISMATCH_FLOOR = 0.40           # strongest penalty when fully CV/speech-dominant w/ no NLP/IR
CV_MISMATCH_CEIL = 1.0             # no penalty when in-domain
CV_MISMATCH_TRIGGER = 0.45         # only penalize when domain_mismatch_score exceeds this
# Explicit CV/speech/robotics job titles are an unambiguous domain declaration; force a
# strong mismatch for them UNLESS they also show significant NLP/IR work (the JD carve-out).
CV_TITLE_TERMS = ("computer vision", "speech", "robotics")
CV_TITLE_FORCED_MISMATCH = 0.80
CV_TITLE_NLP_EXEMPTION_HITS = 5    # >= this many NLP/IR work-hits exempts a CV-titled candidate

# Title-chaser (job-hops for title every ~year) — mild penalty.
TITLE_CHASER_MIN_ROLES = 4
TITLE_CHASER_MAX_AVG_TENURE_MONTHS = 18
TITLE_CHASER_MULTIPLIER = 0.85

# ─────────────────────────────────────────────
# EXPERIENCE RANGE
# ─────────────────────────────────────────────
IDEAL_YOE_MIN = 4.0
IDEAL_YOE_MAX = 12.0
HARD_MIN_YOE = 2.0   # below this → near-disqualifier
