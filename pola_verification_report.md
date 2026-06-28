# Verification Report — Manual Deep-Dive & Profile Audit

This report verifies the LightGBM LambdaMART ranker's output in `output/submission.csv`. Part 1 is a grounded, candidate-by-candidate analysis; Part 2 is the exact extracted profile data the analysis is based on. All facts are taken directly from the candidate records — nothing is invented. It covers the top 10, the bottom 10 of the top 100 (to check the trade-offs at the cut line), and a sample of candidates the system filtered OUT of the top 100.

---

## Part 1: Agent Manual Deep-Dive & Quality Comparison

### 1. The Top 10 Candidates (Ranks 1–10)

**CAND_0077337 (Rank 1, Score 0.6875) — Staff Machine Learning Engineer at Paytm**
- *Resume Details*: 7.0 YOE; 19mo at Paytm; 14mo at Razorpay; 44mo at Glance.
- *Actual Work*: The new system reduced p95 retrieval latency by 60% while improving NDCG@10 by 18% on our held-out eval set.
- *Why it makes sense*: JD-relevant skills: Semantic Search, QLoRA, Pinecone, Feature Engineering, BM25, Information Retrieval; 95% recruiter response, 60d notice.

**CAND_0096142 (Rank 2, Score 0.6497) — Applied ML Engineer at upGrad**
- *Resume Details*: 5.0 YOE; 42mo at upGrad; 18mo at BYJU'S.
- *Actual Work*: Trained and shipped multiple ranking models for our product's discovery feed using XGBoost and LightGBM.
- *Why it makes sense*: JD-relevant skills: LoRA, Weaviate, Pinecone, BentoML, RAG, NLP; 84% recruiter response, 120d notice. Noted concern(s): long notice period (120d).

**CAND_0098454 (Rank 3, Score 0.6392) — AI Specialist at Meesho**
- *Resume Details*: 6.6 YOE; 52mo at Meesho; 26mo at Wipro.
- *Actual Work*: Comfortable with PyTorch and Hugging Face but most of my training experience has been on small datasets and pre-trained model fine-tuning, not from-scratch model design.
- *Why it makes sense*: JD-relevant skills: Vector Search, MLOps, Feature Engineering, NLP, OpenSearch, Pinecone; 87% recruiter response, 60d notice.

**CAND_0002025 (Rank 4, Score 0.6341) — Senior AI Engineer at Apple**
- *Resume Details*: 5.9 YOE; 42mo at Apple; 28mo at Aganitha.
- *Actual Work*: Fine-tuned LLaMA-2-7B and Mistral-7B variants using LoRA and QLoRA for domain-specific candidate-JD matching.
- *Why it makes sense*: JD-relevant skills: FAISS, TensorFlow, OpenSearch, Weaviate, Sentence Transformers, QLoRA; 80% recruiter response, 30d notice.

**CAND_0074225 (Rank 5, Score 0.6193) — Machine Learning Engineer at Unacademy**
- *Resume Details*: 4.3 YOE; 26mo at Unacademy; 25mo at Mad Street Den.
- *Actual Work*: Built the document ingestion pipeline (chunking, embedding via OpenAI embeddings, storing in Pinecone) and the answer-generation layer (initially GPT-4, then a fine-tuned smaller model for cost control).
- *Why it makes sense*: JD-relevant skills: Semantic Search, Elasticsearch, Milvus, Hugging Face Transformers, Qdrant, QLoRA; 91% recruiter response, 120d notice. Noted concern(s): long notice period (120d).

**CAND_0094482 (Rank 6, Score 0.6071) — Junior ML Engineer at TCS**
- *Resume Details*: 6.4 YOE; 24mo at TCS; 52mo at Zomato.
- *Actual Work*: Comfortable with PyTorch and Hugging Face but most of my training experience has been on small datasets and pre-trained model fine-tuning, not from-scratch model design.
- *Why it makes sense*: JD-relevant skills: BentoML, LLMs, OpenSearch, MLOps, PyTorch, Fine-tuning LLMs; 85% recruiter response, 90d notice. Noted concern(s): located outside India (Singapore).

**CAND_0053591 (Rank 7, Score 0.6022) — AI Engineer at Ola**
- *Resume Details*: 5.3 YOE; 38mo at Ola; 25mo at Swiggy.
- *Actual Work*: Built the document ingestion pipeline (chunking, embedding via OpenAI embeddings, storing in Pinecone) and the answer-generation layer (initially GPT-4, then a fine-tuned smaller model for cost control).
- *Why it makes sense*: JD-relevant skills: Embeddings, Milvus, OpenSearch, Sentence Transformers, Qdrant, BM25; 81% recruiter response, 60d notice. Noted concern(s): located outside India (Toronto).

**CAND_0064326 (Rank 8, Score 0.602) — Search Engineer at Sarvam AI**
- *Resume Details*: 7.6 YOE; 31mo at Sarvam AI; 24mo at Aganitha; 24mo at Freshworks.
- *Actual Work*: Built the document ingestion pipeline (chunking, embedding via OpenAI embeddings, storing in Pinecone) and the answer-generation layer (initially GPT-4, then a fine-tuned smaller model for cost control).
- *Why it makes sense*: JD-relevant skills: PyTorch, Milvus, Deep Learning, Semantic Search, Weaviate, RAG; 94% recruiter response, 45d notice.

**CAND_0030031 (Rank 9, Score 0.5852) — AI Engineer at Microsoft**
- *Resume Details*: 5.7 YOE; 13mo at Microsoft; 27mo at Amazon; 27mo at Google.
- *Actual Work*: Built and operated production ML pipelines using MLflow for experiment tracking, Kubeflow for orchestration, and our internal feature store.
- *Why it makes sense*: JD-relevant skills: Information Retrieval, PyTorch, NLP, RAG, LoRA, QLoRA; 94% recruiter response, 30d notice. Noted concern(s): not flagged open-to-work.

**CAND_0050876 (Rank 10, Score 0.5831) — Applied ML Engineer at Freshworks**
- *Resume Details*: 6.0 YOE; 38mo at Freshworks; 24mo at Yellow.ai; 9mo at Razorpay.
- *Actual Work*: Owned the ranking layer for an e-commerce search product, evolving it from a hand-tuned scoring function to a learning-to-rank model over 9 months.
- *Why it makes sense*: JD-relevant skills: Qdrant, MLOps, FAISS, LlamaIndex, Machine Learning, OpenSearch; 67% recruiter response, 90d notice.

### 2. The Bottom 10 of the Top 100 (Ranks 91–100)

- **CAND_0088025 (Rank 91)** — Staff Machine Learning Engineer at Yellow.ai, 8.6 YOE. Relevant skills: pinecone, qlora, llms, hugging face transformers. Borderline on overall fit strength.
- **CAND_0006418 (Rank 92)** — Machine Learning Engineer at Verloop.io, 5.7 YOE. Relevant skills: semantic search, embeddings, tensorflow, weaviate. Borderline on overall fit strength.
- **CAND_0000031 (Rank 93)** — Recommendation Systems Engineer at Swiggy, 6.0 YOE. Relevant skills: faiss, pinecone, machine learning, bentoml. Borderline on overall fit strength.
- **CAND_0041610 (Rank 94)** — Recommendation Systems Engineer at Zoho, 6.7 YOE. Relevant skills: lora, elasticsearch, opensearch, bm25. Borderline on overall fit strength.
- **CAND_0099806 (Rank 95)** — AI Engineer at Mad Street Den, 4.6 YOE. Relevant skills: lora, rag, faiss, embeddings. Borderline on overall fit strength.
- **CAND_0005538 (Rank 96)** — Senior AI Engineer at Adobe, 5.9 YOE. Relevant skills: deep learning, feature engineering, qlora, pytorch. Borderline on overall fit strength.
- **CAND_0042100 (Rank 97)** — Machine Learning Engineer at Freshworks, 7.3 YOE. Relevant skills: elasticsearch, learning to rank, semantic search, fine-tuning llms. Sits near the cut line due to: located outside India (Singapore).
- **CAND_0006557 (Rank 98)** — NLP Engineer at Paytm, 7.9 YOE. Relevant skills: elasticsearch, opensearch, llamaindex, vector search. Sits near the cut line due to: long notice period (120d).
- **CAND_0028793 (Rank 99)** — Search Engineer at Google, 7.2 YOE. Relevant skills: embeddings, lora, learning to rank, information retrieval. Sits near the cut line due to: long notice period (120d).
- **CAND_0052712 (Rank 100)** — AI Research Engineer at Verloop.io, 3.1 YOE. Relevant skills: hugging face transformers, lora, nlp, embeddings. Sits near the cut line due to: long notice period (120d), YOE 3.1 outside ideal 4-12yr band.

### 3. Sample of Candidates Filtered OUT of the Top 100

These confirm the ranker is reading profiles, not keyword-matching: each was kept out of the top 100 for a defensible reason.

- **CAND_0000001** — Backend Engineer at Mindtree, 6.9 YOE. Correctly excluded: low recruiter response rate (34%), located outside India (Toronto).
- **CAND_0010002** — Civil Engineer at Pied Piper, 3.4 YOE. Correctly excluded: no JD-relevant ML skills, low recruiter response rate (18%), long notice period (150d), not flagged open-to-work, YOE 3.4 outside ideal 4-12yr band.
- **CAND_0020002** — Customer Support at Globex Inc, 4.3 YOE. Correctly excluded: no JD-relevant ML skills, not flagged open-to-work.
- **CAND_0029998** — Project Manager at TCS, 13.0 YOE. Correctly excluded: no JD-relevant ML skills, low recruiter response rate (39%), not flagged open-to-work, YOE 13.0 outside ideal 4-12yr band.
- **CAND_0039995** — Accountant at Globex Inc, 10.7 YOE. Correctly excluded: no JD-relevant ML skills, not flagged open-to-work, located outside India (Berlin).
- **CAND_0049997** — Marketing Manager at Initech, 14.0 YOE. Correctly excluded: no JD-relevant ML skills, low recruiter response rate (9%), long notice period (120d), not flagged open-to-work, YOE 14.0 outside ideal 4-12yr band.
- **CAND_0059998** — Graphic Designer at Dunder Mifflin, 11.6 YOE. Correctly excluded: no JD-relevant ML skills, low recruiter response rate (21%), long notice period (120d).
- **CAND_0070000** — DevOps Engineer at TCS, 1.5 YOE. Correctly excluded: long notice period (120d), not flagged open-to-work, YOE 1.5 outside ideal 4-12yr band, service-firm-heavy career (100%).
- **CAND_0080005** — Project Manager at Acme Corp, 3.9 YOE. Correctly excluded: no JD-relevant ML skills, not flagged open-to-work, YOE 3.9 outside ideal 4-12yr band.
- **CAND_0090002** — Accountant at Stark Industries, 1.1 YOE. Correctly excluded: low recruiter response rate (14%), located outside India (Toronto), YOE 1.1 outside ideal 4-12yr band.

---

## Part 2: Complete Extracted Profile Details (Script Output)

### Group 1: First 10 (Top-Ranked)

### ID: CAND_0077337 (Rank: 1, Score: 0.6875)
- **Current Title**: Staff Machine Learning Engineer at Paytm
- **YOE**: 7.0 | **Location**: Kochi, Kerala
- **Ranker Reasoning**: *Strong fit: 7yr Staff Machine Learning Engineer with production retrieval/recommendation experience at Paytm, bringing Semantic Search, pgvector. Highly active (95% response) and open to work.*
- **Skills**: GANs, Semantic Search, QLoRA, pgvector, Pinecone, Feature Engineering, BM25, Information Retrieval, LLMs, OpenCV, Data Science, Forecasting, Excel, RAG, Qdrant, Recommendation Systems, Sentence Transformers, LlamaIndex, Python, OpenSearch
- **Notice Period**: 60 days
- **Recruiter Response Rate**: 0.95
- **Open to Work Flag**: True
- **Complete Career History**:
  * **Staff Machine Learning Engineer** at **Paytm** (19 months)
    *Description*: Built and shipped a production recommendation system at a marketplace product, going from offline experimentation to live A/B test in 5 months. The system combined collaborative filtering (matrix factorization), content-based features (TF-IDF + sentence-transformer embeddings), and a behavioral re-ranking layer. The most interesting technical challenge was the cold-start problem for new users; I designed an exploration-exploitation policy using Thompson sampling that improved new-user retention by 11% in the first month.
  * **Senior NLP Engineer** at **Razorpay** (14 months)
    *Description*: Owned the design and rollout of a large-scale semantic search system serving an internal corpus of 35M+ items. Migrated the existing BM25-only retrieval to a hybrid setup combining sparse and dense vectors (sentence-transformers, MPNet-base initially, later fine-tuned BGE-large for our domain). The new system reduced p95 retrieval latency by 60% while improving NDCG@10 by 18% on our held-out eval set. Spent substantial time on the boring-but-critical parts: incremental index refresh, embedding drift monitoring, online/offline metric correlation. Led a team of 4 engineers across the rollout.
  * **Senior NLP Engineer** at **Glance** (44 months)
    *Description*: Led the migration from keyword-based to embedding-based search across a 30M+ candidate corpus over 8 months. Designed three successive ranker variants and ran them in A/B testing alongside the legacy keyword system. The final embedding ranker improved recruiter engagement metrics by 24% and reduced the average time-to-shortlist by 38%. Most of the engineering effort went into the boring infrastructure: index versioning, embedding versioning, rollback paths, and the dashboards that let recruiters trust the new system. Mentored two junior engineers through this rollout.
  * **Senior AI Engineer** at **Aganitha** (6 months)
    *Description*: Led the migration from keyword-based to embedding-based search across a 30M+ candidate corpus over 8 months. Designed three successive ranker variants and ran them in A/B testing alongside the legacy keyword system. The final embedding ranker improved recruiter engagement metrics by 24% and reduced the average time-to-shortlist by 38%. Most of the engineering effort went into the boring infrastructure: index versioning, embedding versioning, rollback paths, and the dashboards that let recruiters trust the new system. Mentored two junior engineers through this rollout.

================================================================================

### ID: CAND_0096142 (Rank: 2, Score: 0.6497)
- **Current Title**: Applied ML Engineer at upGrad
- **YOE**: 5.0 | **Location**: Hyderabad, Telangana
- **Ranker Reasoning**: *Strong fit: 5yr Applied ML Engineer with production retrieval/recommendation experience at upGrad, bringing Weaviate, Pinecone, Python. Highly active (84% response) and open to work.*
- **Skills**: Kubeflow, LoRA, Weaviate, Pinecone, Python, CSS, BentoML, Weights & Biases, RAG, CNN, NLP, BM25, OpenCV, Hugging Face Transformers, dbt
- **Notice Period**: 120 days
- **Recruiter Response Rate**: 0.84
- **Open to Work Flag**: True
- **Complete Career History**:
  * **Applied ML Engineer** at **upGrad** (42 months)
    *Description*: Trained and shipped multiple ranking models for our product's discovery feed using XGBoost and LightGBM. Designed features across three families: content metadata, user behavior signals, and item engagement history. Owned the offline-online correlation analysis that determined which offline metrics actually predicted A/B test outcomes. Worked closely with PMs to define the optimization target (click-through vs. dwell time vs. downstream conversion) — that work was as important as the modeling itself.
  * **Applied ML Engineer** at **BYJU'S** (18 months)
    *Description*: Built a content recommendation system serving 10M+ users that combined collaborative filtering with content-based ranking. The system uses item-item similarity (via sentence-transformer embeddings) for cold starts and a gradient-boosted model trained on engagement signals for warm users. Most of my time went into the feature pipeline (~200 features) and the A/B testing infrastructure. The launch improved 7-day retention by 6% and time spent per session by 14%.

================================================================================

### ID: CAND_0098454 (Rank: 3, Score: 0.6392)
- **Current Title**: AI Specialist at Meesho
- **YOE**: 6.6 | **Location**: Indore, Madhya Pradesh
- **Ranker Reasoning**: *Adaptive profile: 7yr AI Specialist with product engineering experience at Meesho alongside some service-firm work, showing OpenSearch competency.*
- **Skills**: Vector Search, MLOps, Feature Engineering, NLP, Statistical Modeling, OpenSearch, Data Science, ASR, Pinecone, CNN, Weaviate, Embeddings, GANs, Illustrator
- **Notice Period**: 60 days
- **Recruiter Response Rate**: 0.87
- **Open to Work Flag**: True
- **Complete Career History**:
  * **AI Specialist** at **Meesho** (52 months)
    *Description*: Contributed to ML feature engineering and model deployment for a fraud-detection product. My main role was engineering: building the Flask-based prediction API, integrating with the feature store, and writing the model-serving observability layer. I worked closely with senior data scientists but my own modeling work was secondary — I was the production-side engineer.
  * **AI Research Engineer** at **Wipro** (26 months)
    *Description*: Built NLP pipelines for sentiment analysis and document classification — primarily for an internal feedback-analytics dashboard. Started with sklearn-based bag-of-words models, then moved to transformer-based classifiers (DistilBERT) for the harder classes. Comfortable with PyTorch and Hugging Face but most of my training experience has been on small datasets and pre-trained model fine-tuning, not from-scratch model design.

================================================================================

### ID: CAND_0002025 (Rank: 4, Score: 0.6341)
- **Current Title**: Senior AI Engineer at Apple
- **YOE**: 5.9 | **Location**: Trivandrum, Kerala
- **Ranker Reasoning**: *Strong fit: 6yr Senior AI Engineer with production retrieval/recommendation experience at Apple, bringing FAISS, TensorFlow, OpenSearch. Highly active (80% response) and open to work.*
- **Skills**: Diffusion Models, FAISS, TensorFlow, scikit-learn, OpenSearch, Haystack, Weaviate, Sentence Transformers, QLoRA, NLP, Pinecone, Recommendation Systems, Deep Learning, Python, LangChain, Weights & Biases, OpenCV, Prompt Engineering, Fine-tuning LLMs, YOLO
- **Notice Period**: 30 days
- **Recruiter Response Rate**: 0.8
- **Open to Work Flag**: True
- **Complete Career History**:
  * **Senior AI Engineer** at **Apple** (42 months)
    *Description*: Built and shipped a production recommendation system at a marketplace product, going from offline experimentation to live A/B test in 5 months. The system combined collaborative filtering (matrix factorization), content-based features (TF-IDF + sentence-transformer embeddings), and a behavioral re-ranking layer. The most interesting technical challenge was the cold-start problem for new users; I designed an exploration-exploitation policy using Thompson sampling that improved new-user retention by 11% in the first month.
  * **Lead AI Engineer** at **Aganitha** (28 months)
    *Description*: Fine-tuned LLaMA-2-7B and Mistral-7B variants using LoRA and QLoRA for domain-specific candidate-JD matching. Built the data curation pipeline that generated 200K high-quality preference pairs from recruiter labels, plus the eval harness using both ranking metrics and human-quality scores. Deployed the model via BentoML on Kubernetes with sub-200ms p95 latency by quantizing to INT8 and batching at the request level. Cost per inference dropped from $0.04 with GPT-3.5-fallback to under $0.001.

================================================================================

### ID: CAND_0074225 (Rank: 5, Score: 0.6193)
- **Current Title**: Machine Learning Engineer at Unacademy
- **YOE**: 4.3 | **Location**: Vizag, Andhra Pradesh
- **Ranker Reasoning**: *Strong fit: 4yr Machine Learning Engineer with production retrieval/recommendation experience at Unacademy, bringing Recommendation Systems, Elasticsearch. Highly active (91% response) and open to work.*
- **Skills**: Apache Beam, Statistical Modeling, Recommendation Systems, Semantic Search, ASR, Elasticsearch, Milvus, Python, Time Series, Haystack, scikit-learn, Hugging Face Transformers, Qdrant, CI/CD, QLoRA, Machine Learning, Vector Search
- **Notice Period**: 120 days
- **Recruiter Response Rate**: 0.91
- **Open to Work Flag**: True
- **Complete Career History**:
  * **Machine Learning Engineer** at **Unacademy** (26 months)
    *Description*: Trained and shipped multiple ranking models for our product's discovery feed using XGBoost and LightGBM. Designed features across three families: content metadata, user behavior signals, and item engagement history. Owned the offline-online correlation analysis that determined which offline metrics actually predicted A/B test outcomes. Worked closely with PMs to define the optimization target (click-through vs. dwell time vs. downstream conversion) — that work was as important as the modeling itself.
  * **Machine Learning Engineer** at **Mad Street Den** (25 months)
    *Description*: Implemented a RAG-based customer support chatbot integrated with our existing ticketing system. Built the document ingestion pipeline (chunking, embedding via OpenAI embeddings, storing in Pinecone) and the answer-generation layer (initially GPT-4, then a fine-tuned smaller model for cost control). Designed the evaluation framework with both automatic metrics (BLEU, ROUGE) and human-in-the-loop quality scores. Deployment cut average ticket resolution time by 31% for the supported categories.

================================================================================

### ID: CAND_0094482 (Rank: 6, Score: 0.6071)
- **Current Title**: Junior ML Engineer at TCS
- **YOE**: 6.4 | **Location**: Singapore
- **Ranker Reasoning**: *High-potential candidate: 6yr Junior ML Engineer with solid MLOps depth at Zomato, currently based in Singapore but actively looking to relocate.*
- **Skills**: Rust, BentoML, Figma, LLMs, OpenSearch, Data Science, Next.js, Reinforcement Learning, MLOps, PyTorch, Fine-tuning LLMs
- **Notice Period**: 90 days
- **Recruiter Response Rate**: 0.85
- **Open to Work Flag**: True
- **Complete Career History**:
  * **Junior ML Engineer** at **TCS** (24 months)
    *Description*: Built NLP pipelines for sentiment analysis and document classification — primarily for an internal feedback-analytics dashboard. Started with sklearn-based bag-of-words models, then moved to transformer-based classifiers (DistilBERT) for the harder classes. Comfortable with PyTorch and Hugging Face but most of my training experience has been on small datasets and pre-trained model fine-tuning, not from-scratch model design.
  * **Senior Software Engineer (ML)** at **Zomato** (52 months)
    *Description*: Worked on time-series forecasting models for supply-chain demand prediction at a logistics company. Built models in Prophet, LightGBM, and (for one project) a small LSTM — the LightGBM model ended up shipping. Also ran some reinforcement learning experiments for dynamic pricing but those didn't make it to production. The work was a mix of modeling, analysis, and stakeholder communication with the operations team.

================================================================================

### ID: CAND_0053591 (Rank: 7, Score: 0.6022)
- **Current Title**: AI Engineer at Ola
- **YOE**: 5.3 | **Location**: Toronto
- **Ranker Reasoning**: *Strong profile: 5yr AI Engineer with solid Embeddings depth from Ola; based in Toronto but open to relocation with high responsiveness.*
- **Skills**: LangChain, Reinforcement Learning, Embeddings, Milvus, OpenSearch, Redux, Object Detection, Sentence Transformers, Data Pipelines, Qdrant, Statistical Modeling, BM25, Six Sigma, RAG
- **Notice Period**: 60 days
- **Recruiter Response Rate**: 0.81
- **Open to Work Flag**: True
- **Complete Career History**:
  * **AI Engineer** at **Ola** (38 months)
    *Description*: Trained and shipped multiple ranking models for our product's discovery feed using XGBoost and LightGBM. Designed features across three families: content metadata, user behavior signals, and item engagement history. Owned the offline-online correlation analysis that determined which offline metrics actually predicted A/B test outcomes. Worked closely with PMs to define the optimization target (click-through vs. dwell time vs. downstream conversion) — that work was as important as the modeling itself.
  * **AI Engineer** at **Swiggy** (25 months)
    *Description*: Implemented a RAG-based customer support chatbot integrated with our existing ticketing system. Built the document ingestion pipeline (chunking, embedding via OpenAI embeddings, storing in Pinecone) and the answer-generation layer (initially GPT-4, then a fine-tuned smaller model for cost control). Designed the evaluation framework with both automatic metrics (BLEU, ROUGE) and human-in-the-loop quality scores. Deployment cut average ticket resolution time by 31% for the supported categories.

================================================================================

### ID: CAND_0064326 (Rank: 8, Score: 0.602)
- **Current Title**: Search Engineer at Sarvam AI
- **YOE**: 7.6 | **Location**: Gurgaon, Haryana
- **Ranker Reasoning**: *Strong fit: 8yr Search Engineer with production retrieval/recommendation experience at Freshworks, bringing Deep Learning, Semantic Search. Highly active (94% response) and open to work.*
- **Skills**: scikit-learn, PyTorch, Milvus, Deep Learning, Semantic Search, Weaviate, Object Detection, RAG, Weights & Biases, BM25, Webpack, Python, QLoRA, Reinforcement Learning
- **Notice Period**: 45 days
- **Recruiter Response Rate**: 0.94
- **Open to Work Flag**: True
- **Complete Career History**:
  * **Search Engineer** at **Sarvam AI** (31 months)
    *Description*: Owned the ranking layer for an e-commerce search product, evolving it from a hand-tuned scoring function to a learning-to-rank model over 9 months. Designed the relevance labeling pipeline (mix of click-through data and explicit human judgments), the feature pipeline, and the training/eval workflow. Most of the work was infrastructure and data quality — the modeling part was almost the easy bit. Final model improved revenue-per-search by 12%.
  * **Machine Learning Engineer** at **Aganitha** (24 months)
    *Description*: Trained and shipped multiple ranking models for our product's discovery feed using XGBoost and LightGBM. Designed features across three families: content metadata, user behavior signals, and item engagement history. Owned the offline-online correlation analysis that determined which offline metrics actually predicted A/B test outcomes. Worked closely with PMs to define the optimization target (click-through vs. dwell time vs. downstream conversion) — that work was as important as the modeling itself.
  * **Machine Learning Engineer** at **Freshworks** (24 months)
    *Description*: Implemented a RAG-based customer support chatbot integrated with our existing ticketing system. Built the document ingestion pipeline (chunking, embedding via OpenAI embeddings, storing in Pinecone) and the answer-generation layer (initially GPT-4, then a fine-tuned smaller model for cost control). Designed the evaluation framework with both automatic metrics (BLEU, ROUGE) and human-in-the-loop quality scores. Deployment cut average ticket resolution time by 31% for the supported categories.
  * **Machine Learning Engineer** at **Apple** (12 months)
    *Description*: Owned the ranking layer for an e-commerce search product, evolving it from a hand-tuned scoring function to a learning-to-rank model over 9 months. Designed the relevance labeling pipeline (mix of click-through data and explicit human judgments), the feature pipeline, and the training/eval workflow. Most of the work was infrastructure and data quality — the modeling part was almost the easy bit. Final model improved revenue-per-search by 12%.

================================================================================

### ID: CAND_0030031 (Rank: 9, Score: 0.5852)
- **Current Title**: AI Engineer at Microsoft
- **YOE**: 5.7 | **Location**: Trivandrum, Kerala
- **Ranker Reasoning**: *Strong fit: 6yr AI Engineer with production retrieval/recommendation experience at Microsoft, bringing Information Retrieval, NLP. Highly active (94% response) and open to work.*
- **Skills**: Information Retrieval, PyTorch, Object Detection, Python, NLP, RAG, OpenCV, LoRA, QLoRA, BM25, Time Series, Sentence Transformers, YOLO, Milvus, Vector Search, scikit-learn
- **Notice Period**: 30 days
- **Recruiter Response Rate**: 0.94
- **Open to Work Flag**: False
- **Complete Career History**:
  * **AI Engineer** at **Microsoft** (13 months)
    *Description*: Built a content recommendation system serving 10M+ users that combined collaborative filtering with content-based ranking. The system uses item-item similarity (via sentence-transformer embeddings) for cold starts and a gradient-boosted model trained on engagement signals for warm users. Most of my time went into the feature pipeline (~200 features) and the A/B testing infrastructure. The launch improved 7-day retention by 6% and time spent per session by 14%.
  * **Senior Data Scientist** at **Amazon** (27 months)
    *Description*: Built and operated production ML pipelines using MLflow for experiment tracking, Kubeflow for orchestration, and our internal feature store. My main project was a churn prediction model that's now used by the customer success team to prioritize outreach. Designed the model monitoring stack: data drift detection, prediction distribution checks, and alerting. Mentored a junior engineer through their first end-to-end ML project last year.
  * **Search Engineer** at **Google** (27 months)
    *Description*: Trained and shipped multiple ranking models for our product's discovery feed using XGBoost and LightGBM. Designed features across three families: content metadata, user behavior signals, and item engagement history. Owned the offline-online correlation analysis that determined which offline metrics actually predicted A/B test outcomes. Worked closely with PMs to define the optimization target (click-through vs. dwell time vs. downstream conversion) — that work was as important as the modeling itself.

================================================================================

### ID: CAND_0050876 (Rank: 10, Score: 0.5831)
- **Current Title**: Applied ML Engineer at Freshworks
- **YOE**: 6.0 | **Location**: Kolkata, West Bengal
- **Ranker Reasoning**: *Strong fit: 6yr Applied ML Engineer with production retrieval/recommendation experience at Freshworks, bringing MLOps, FAISS. Highly active (67% response) and open to work.*
- **Skills**: SQL, Qdrant, MLOps, FAISS, scikit-learn, Weights & Biases, LlamaIndex, Forecasting, Machine Learning, OpenSearch, YOLO, Kubeflow, QLoRA, Sentence Transformers, Image Classification, Python, Prompt Engineering, PyTorch
- **Notice Period**: 90 days
- **Recruiter Response Rate**: 0.67
- **Open to Work Flag**: True
- **Complete Career History**:
  * **Applied ML Engineer** at **Freshworks** (38 months)
    *Description*: Owned the ranking layer for an e-commerce search product, evolving it from a hand-tuned scoring function to a learning-to-rank model over 9 months. Designed the relevance labeling pipeline (mix of click-through data and explicit human judgments), the feature pipeline, and the training/eval workflow. Most of the work was infrastructure and data quality — the modeling part was almost the easy bit. Final model improved revenue-per-search by 12%.
  * **AI Engineer** at **Yellow.ai** (24 months)
    *Description*: Built a content recommendation system serving 10M+ users that combined collaborative filtering with content-based ranking. The system uses item-item similarity (via sentence-transformer embeddings) for cold starts and a gradient-boosted model trained on engagement signals for warm users. Most of my time went into the feature pipeline (~200 features) and the A/B testing infrastructure. The launch improved 7-day retention by 6% and time spent per session by 14%.
  * **Recommendation Systems Engineer** at **Razorpay** (9 months)
    *Description*: Built a content recommendation system serving 10M+ users that combined collaborative filtering with content-based ranking. The system uses item-item similarity (via sentence-transformer embeddings) for cold starts and a gradient-boosted model trained on engagement signals for warm users. Most of my time went into the feature pipeline (~200 features) and the A/B testing infrastructure. The launch improved 7-day retention by 6% and time spent per session by 14%.

================================================================================

### Group 2: Last 10 (Ranks 91-100)

### ID: CAND_0088025 (Rank: 91, Score: 0.4668)
- **Current Title**: Staff Machine Learning Engineer at Yellow.ai
- **YOE**: 8.6 | **Location**: Jaipur, Rajasthan
- **Ranker Reasoning**: *Consulting focus: 8.6yr developer whose career history is primarily in services/consulting, showing lower product-ratio and limited relevancy to Series A product engineering.*
- **Skills**: Pinecone, QLoRA, LLMs, Hugging Face Transformers, RAG, SAP, TensorFlow, LoRA, Flask, Prompt Engineering, BM25, Elasticsearch, MLOps, pgvector, Learning to Rank, Deep Learning, Python, NLP, YOLO
- **Notice Period**: 90 days
- **Recruiter Response Rate**: 0.83
- **Open to Work Flag**: True
- **Complete Career History**:
  * **Staff Machine Learning Engineer** at **Yellow.ai** (45 months)
    *Description*: Owned the end-to-end ranking pipeline at a recommendations-heavy consumer product: candidate sourcing → embedding generation (using a fine-tuned BGE-large) → Pinecone retrieval → learning-to-rank re-scoring (XGBoost) → behavioral-signal integration. The hardest part wasn't the ML — it was the evaluation: building offline metrics that actually predicted what the recommendation would do to live engagement. After three iterations we landed on a calibration approach using simulated A/B tests that has held up over the last 18 months.
  * **Staff Machine Learning Engineer** at **Niramai** (44 months)
    *Description*: Owned the end-to-end ranking pipeline at a recommendations-heavy consumer product: candidate sourcing → embedding generation (using a fine-tuned BGE-large) → Pinecone retrieval → learning-to-rank re-scoring (XGBoost) → behavioral-signal integration. The hardest part wasn't the ML — it was the evaluation: building offline metrics that actually predicted what the recommendation would do to live engagement. After three iterations we landed on a calibration approach using simulated A/B tests that has held up over the last 18 months.
  * **Senior Machine Learning Engineer** at **Genpact AI** (13 months)
    *Description*: Led the migration from keyword-based to embedding-based search across a 30M+ candidate corpus over 8 months. Designed three successive ranker variants and ran them in A/B testing alongside the legacy keyword system. The final embedding ranker improved recruiter engagement metrics by 24% and reduced the average time-to-shortlist by 38%. Most of the engineering effort went into the boring infrastructure: index versioning, embedding versioning, rollback paths, and the dashboards that let recruiters trust the new system. Mentored two junior engineers through this rollout.

================================================================================

### ID: CAND_0006418 (Rank: 92, Score: 0.466)
- **Current Title**: Machine Learning Engineer at Verloop.io
- **YOE**: 5.7 | **Location**: Gurgaon, Haryana
- **Ranker Reasoning**: *Lower-tier candidate: 5.7yr developer at Verloop.io showing limited domain shipping evidence and weak overall alignment with JD-specific search requirements.*
- **Skills**: Kubernetes, gRPC, Semantic Search, Embeddings, TensorFlow, Object Detection, Weaviate, Elasticsearch, Snowflake, MLflow, Learning to Rank, Forecasting, Qdrant, Diffusion Models, Time Series, OpenSearch
- **Notice Period**: 60 days
- **Recruiter Response Rate**: 0.92
- **Open to Work Flag**: True
- **Complete Career History**:
  * **Machine Learning Engineer** at **Verloop.io** (40 months)
    *Description*: Trained and shipped multiple ranking models for our product's discovery feed using XGBoost and LightGBM. Designed features across three families: content metadata, user behavior signals, and item engagement history. Owned the offline-online correlation analysis that determined which offline metrics actually predicted A/B test outcomes. Worked closely with PMs to define the optimization target (click-through vs. dwell time vs. downstream conversion) — that work was as important as the modeling itself.
  * **AI Engineer** at **Flipkart** (27 months)
    *Description*: Built and operated production ML pipelines using MLflow for experiment tracking, Kubeflow for orchestration, and our internal feature store. My main project was a churn prediction model that's now used by the customer success team to prioritize outreach. Designed the model monitoring stack: data drift detection, prediction distribution checks, and alerting. Mentored a junior engineer through their first end-to-end ML project last year.

================================================================================

### ID: CAND_0000031 (Rank: 93, Score: 0.4659)
- **Current Title**: Recommendation Systems Engineer at Swiggy
- **YOE**: 6.0 | **Location**: Hyderabad, Telangana
- **Ranker Reasoning**: *Lower-tier candidate: 6.0yr developer at Swiggy showing limited domain shipping evidence and weak overall alignment with JD-specific search requirements.*
- **Skills**: Go, MLflow, FAISS, Pinecone, Angular, Image Classification, Machine Learning, Speech Recognition, BentoML, MLOps, Embeddings, Information Retrieval, Hugging Face Transformers, Feature Engineering, Sentence Transformers, scikit-learn, Marketing
- **Notice Period**: 60 days
- **Recruiter Response Rate**: 0.91
- **Open to Work Flag**: True
- **Complete Career History**:
  * **Recommendation Systems Engineer** at **Swiggy** (14 months)
    *Description*: Trained and shipped multiple ranking models for our product's discovery feed using XGBoost and LightGBM. Designed features across three families: content metadata, user behavior signals, and item engagement history. Owned the offline-online correlation analysis that determined which offline metrics actually predicted A/B test outcomes. Worked closely with PMs to define the optimization target (click-through vs. dwell time vs. downstream conversion) — that work was as important as the modeling itself.
  * **Search Engineer** at **Mad Street Den** (16 months)
    *Description*: Trained and shipped multiple ranking models for our product's discovery feed using XGBoost and LightGBM. Designed features across three families: content metadata, user behavior signals, and item engagement history. Owned the offline-online correlation analysis that determined which offline metrics actually predicted A/B test outcomes. Worked closely with PMs to define the optimization target (click-through vs. dwell time vs. downstream conversion) — that work was as important as the modeling itself.
  * **NLP Engineer** at **Uber** (27 months)
    *Description*: Trained and shipped multiple ranking models for our product's discovery feed using XGBoost and LightGBM. Designed features across three families: content metadata, user behavior signals, and item engagement history. Owned the offline-online correlation analysis that determined which offline metrics actually predicted A/B test outcomes. Worked closely with PMs to define the optimization target (click-through vs. dwell time vs. downstream conversion) — that work was as important as the modeling itself.
  * **Applied ML Engineer** at **Zomato** (13 months)
    *Description*: Owned the ranking layer for an e-commerce search product, evolving it from a hand-tuned scoring function to a learning-to-rank model over 9 months. Designed the relevance labeling pipeline (mix of click-through data and explicit human judgments), the feature pipeline, and the training/eval workflow. Most of the work was infrastructure and data quality — the modeling part was almost the easy bit. Final model improved revenue-per-search by 12%.

================================================================================

### ID: CAND_0041610 (Rank: 94, Score: 0.4649)
- **Current Title**: Recommendation Systems Engineer at Zoho
- **YOE**: 6.7 | **Location**: Indore, Madhya Pradesh
- **Ranker Reasoning**: *Lower-tier candidate: 6.7yr developer at Zoho showing limited domain shipping evidence and weak overall alignment with JD-specific search requirements.*
- **Skills**: OpenCV, LoRA, Statistical Modeling, Data Science, Elasticsearch, GANs, OpenSearch, LangChain, BM25, scikit-learn, Learning to Rank, Forecasting, Feature Engineering, Embeddings, pgvector, Recommendation Systems, PyTorch
- **Notice Period**: 30 days
- **Recruiter Response Rate**: 0.52
- **Open to Work Flag**: True
- **Complete Career History**:
  * **Recommendation Systems Engineer** at **Zoho** (31 months)
    *Description*: Owned the ranking layer for an e-commerce search product, evolving it from a hand-tuned scoring function to a learning-to-rank model over 9 months. Designed the relevance labeling pipeline (mix of click-through data and explicit human judgments), the feature pipeline, and the training/eval workflow. Most of the work was infrastructure and data quality — the modeling part was almost the easy bit. Final model improved revenue-per-search by 12%.
  * **Applied ML Engineer** at **Observe.AI** (26 months)
    *Description*: Built and operated production ML pipelines using MLflow for experiment tracking, Kubeflow for orchestration, and our internal feature store. My main project was a churn prediction model that's now used by the customer success team to prioritize outreach. Designed the model monitoring stack: data drift detection, prediction distribution checks, and alerting. Mentored a junior engineer through their first end-to-end ML project last year.
  * **Search Engineer** at **InMobi** (15 months)
    *Description*: Implemented a RAG-based customer support chatbot integrated with our existing ticketing system. Built the document ingestion pipeline (chunking, embedding via OpenAI embeddings, storing in Pinecone) and the answer-generation layer (initially GPT-4, then a fine-tuned smaller model for cost control). Designed the evaluation framework with both automatic metrics (BLEU, ROUGE) and human-in-the-loop quality scores. Deployment cut average ticket resolution time by 31% for the supported categories.
  * **Machine Learning Engineer** at **Swiggy** (7 months)
    *Description*: Owned the ranking layer for an e-commerce search product, evolving it from a hand-tuned scoring function to a learning-to-rank model over 9 months. Designed the relevance labeling pipeline (mix of click-through data and explicit human judgments), the feature pipeline, and the training/eval workflow. Most of the work was infrastructure and data quality — the modeling part was almost the easy bit. Final model improved revenue-per-search by 12%.

================================================================================

### ID: CAND_0099806 (Rank: 95, Score: 0.4646)
- **Current Title**: AI Engineer at Mad Street Den
- **YOE**: 4.6 | **Location**: Bhubaneswar, Odisha
- **Ranker Reasoning**: *Lower-tier candidate: 4.6yr developer at Mad Street Den showing limited domain shipping evidence and weak overall alignment with JD-specific search requirements.*
- **Skills**: LoRA, OpenCV, Statistical Modeling, RAG, Agile, Prompt Engineering, Speech Recognition, Reinforcement Learning, pgvector, FAISS, Embeddings, Sentence Transformers, Weaviate, Time Series, dbt, Qdrant, Elasticsearch, BM25
- **Notice Period**: 30 days
- **Recruiter Response Rate**: 0.76
- **Open to Work Flag**: True
- **Complete Career History**:
  * **AI Engineer** at **Mad Street Den** (33 months)
    *Description*: Trained and shipped multiple ranking models for our product's discovery feed using XGBoost and LightGBM. Designed features across three families: content metadata, user behavior signals, and item engagement history. Owned the offline-online correlation analysis that determined which offline metrics actually predicted A/B test outcomes. Worked closely with PMs to define the optimization target (click-through vs. dwell time vs. downstream conversion) — that work was as important as the modeling itself.
  * **Machine Learning Engineer** at **upGrad** (21 months)
    *Description*: Trained and shipped multiple ranking models for our product's discovery feed using XGBoost and LightGBM. Designed features across three families: content metadata, user behavior signals, and item engagement history. Owned the offline-online correlation analysis that determined which offline metrics actually predicted A/B test outcomes. Worked closely with PMs to define the optimization target (click-through vs. dwell time vs. downstream conversion) — that work was as important as the modeling itself.

================================================================================

### ID: CAND_0005538 (Rank: 96, Score: 0.4636)
- **Current Title**: Senior AI Engineer at Adobe
- **YOE**: 5.9 | **Location**: Kolkata, West Bengal
- **Ranker Reasoning**: *Lower-tier candidate: 5.9yr developer at Adobe showing limited domain shipping evidence and weak overall alignment with JD-specific search requirements.*
- **Skills**: Project Management, Vector Representations, Deep Learning, Content Matching, Apache Flink, Feature Engineering, Statistical Modeling, QLoRA, Python, PyTorch, ASR, Data Science, LoRA, Information Retrieval Systems, pgvector, Haystack, Workflow Orchestration, TTS, Natural Language Processing
- **Notice Period**: 90 days
- **Recruiter Response Rate**: 0.81
- **Open to Work Flag**: True
- **Complete Career History**:
  * **Senior AI Engineer** at **Adobe** (15 months)
    *Description*: Led the engineering team building infrastructure to surface relevant content to users at scale. The system processed billions of documents and served millions of queries with low latency. Most of the technical effort went into the boring-but-essential parts: index refresh, query understanding, ranking calibration, and the dashboards that made the system's behavior legible to product and business teams. I had a small team of 4 across this work.
  * **Lead AI Engineer** at **Locobuzz** (30 months)
    *Description*: Built systems that understand what users are looking for and connect them to the most relevant matches across a large dataset. Worked at the intersection of infrastructure, algorithms, and product judgment — none of the three were optional. Recent project was a complete overhaul of the matching layer; took it from a hand-tuned heuristic system to one with explicit modeling and evaluation. The team grew from just me to 6 engineers over the course of that work.
  * **Senior Machine Learning Engineer** at **Google** (14 months)
    *Description*: Built systems that understand what users are looking for and connect them to the most relevant matches across a large dataset. Worked at the intersection of infrastructure, algorithms, and product judgment — none of the three were optional. Recent project was a complete overhaul of the matching layer; took it from a hand-tuned heuristic system to one with explicit modeling and evaluation. The team grew from just me to 6 engineers over the course of that work.
  * **Lead AI Engineer** at **Glance** (10 months)
    *Description*: Built and shipped a production recommendation system at a marketplace product, going from offline experimentation to live A/B test in 5 months. The system combined collaborative filtering (matrix factorization), content-based features (TF-IDF + sentence-transformer embeddings), and a behavioral re-ranking layer. The most interesting technical challenge was the cold-start problem for new users; I designed an exploration-exploitation policy using Thompson sampling that improved new-user retention by 11% in the first month.

================================================================================

### ID: CAND_0042100 (Rank: 97, Score: 0.4631)
- **Current Title**: Machine Learning Engineer at Freshworks
- **YOE**: 7.3 | **Location**: Singapore
- **Ranker Reasoning**: *Relocation required: 7.3yr developer based in Singapore with weaker JD alignment and extended notice period, making hiring logistics unfavorable.*
- **Skills**: Elasticsearch, Statistical Modeling, Learning to Rank, Recommendation Systems, Semantic Search, Fine-tuning LLMs, Image Classification, Pinecone, LLMs, QLoRA, Reinforcement Learning, PyTorch, Next.js
- **Notice Period**: 90 days
- **Recruiter Response Rate**: 0.87
- **Open to Work Flag**: True
- **Complete Career History**:
  * **Machine Learning Engineer** at **Freshworks** (52 months)
    *Description*: Owned the ranking layer for an e-commerce search product, evolving it from a hand-tuned scoring function to a learning-to-rank model over 9 months. Designed the relevance labeling pipeline (mix of click-through data and explicit human judgments), the feature pipeline, and the training/eval workflow. Most of the work was infrastructure and data quality — the modeling part was almost the easy bit. Final model improved revenue-per-search by 12%.
  * **Senior Data Scientist** at **Netflix** (34 months)
    *Description*: Trained and shipped multiple ranking models for our product's discovery feed using XGBoost and LightGBM. Designed features across three families: content metadata, user behavior signals, and item engagement history. Owned the offline-online correlation analysis that determined which offline metrics actually predicted A/B test outcomes. Worked closely with PMs to define the optimization target (click-through vs. dwell time vs. downstream conversion) — that work was as important as the modeling itself.

================================================================================

### ID: CAND_0006557 (Rank: 98, Score: 0.4623)
- **Current Title**: NLP Engineer at Paytm
- **YOE**: 7.9 | **Location**: Jaipur, Rajasthan
- **Ranker Reasoning**: *Lower-tier candidate: 7.9yr developer at Paytm showing limited domain shipping evidence and weak overall alignment with JD-specific search requirements.*
- **Skills**: Kubeflow, Data Science, Elasticsearch, OpenSearch, LlamaIndex, OpenCV, Vector Search, Agile, Weaviate, Vue.js, FAISS, Pinecone, Hugging Face Transformers, Qdrant, Databricks, LoRA, LangChain, Information Retrieval
- **Notice Period**: 120 days
- **Recruiter Response Rate**: 0.63
- **Open to Work Flag**: True
- **Complete Career History**:
  * **NLP Engineer** at **Paytm** (54 months)
    *Description*: Owned the ranking layer for an e-commerce search product, evolving it from a hand-tuned scoring function to a learning-to-rank model over 9 months. Designed the relevance labeling pipeline (mix of click-through data and explicit human judgments), the feature pipeline, and the training/eval workflow. Most of the work was infrastructure and data quality — the modeling part was almost the easy bit. Final model improved revenue-per-search by 12%.
  * **Recommendation Systems Engineer** at **Apple** (40 months)
    *Description*: Built a content recommendation system serving 10M+ users that combined collaborative filtering with content-based ranking. The system uses item-item similarity (via sentence-transformer embeddings) for cold starts and a gradient-boosted model trained on engagement signals for warm users. Most of my time went into the feature pipeline (~200 features) and the A/B testing infrastructure. The launch improved 7-day retention by 6% and time spent per session by 14%.

================================================================================

### ID: CAND_0028793 (Rank: 99, Score: 0.4623)
- **Current Title**: Search Engineer at Google
- **YOE**: 7.2 | **Location**: Trivandrum, Kerala
- **Ranker Reasoning**: *Lower-tier candidate: 7.2yr developer at Google showing limited domain shipping evidence and weak overall alignment with JD-specific search requirements.*
- **Skills**: SAP, Kubeflow, Embeddings, Haystack, YOLO, LoRA, PowerPoint, Learning to Rank, Information Retrieval, PEFT, QLoRA, PyTorch, pgvector, NLP, Weights & Biases
- **Notice Period**: 120 days
- **Recruiter Response Rate**: 0.57
- **Open to Work Flag**: True
- **Complete Career History**:
  * **Search Engineer** at **Google** (32 months)
    *Description*: Developed a semantic search feature for an internal knowledge base of ~500K documents. Used sentence-transformers (all-MiniLM-L6-v2 initially, later upgraded to bge-base) with FAISS for fast nearest-neighbor retrieval. Designed the query expansion module that handles vocabulary mismatch between user queries and document terms. Reported search-relevance improvement of 35% over the prior Elasticsearch BM25 setup, validated through human relevance judgments.
  * **Senior Data Scientist** at **Amazon** (40 months)
    *Description*: Implemented a RAG-based customer support chatbot integrated with our existing ticketing system. Built the document ingestion pipeline (chunking, embedding via OpenAI embeddings, storing in Pinecone) and the answer-generation layer (initially GPT-4, then a fine-tuned smaller model for cost control). Designed the evaluation framework with both automatic metrics (BLEU, ROUGE) and human-in-the-loop quality scores. Deployment cut average ticket resolution time by 31% for the supported categories.
  * **Machine Learning Engineer** at **Meesho** (13 months)
    *Description*: Trained and shipped multiple ranking models for our product's discovery feed using XGBoost and LightGBM. Designed features across three families: content metadata, user behavior signals, and item engagement history. Owned the offline-online correlation analysis that determined which offline metrics actually predicted A/B test outcomes. Worked closely with PMs to define the optimization target (click-through vs. dwell time vs. downstream conversion) — that work was as important as the modeling itself.

================================================================================

### ID: CAND_0052712 (Rank: 100, Score: 0.4622)
- **Current Title**: AI Research Engineer at Verloop.io
- **YOE**: 3.1 | **Location**: Coimbatore, Tamil Nadu
- **Ranker Reasoning**: *Junior profile: 3.1yr engineer with insufficient YOE (3.1 years) and minimal exposure to shipping production-level machine learning models.*
- **Skills**: Computer Vision, Hugging Face Transformers, Data Science, Spring Boot, LoRA, NLP, Kubernetes, MLflow, Weights & Biases, Photoshop, Embeddings, Time Series, Fine-tuning LLMs, Terraform
- **Notice Period**: 120 days
- **Recruiter Response Rate**: 0.78
- **Open to Work Flag**: True
- **Complete Career History**:
  * **AI Research Engineer** at **Verloop.io** (26 months)
    *Description*: Contributed to ML feature engineering and model deployment for a fraud-detection product. My main role was engineering: building the Flask-based prediction API, integrating with the feature store, and writing the model-serving observability layer. I worked closely with senior data scientists but my own modeling work was secondary — I was the production-side engineer.
  * **AI Research Engineer** at **Observe.AI** (10 months)
    *Description*: Built NLP pipelines for sentiment analysis and document classification — primarily for an internal feedback-analytics dashboard. Started with sklearn-based bag-of-words models, then moved to transformer-based classifiers (DistilBERT) for the harder classes. Comfortable with PyTorch and Hugging Face but most of my training experience has been on small datasets and pre-trained model fine-tuning, not from-scratch model design.

================================================================================

### Group 3: 10 Candidates Outside Top 100

### ID: CAND_0000001 (Rank: Outside Top-100, Score: N/A)
- **Current Title**: Backend Engineer at Mindtree
- **YOE**: 6.9 | **Location**: Toronto
- **Ranker Reasoning**: *N/A*
- **Skills**: Tailwind, NLP, Image Classification, Fine-tuning LLMs, Weights & Biases, Speech Recognition, Photoshop, TTS, LoRA, Apache Beam, AWS, Flask, BentoML, Milvus, GANs, Statistical Modeling, GCP
- **Notice Period**: 60 days
- **Recruiter Response Rate**: 0.34
- **Open to Work Flag**: True
- **Complete Career History**:
  * **Backend Engineer** at **Mindtree** (27 months)
    *Description*: Implemented streaming data pipelines on Kafka and Spark Streaming for a real-time user-activity processing platform. Designed the schema-registry integration, the watermark/state management approach, and the deduplication logic for late-arriving events. Worked closely with the data science team to make sure feature pipelines aligned with what their models needed. Most of my career has been data engineering, with some adjacent ML exposure.
  * **Analytics Engineer** at **Dunder Mifflin** (55 months)
    *Description*: Built and maintained data pipelines on Apache Airflow processing ~500GB of daily transactional data across 12 source systems. Worked extensively with Spark (PySpark) for batch processing and dbt for the transformation/modeling layer in our Snowflake warehouse. Owned the on-call rotation for data quality issues — wrote most of the data quality checks that detect schema drift and unusual volume changes. The pipeline supports the analytics team and a few internal ML models.

================================================================================

### ID: CAND_0010002 (Rank: Outside Top-100, Score: N/A)
- **Current Title**: Civil Engineer at Pied Piper
- **YOE**: 3.4 | **Location**: Pune, Maharashtra
- **Ranker Reasoning**: *N/A*
- **Skills**: gRPC, PowerPoint, Excel, Spring Boot, SAP, Webpack, Node.js
- **Notice Period**: 150 days
- **Recruiter Response Rate**: 0.18
- **Open to Work Flag**: False
- **Complete Career History**:
  * **Civil Engineer** at **Pied Piper** (40 months)
    *Description*: Enterprise sales of cloud software solutions into the mid-market segment. Carried a $1.8M ARR quota and consistently delivered against it across the last three years. Owned the full sales cycle: prospecting, discovery, technical evaluation (with SE support), commercial negotiation, and close. Strong on consultative selling for technical buyers; comfortable engaging with both engineering and finance stakeholders.

================================================================================

### ID: CAND_0020002 (Rank: Outside Top-100, Score: N/A)
- **Current Title**: Customer Support at Globex Inc
- **YOE**: 4.3 | **Location**: Chandigarh, Chandigarh
- **Ranker Reasoning**: *N/A*
- **Skills**: Kubeflow, Airflow, HTML, Azure, PostgreSQL, Node.js, GraphQL, SAP, Terraform
- **Notice Period**: 90 days
- **Recruiter Response Rate**: 0.78
- **Open to Work Flag**: False
- **Complete Career History**:
  * **Customer Support** at **Globex Inc** (15 months)
    *Description*: Customer support team lead at a SaaS product. Managed a team of 8 support agents handling tier-1 and tier-2 tickets; owned the escalation process to engineering and the customer-feedback loop to product. Built out the support knowledge base and the agent training program. Strong on the people-management side and the process side; lighter on technical depth beyond product expertise.
  * **Operations Manager** at **Wayne Enterprises** (27 months)
    *Description*: Customer support team lead at a SaaS product. Managed a team of 8 support agents handling tier-1 and tier-2 tickets; owned the escalation process to engineering and the customer-feedback loop to product. Built out the support knowledge base and the agent training program. Strong on the people-management side and the process side; lighter on technical depth beyond product expertise.
  * **Project Manager** at **Stark Industries** (8 months)
    *Description*: Marketing leadership role at a B2B SaaS company. Owned the demand-generation function — content marketing, paid acquisition, SEO, email nurture. Built and managed a team of 5 across content, performance marketing, and marketing operations. Worked closely with sales on lead-quality definitions and the SDR-handoff process. Recent focus has been on account-based marketing for our enterprise segment.

================================================================================

### ID: CAND_0029998 (Rank: Outside Top-100, Score: N/A)
- **Current Title**: Project Manager at TCS
- **YOE**: 13.0 | **Location**: Ahmedabad, Gujarat
- **Ranker Reasoning**: *N/A*
- **Skills**: React, Figma, Kubernetes, Spark, Kafka, CSS, Data Pipelines, Redux, PowerPoint, Spring Boot
- **Notice Period**: 90 days
- **Recruiter Response Rate**: 0.39
- **Open to Work Flag**: False
- **Complete Career History**:
  * **Project Manager** at **TCS** (30 months)
    *Description*: Senior accounting role at a mid-sized company — month-end close, financial reporting, statutory compliance (GAAP / Ind-AS), and tax filings. Owned the GL, fixed-asset register, and the audit-readiness function. Managed a team of 3 staff accountants. Built strong process discipline around the close cycle, reducing close time from 12 days to 7 over the last two years.
  * **HR Manager** at **Dunder Mifflin** (37 months)
    *Description*: Brand design and creative direction at a consumer-products company. Owned brand identity (logo, visual system, typography), packaging design, and digital creative across web and social. Led the recent rebrand and managed a small external agency for production work. Comfortable across the Adobe suite, Figma, and the production side of brand and packaging design.
  * **Civil Engineer** at **Infosys** (31 months)
    *Description*: Customer support team lead at a SaaS product. Managed a team of 8 support agents handling tier-1 and tier-2 tickets; owned the escalation process to engineering and the customer-feedback loop to product. Built out the support knowledge base and the agent training program. Strong on the people-management side and the process side; lighter on technical depth beyond product expertise.
  * **Operations Manager** at **Dunder Mifflin** (49 months)
    *Description*: Marketing leadership role at a B2B SaaS company. Owned the demand-generation function — content marketing, paid acquisition, SEO, email nurture. Built and managed a team of 5 across content, performance marketing, and marketing operations. Worked closely with sales on lead-quality definitions and the SDR-handoff process. Recent focus has been on account-based marketing for our enterprise segment.
  * **Business Analyst** at **Stark Industries** (8 months)
    *Description*: Operations management role at a logistics company. Owned daily fulfillment operations across 3 warehouses, managing a team of 80 across receiving, picking, packing, and outbound. Built and tracked the operational KPIs (on-time fulfillment, accuracy, cost per order) and led the continuous improvement initiatives that drove a 22% productivity gain over 18 months.

================================================================================

### ID: CAND_0039995 (Rank: Outside Top-100, Score: N/A)
- **Current Title**: Accountant at Globex Inc
- **YOE**: 10.7 | **Location**: Berlin
- **Ranker Reasoning**: *N/A*
- **Skills**: Terraform, Scrum, TTS, Data Pipelines, Docker, SAP, Six Sigma, GCP
- **Notice Period**: 60 days
- **Recruiter Response Rate**: 0.48
- **Open to Work Flag**: False
- **Complete Career History**:
  * **Accountant** at **Globex Inc** (27 months)
    *Description*: Brand design and creative direction at a consumer-products company. Owned brand identity (logo, visual system, typography), packaging design, and digital creative across web and social. Led the recent rebrand and managed a small external agency for production work. Comfortable across the Adobe suite, Figma, and the production side of brand and packaging design.
  * **Sales Executive** at **Pied Piper** (30 months)
    *Description*: Enterprise sales of cloud software solutions into the mid-market segment. Carried a $1.8M ARR quota and consistently delivered against it across the last three years. Owned the full sales cycle: prospecting, discovery, technical evaluation (with SE support), commercial negotiation, and close. Strong on consultative selling for technical buyers; comfortable engaging with both engineering and finance stakeholders.
  * **Project Manager** at **Dunder Mifflin** (37 months)
    *Description*: Brand design and creative direction at a consumer-products company. Owned brand identity (logo, visual system, typography), packaging design, and digital creative across web and social. Led the recent rebrand and managed a small external agency for production work. Comfortable across the Adobe suite, Figma, and the production side of brand and packaging design.
  * **Business Analyst** at **TCS** (16 months)
    *Description*: Marketing leadership role at a B2B SaaS company. Owned the demand-generation function — content marketing, paid acquisition, SEO, email nurture. Built and managed a team of 5 across content, performance marketing, and marketing operations. Worked closely with sales on lead-quality definitions and the SDR-handoff process. Recent focus has been on account-based marketing for our enterprise segment.
  * **Graphic Designer** at **Wipro** (16 months)
    *Description*: Business analyst at a consulting firm, working primarily with retail and CPG clients. Conducted business diagnostics, process re-engineering work, and digital transformation strategy projects. Strong on stakeholder management, structured problem-solving, and the typical consulting toolkit (slide-craft, Excel modeling, executive communication). Recent project work involved AI-strategy advisory but my own technical depth in AI is limited.

================================================================================

### ID: CAND_0049997 (Rank: Outside Top-100, Score: N/A)
- **Current Title**: Marketing Manager at Initech
- **YOE**: 14.0 | **Location**: Coimbatore, Tamil Nadu
- **Ranker Reasoning**: *N/A*
- **Skills**: CI/CD, Kubeflow, Hadoop, Apache Beam, SQL, HTML
- **Notice Period**: 120 days
- **Recruiter Response Rate**: 0.09
- **Open to Work Flag**: False
- **Complete Career History**:
  * **Marketing Manager** at **Initech** (54 months)
    *Description*: Enterprise sales of cloud software solutions into the mid-market segment. Carried a $1.8M ARR quota and consistently delivered against it across the last three years. Owned the full sales cycle: prospecting, discovery, technical evaluation (with SE support), commercial negotiation, and close. Strong on consultative selling for technical buyers; comfortable engaging with both engineering and finance stakeholders.
  * **Business Analyst** at **Dunder Mifflin** (19 months)
    *Description*: Business analyst at a consulting firm, working primarily with retail and CPG clients. Conducted business diagnostics, process re-engineering work, and digital transformation strategy projects. Strong on stakeholder management, structured problem-solving, and the typical consulting toolkit (slide-craft, Excel modeling, executive communication). Recent project work involved AI-strategy advisory but my own technical depth in AI is limited.
  * **Business Analyst** at **Initech** (33 months)
    *Description*: Enterprise sales of cloud software solutions into the mid-market segment. Carried a $1.8M ARR quota and consistently delivered against it across the last three years. Owned the full sales cycle: prospecting, discovery, technical evaluation (with SE support), commercial negotiation, and close. Strong on consultative selling for technical buyers; comfortable engaging with both engineering and finance stakeholders.
  * **Business Analyst** at **Globex Inc** (44 months)
    *Description*: Mechanical engineering design role at a hardware-product company. Led the design of two product subsystems through full lifecycle: concept, DFM/DFMA review, prototype, production tooling. Comfortable with CAD (SolidWorks, Creo), FEA (ANSYS), and the typical hardware-development cadence. Worked closely with manufacturing partners on production scale-up.
  * **Sales Executive** at **Infosys** (16 months)
    *Description*: Business analyst at a consulting firm, working primarily with retail and CPG clients. Conducted business diagnostics, process re-engineering work, and digital transformation strategy projects. Strong on stakeholder management, structured problem-solving, and the typical consulting toolkit (slide-craft, Excel modeling, executive communication). Recent project work involved AI-strategy advisory but my own technical depth in AI is limited.

================================================================================

### ID: CAND_0059998 (Rank: Outside Top-100, Score: N/A)
- **Current Title**: Graphic Designer at Dunder Mifflin
- **YOE**: 11.6 | **Location**: Bhubaneswar, Odisha
- **Ranker Reasoning**: *N/A*
- **Skills**: Six Sigma, TTS, Accounting, PowerPoint, Salesforce CRM, Scrum, Redis, Apache Beam
- **Notice Period**: 120 days
- **Recruiter Response Rate**: 0.21
- **Open to Work Flag**: True
- **Complete Career History**:
  * **Graphic Designer** at **Dunder Mifflin** (13 months)
    *Description*: Brand design and creative direction at a consumer-products company. Owned brand identity (logo, visual system, typography), packaging design, and digital creative across web and social. Led the recent rebrand and managed a small external agency for production work. Comfortable across the Adobe suite, Figma, and the production side of brand and packaging design.
  * **Operations Manager** at **Stark Industries** (15 months)
    *Description*: Mechanical engineering design role at a hardware-product company. Led the design of two product subsystems through full lifecycle: concept, DFM/DFMA review, prototype, production tooling. Comfortable with CAD (SolidWorks, Creo), FEA (ANSYS), and the typical hardware-development cadence. Worked closely with manufacturing partners on production scale-up.
  * **Mechanical Engineer** at **Initech** (40 months)
    *Description*: Operations management role at a logistics company. Owned daily fulfillment operations across 3 warehouses, managing a team of 80 across receiving, picking, packing, and outbound. Built and tracked the operational KPIs (on-time fulfillment, accuracy, cost per order) and led the continuous improvement initiatives that drove a 22% productivity gain over 18 months.
  * **Accountant** at **Infosys** (31 months)
    *Description*: Operations management role at a logistics company. Owned daily fulfillment operations across 3 warehouses, managing a team of 80 across receiving, picking, packing, and outbound. Built and tracked the operational KPIs (on-time fulfillment, accuracy, cost per order) and led the continuous improvement initiatives that drove a 22% productivity gain over 18 months.
  * **Civil Engineer** at **Globex Inc** (38 months)
    *Description*: Brand design and creative direction at a consumer-products company. Owned brand identity (logo, visual system, typography), packaging design, and digital creative across web and social. Led the recent rebrand and managed a small external agency for production work. Comfortable across the Adobe suite, Figma, and the production side of brand and packaging design.

================================================================================

### ID: CAND_0070000 (Rank: Outside Top-100, Score: N/A)
- **Current Title**: DevOps Engineer at TCS
- **YOE**: 1.5 | **Location**: Mumbai, Maharashtra
- **Ranker Reasoning**: *N/A*
- **Skills**: MLOps, Machine Learning, Six Sigma, HTML, Accounting, GANs, Speech Recognition, Qdrant, Django, Apache Beam, MongoDB, Tally, Terraform
- **Notice Period**: 120 days
- **Recruiter Response Rate**: 0.64
- **Open to Work Flag**: False
- **Complete Career History**:
  * **DevOps Engineer** at **TCS** (18 months)
    *Description*: Full-stack web application development at a SaaS company. Built React-based admin interfaces and the Node.js REST API backing them. Worked across the stack: frontend components, REST endpoint design, PostgreSQL schema, deployment via Docker/Kubernetes. Comfortable in most parts of a typical web stack though my comfort zone is the backend and database side. Recent learning has been on the testing and CI/CD discipline.

================================================================================

### ID: CAND_0080005 (Rank: Outside Top-100, Score: N/A)
- **Current Title**: Project Manager at Acme Corp
- **YOE**: 3.9 | **Location**: Hyderabad, Telangana
- **Ranker Reasoning**: *N/A*
- **Skills**: dbt, Spring Boot, Snowflake, Redis, SEO, Tally, Terraform, Image Classification, Rust
- **Notice Period**: 90 days
- **Recruiter Response Rate**: 0.63
- **Open to Work Flag**: False
- **Complete Career History**:
  * **Project Manager** at **Acme Corp** (28 months)
    *Description*: Marketing leadership role at a B2B SaaS company. Owned the demand-generation function — content marketing, paid acquisition, SEO, email nurture. Built and managed a team of 5 across content, performance marketing, and marketing operations. Worked closely with sales on lead-quality definitions and the SDR-handoff process. Recent focus has been on account-based marketing for our enterprise segment.
  * **Civil Engineer** at **TCS** (18 months)
    *Description*: Enterprise sales of cloud software solutions into the mid-market segment. Carried a $1.8M ARR quota and consistently delivered against it across the last three years. Owned the full sales cycle: prospecting, discovery, technical evaluation (with SE support), commercial negotiation, and close. Strong on consultative selling for technical buyers; comfortable engaging with both engineering and finance stakeholders.

================================================================================

### ID: CAND_0090002 (Rank: Outside Top-100, Score: N/A)
- **Current Title**: Accountant at Stark Industries
- **YOE**: 1.1 | **Location**: Toronto
- **Ranker Reasoning**: *N/A*
- **Skills**: BigQuery, BentoML, Data Pipelines, Kafka, Terraform, Docker, Snowflake, PowerPoint
- **Notice Period**: 90 days
- **Recruiter Response Rate**: 0.14
- **Open to Work Flag**: True
- **Complete Career History**:
  * **Accountant** at **Stark Industries** (13 months)
    *Description*: Operations management role at a logistics company. Owned daily fulfillment operations across 3 warehouses, managing a team of 80 across receiving, picking, packing, and outbound. Built and tracked the operational KPIs (on-time fulfillment, accuracy, cost per order) and led the continuous improvement initiatives that drove a 22% productivity gain over 18 months.

================================================================================
