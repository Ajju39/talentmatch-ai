# TalentMatch AI 🎯
### AI-Powered Talent Intelligence Platform

![TalentMatch AI](https://img.shields.io/badge/Powered%20by-Claude%20(Anthropic)-gold?style=flat-square)
![Stack](https://img.shields.io/badge/Stack-FastAPI%20%7C%20FAISS%20%7C%20LangChain%20%7C%20RAG-teal?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)

---

## 🧠 What It Does

TalentMatch AI is a **two-sided talent intelligence platform** that serves both job seekers and hiring companies:

### For Candidates 👤
- **AI Job Matching** — Upload your profile, get matched to best-fit roles using RAG + FAISS vector search
- **Skill Gap Analysis** — Personalized learning roadmap to bridge the gap between where you are and where you want to go
- **Career Growth Advisor** — Conversational AI coach powered by Claude (Anthropic) that knows your exact profile
- **Salary Insights** — Real-time market benchmarks for your specific skills and experience level

### For Companies 🏢
- **Talent Intelligence** — Find candidates that match your exact requirements
- **Market Benchmarking** — Competitive salary ranges and talent pool insights
- **AI Hiring Advisor** — Strategic recommendations for sourcing and screening

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     TalentMatch AI                          │
├─────────────────────────────────────────────────────────────┤
│  Frontend (index.html)                                      │
│  ├── Candidate View: Profile → Job Match → Chat Coach       │
│  └── Company View: Requirements → Talent Search → Insights  │
├─────────────────────────────────────────────────────────────┤
│  Backend (FastAPI - main.py)                                │
│  ├── /api/match-jobs     → RAG job matching                 │
│  ├── /api/chat           → Conversational AI (Claude)       │
│  └── /api/company/find-talent → Hiring intelligence        │
├─────────────────────────────────────────────────────────────┤
│  RAG Engine                                                  │
│  ├── SentenceTransformers  (all-MiniLM-L6-v2 embeddings)   │
│  ├── FAISS Index           (cosine similarity search)       │
│  └── Knowledge Base        (jobs.json + careers.json)       │
├─────────────────────────────────────────────────────────────┤
│  LLM Layer                                                   │
│  ├── Claude (Anthropic) — claude-sonnet-4                   │
│  └── LangChain-compatible prompt architecture               │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/Ajju39/talentmatch-ai
cd talentmatch-ai
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set your API key
```bash
export ANTHROPIC_API_KEY=your_key_here
```

### 4. Run the server
```bash
python main.py
```

### 5. Open the app
```
http://localhost:8000
```

---

## 🧩 Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | Claude (Anthropic) — claude-sonnet-4 |
| **RAG** | FAISS + SentenceTransformers |
| **Backend** | FastAPI + Python |
| **Embeddings** | all-MiniLM-L6-v2 (384-dim vectors) |
| **Frontend** | Vanilla HTML/CSS/JS (no build needed) |
| **Orchestration** | LangChain-compatible architecture |

---

## 📁 Project Structure

```
talentmatch-ai/
├── main.py              # FastAPI backend + RAG engine
├── index.html           # Full-stack frontend UI
├── requirements.txt     # Python dependencies
├── data/
│   ├── jobs.json        # Job knowledge base (8 roles)
│   └── careers.json     # Career path knowledge base (7 levels)
└── README.md
```

---

## 🎥 Demo

> **Live Demo Video:** [Loom recording link here]

**Demo flow:**
1. Enter candidate profile (pre-filled with sample data)
2. Click "Analyze My Profile" → RAG retrieves top job matches
3. Chat with AI career coach → Claude gives personalized advice
4. Switch to Company view → Hiring intelligence for recruiters
5. Market Intelligence tab → Real-time talent market data

---

## 🔮 What This Demonstrates

This project was built to showcase the exact technical capabilities needed for **Astoria AI's** talent intelligence platform:

| Astoria AI Need | This Project Shows |
|---|---|
| Conversational AI | Claude-powered career coach with context |
| Vector search & embeddings | FAISS + SentenceTransformers RAG pipeline |
| Personalization | Profile-aware recommendations per user |
| LLM-based tools & APIs | FastAPI + Claude Anthropic SDK |
| Full stack delivery | Frontend + Backend + AI layer in one project |
| Two-sided platform | Both candidate AND company views |

---

## 👨‍💻 Built By

**Mallikarjun (Arjun) Gannavaram**
- 7+ years AI/Data Engineering
- Production RAG, LangChain, Claude API experience
- linkedin.com/in/mallikarjungannavaram

*Built in response to Astoria AI's engineering opportunity — demonstrating how I'd contribute from day one.*

---

## 📄 License

MIT License — feel free to explore, extend, and build on this.
