"""
TalentMatch AI - Backend
FastAPI + Claude/Anthropic + FAISS RAG Engine
"""

import os
import json
import numpy as np

# ── Load .env file automatically ─────────────────────────────────
from pathlib import Path
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import anthropic
from sentence_transformers import SentenceTransformer
import faiss

# ── App Setup ─────────────────────────────────────────────────────
app = FastAPI(title="TalentMatch AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Models ────────────────────────────────────────────────────────
class CandidateProfile(BaseModel):
    name: str
    current_role: str
    years_experience: int
    skills: List[str]
    goal: str
    domain_interest: Optional[str] = ""

class ChatMessage(BaseModel):
    message: str
    profile: Optional[CandidateProfile] = None
    history: Optional[List[dict]] = []
    mode: Optional[str] = "candidate"

class CompanyQuery(BaseModel):
    company_name: str
    role_needed: str
    required_skills: List[str]
    experience_needed: int

# ── RAG Engine ────────────────────────────────────────────────────
class TalentRAGEngine:
    def __init__(self):
        print("Loading embedding model...")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.jobs = []
        self.careers = []
        self.job_index = None
        self.career_index = None
        self._build_indexes()

    def _build_indexes(self):
        with open("data/jobs.json") as f:
            self.jobs = json.load(f)
        with open("data/careers.json") as f:
            self.careers = json.load(f)

        job_texts = [
            f"{j['title']} {j['description']} {' '.join(j['skills'])} {j['domain']}"
            for j in self.jobs
        ]
        job_embeddings = self.encoder.encode(job_texts, normalize_embeddings=True)
        self.job_index = faiss.IndexFlatIP(job_embeddings.shape[1])
        self.job_index.add(job_embeddings.astype('float32'))

        career_texts = [
            f"{c['role']} {c['advice']} {' '.join(c['skills_needed'])} {' '.join(c['skills_to_grow'])}"
            for c in self.careers
        ]
        career_embeddings = self.encoder.encode(career_texts, normalize_embeddings=True)
        self.career_index = faiss.IndexFlatIP(career_embeddings.shape[1])
        self.career_index.add(career_embeddings.astype('float32'))

        print(f"✅ RAG indexes built: {len(self.jobs)} jobs, {len(self.careers)} career paths")

    def search_jobs(self, query: str, top_k: int = 3) -> List[dict]:
        query_vec = self.encoder.encode([query], normalize_embeddings=True).astype('float32')
        scores, indices = self.job_index.search(query_vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            job = self.jobs[idx].copy()
            job['match_score'] = round(float(score) * 100, 1)
            results.append(job)
        return results

    def search_careers(self, query: str, top_k: int = 2) -> List[dict]:
        query_vec = self.encoder.encode([query], normalize_embeddings=True).astype('float32')
        scores, indices = self.career_index.search(query_vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            career = self.careers[idx].copy()
            career['relevance_score'] = round(float(score) * 100, 1)
            results.append(career)
        return results

# Initialize RAG engine
rag = TalentRAGEngine()

# ── Claude AI Client ──────────────────────────────────────────────
claude_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

# ── System Prompts ────────────────────────────────────────────────
def build_system_prompt(mode: str, profile: Optional[CandidateProfile] = None) -> str:

    if mode == "candidate" and profile:
        return f"""You are TalentMatch AI — a smart, helpful assistant and career coach.

Current user profile:
- Name: {profile.name}
- Role: {profile.current_role}
- Experience: {profile.years_experience} years
- Skills: {', '.join(profile.skills)}
- Goal: {profile.goal}

Core rules:
1. Answer ANY question the user asks — career-related or not
2. If they ask about a company (Google, Amazon, Microsoft, Meta, etc.) — give real, accurate, detailed information about that company: culture, interview process, salaries, tech stack, hiring tips
3. If they ask about technologies, tools, programming languages — explain them clearly and accurately
4. If they ask general knowledge questions — answer them directly and helpfully
5. ONLY reference the user profile or job/career context when it is directly relevant to what they asked
6. Never force career advice into a question that isn't about careers
7. Be conversational, warm, and genuinely helpful — like a smart friend who happens to know a lot about tech and careers

When the question IS career-related, use the profile context and any retrieved job/career data to give personalized, specific advice."""

    elif mode == "company":
        return """You are TalentMatch AI — a talent intelligence advisor for hiring managers and recruiters.

Core rules:
1. Answer ANY question asked — not just hiring questions
2. If they ask about a specific company, technology, or market — give accurate, real-world information
3. When asked about hiring, provide: market insights, salary benchmarks, candidate pool analysis, screening criteria, and sourcing strategies
4. Be data-driven, professional, and specific
5. If asked about competitors or industry trends — answer factually and thoroughly
6. Never give vague or generic advice — always be concrete and actionable"""

    else:
        return """You are TalentMatch AI — a helpful, knowledgeable assistant covering careers, technology, companies, and the job market.

Answer all questions accurately and helpfully. Be conversational and specific. Never refuse a reasonable question."""


def chat_with_claude(
    message: str,
    system_prompt: str,
    history: List[dict],
    context: str = ""
) -> str:
    """Send message to Claude with optional RAG context."""

    # Only inject RAG context if it's a career/job related question
    career_keywords = ["job", "career", "skill", "salary", "role", "hire", "work", "engineer",
                       "developer", "match", "resume", "interview", "learn", "grow", "path",
                       "experience", "position", "company hiring", "opportunity", "promotion"]

    is_career_question = any(kw in message.lower() for kw in career_keywords)

    full_message = message
    if context and is_career_question:
        full_message = f"""[Relevant career data from TalentMatch knowledge base:]
{context}

[User question:]
{message}

Use the context above only if it is directly relevant. Answer the question fully and helpfully."""

    messages = []
    for h in history[-6:]:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": full_message})

    response = claude_client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        system=system_prompt,
        messages=messages
    )
    return response.content[0].text


# ── API Routes ────────────────────────────────────────────────────

@app.get("/")
def serve_frontend():
    return FileResponse("index.html")

@app.post("/api/match-jobs")
def match_jobs(profile: CandidateProfile):
    """Find best-fit jobs for a candidate profile."""
    query = f"{profile.current_role} {' '.join(profile.skills)} {profile.goal} {profile.domain_interest}"
    matches = rag.search_jobs(query, top_k=4)
    career_query = f"{profile.current_role} {profile.goal} {profile.years_experience} years"
    career_paths = rag.search_careers(career_query, top_k=2)
    return {
        "candidate": profile.name,
        "job_matches": matches,
        "career_paths": career_paths,
        "total_matches": len(matches)
    }

@app.post("/api/chat")
def chat(msg: ChatMessage):
    """Conversational AI chat endpoint with smart RAG injection."""
    try:
        system = build_system_prompt(msg.mode, msg.profile)

        if msg.profile:
            profile_query = f"{msg.profile.current_role} {' '.join(msg.profile.skills)} {msg.profile.goal} {msg.message}"
        else:
            profile_query = msg.message

        relevant_jobs = rag.search_jobs(profile_query, top_k=3)
        relevant_careers = rag.search_careers(profile_query, top_k=2)

        context_parts = []
        if relevant_jobs:
            jobs_text = "\n".join([
                f"- {j['title']} at {j['company']} | Skills: {', '.join(j['skills'][:5])} | Salary: {j['salary_range']} | Match: {j['match_score']}%"
                for j in relevant_jobs
            ])
            context_parts.append(f"JOB MATCHES:\n{jobs_text}")

        if relevant_careers:
            careers_text = "\n".join([
                f"- {c['role']} | Next step: {c['next_role']} | Salary: {c['avg_salary']} | Skills to grow: {', '.join(c['skills_to_grow'][:3])}"
                for c in relevant_careers
            ])
            context_parts.append(f"CAREER PATHS:\n{careers_text}")

        context = "\n\n".join(context_parts)

        response = chat_with_claude(
            message=msg.message,
            system_prompt=system,
            history=msg.history or [],
            context=context
        )

        return {
            "response": response,
            "retrieved_jobs": relevant_jobs[:2],
            "mode": msg.mode
        }

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"\n❌ CHAT ERROR:\n{error_details}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/company/find-talent")
def find_talent(query: CompanyQuery):
    """Help companies find best-fit candidates and get market insights."""
    search_query = f"{query.role_needed} {' '.join(query.required_skills)} {query.experience_needed} years"
    matching_roles = rag.search_jobs(search_query, top_k=3)

    system = build_system_prompt("company")
    message = f"""Company: {query.company_name}
Role: {query.role_needed}
Required skills: {', '.join(query.required_skills) if query.required_skills else 'Not specified'}
Experience needed: {query.experience_needed}+ years

Please provide:
1. Talent market insight — how competitive is this talent pool right now?
2. Competitive salary range recommendation for this role in 2025
3. Top 3 must-have skills to prioritize when screening candidates
4. One creative sourcing tip to find candidates others miss
5. Estimated time-to-hire for this role"""

    ai_insights = chat_with_claude(message, system, [])

    return {
        "company": query.company_name,
        "role": query.role_needed,
        "market_benchmark_roles": matching_roles,
        "ai_insights": ai_insights
    }

@app.get("/api/health")
def health():
    return {"status": "ok", "rag_jobs": len(rag.jobs), "rag_careers": len(rag.careers)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)