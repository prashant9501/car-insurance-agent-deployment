# 🚗 Car Insurance AI Agent

> A production-ready multi-agent car insurance assistant built with **LangGraph** supervisor pattern, **GPT-6 Luna** (text + image) for damage analysis, **RAG** over IRDAI guidelines, and the actual math used by Indian motor insurers.

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.2-1C3C3C)](https://langchain-ai.github.io/langgraph/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.162-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## ✨ What it does

A chat assistant that handles three real insurance workflows end-to-end:

| Workflow | Example | What happens |
|---|---|---|
| 📋 **Quote** | "Get me a quote for a 2-year-old Maruti Swift, 1 NCB" | Looks up vehicle specs → calculates IDV per IRDAI depreciation slab → returns 3 plans (TP / Comprehensive / Zero-Dep) with full premium breakdown |
| 🛠️ **Claim** | Upload damage photo + "I had an accident" | GPT-6 Luna looks at the photo, classifies damage severity → estimates repair cost → calculates payout (after depreciation + deductible) |
| ❓ **Policy Q&A** | "What is the NCB ladder?" | RAG over IRDAI guidelines + policy terms → grounded, accurate answer |

## 📓 Build it from scratch (Colab notebook)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prashant9501/car-insurance-agent-deployment/blob/main/notebooks/Building_a_Multi_Agent_Car_Insurance_AI_from_Scratch.ipynb)

[`notebooks/Building_a_Multi_Agent_Car_Insurance_AI_from_Scratch.ipynb`](notebooks/Building_a_Multi_Agent_Car_Insurance_AI_from_Scratch.ipynb)
builds the same system step by step — premium maths, tools, the vision tool, RAG, the supervisor and
the LangGraph workflow — with an explanation after every cell. It needs only an OpenAI key stored in
Colab Secrets as `OPENAI_API_KEY`, and runs end to end in under a minute of compute.

## 🏗️ Architecture

```
                    ┌──────────────────┐
                    │   SUPERVISOR     │  ← Classifies intent
                    │   (router)       │
                    └────────┬─────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
   ┌─────────┐        ┌──────────┐        ┌──────────┐
   │  QUOTE  │        │  CLAIMS  │        │  POLICY  │
   │  AGENT  │        │  AGENT   │        │   Q&A    │
   └────┬────┘        └─────┬────┘        └────┬─────┘
        │                   │                   │
        ▼                   ▼                   ▼
   ┌─────────┐        ┌──────────┐        ┌──────────┐
   │ 3 tools │        │ 3 tools  │        │ 1 tool   │
   │ vehicle │        │ verify   │        │ kb_search│
   │ premium │        │ vision   │        │ (RAG)    │
   │ pincode │        │ payout   │        │          │
   └─────────┘        └──────────┘        └──────────┘
                             │
                    ┌────────▼─────────┐
                    │ LangGraph        │
                    │ checkpointer     │  ← Resumable conversations
                    └──────────────────┘
```

**Why supervisor pattern?**
- One cheap classification call → only the right agent's tools and prompt are used
- Each agent has a focused tool set → fewer hallucinations
- Easy to add new agents without touching existing ones

## 🇮🇳 Real Indian insurance domain

This isn't a generic Western insurance demo. It models **actual** Indian motor insurance:

| Concept | Implementation |
|---|---|
| **IDV** (Insured Declared Value) | IRDAI depreciation slabs: 5% → 50% over 5 years |
| **NCB** (No Claim Bonus) | Real ladder: 20% → 25% → 35% → 45% → 50% |
| **OD vs TP** | Own Damage premium (3% of IDV) + Third-Party tariff per engine cc |
| **GST** | 18% GST on motor premiums |
| **Zero Depreciation** | 15% loading on OD; no depreciation on parts at claim time |
| **IRDAI** | Policy terms ingested as RAG knowledge base |

Sample quote for a 2-year-old Maruti Swift with 1 year NCB:
- Third Party Only: **₹4,030**
- Comprehensive: **₹18,757**
- Comprehensive + Zero Dep: **₹20,966**

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph (supervisor + checkpointing) |
| LLM | `gpt-6-luna` for routing, all three agents and photo analysis (`reasoning_effort="none"`; set in `backend/app/llm.py` / `.env`) |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector DB | ChromaDB (persistent) |
| Backend | FastAPI + Pydantic v2 |
| Frontend | Vanilla HTML/CSS/JavaScript (single page) |
| Deployment | Docker + Docker Compose + NGINX |
| Total LOC | ~1,500 |

## 📁 Project Structure

```
car-insurance-agent/
├── backend/
│   ├── app/
│   │   ├── agents/                # 3 specialist agents + supervisor
│   │   │   ├── supervisor.py       # Intent classifier + router
│   │   │   ├── quote_agent.py      # Premium quote agent
│   │   │   ├── claims_agent.py     # FNOL with vision
│   │   │   └── policy_qa_agent.py  # RAG-based Q&A
│   │   ├── tools/                  # 7 tools across 3 categories
│   │   │   ├── quote_tools.py      # vehicle_lookup, calculate_premium, pincode_risk
│   │   │   ├── claims_tools.py     # verify_policy, analyze_damage_image, estimate_payout
│   │   │   └── rag_tools.py        # policy_kb_search
│   │   ├── workflows/
│   │   │   ├── state.py            # AgentState TypedDict
│   │   │   └── main_graph.py       # LangGraph wiring
│   │   ├── services/
│   │   │   ├── premium_calc.py     # IRDAI premium math
│   │   │   └── vehicle_catalog.py  # Mock VAHAN database
│   │   ├── llm.py                  # The chat model every agent uses (gpt-6-luna)
│   │   └── main.py                 # FastAPI app
│   ├── data/
│   │   └── knowledge_base/         # IRDAI guidelines, policy terms, FAQs
│   ├── scripts/
│   │   └── ingest_kb.py            # Load KB into ChromaDB
│   ├── tests/
│   │   ├── test_premium_calc.py
│   │   ├── test_claims_tools.py
│   │   └── test_vehicle_catalog.py
│   ├── pytest.ini
│   ├── requirements.txt
│   └── .env.example
├── notebooks/
│   └── Building_a_Multi_Agent_Car_Insurance_AI_from_Scratch.ipynb  # Colab lab
├── frontend/
│   ├── index.html                  # Chat UI
│   └── static/
│       ├── style.css
│       └── app.js
├── deployment/
│   ├── docker/
│   │   └── docker-compose.yml
│   └── systemd/
│       └── setup_ec2.sh            # One-shot EC2 provisioning (also writes the NGINX config)
├── Dockerfile
├── README.md
└── LICENSE
```

## 🚀 Quick Start (local, with Docker)

### Prerequisites
- Docker with Docker Compose
- An OpenAI API key (~$5–10 covers extensive testing)

```bash
git clone https://github.com/k21academyuk/car-insurance-agent.git
cd car-insurance-agent
cp backend/.env.example backend/.env      # then put your key in OPENAI_API_KEY
cd deployment/docker
docker compose up --build
```

Open `http://localhost:8000/`. The policy knowledge base is loaded into ChromaDB automatically the
first time the app starts (look for `Knowledge base loaded: 19 chunks` in the logs).

## 🚢 Deploy to AWS EC2

Tested on a **t3.medium** with **Ubuntu 24.04 LTS or 26.04 LTS**. Use a 20 GB gp3 root disk and a
security group that allows SSH (22) from your IP and HTTP/HTTPS (80/443).

```bash
# On the EC2 instance, as the ubuntu user:
cd ~
git clone https://github.com/k21academyuk/car-insurance-agent.git
cd car-insurance-agent
sudo bash deployment/systemd/setup_ec2.sh     # ~5 min; ends with "Deployment Complete"

# Add your OpenAI key, then start the service
sudo nano backend/.env
sudo systemctl start car-insurance-agent
```

The app is then live at `http://YOUR-EC2-IP/` (API docs at `/docs`, health check at `/health`).
The knowledge base loads automatically on first start; check with
`cd deployment/docker && sudo docker compose logs app | grep "Knowledge base"`.

The script installs Docker, NGINX (reverse proxy on port 80), certbot (for adding HTTPS later),
UFW (22/80/443) and a systemd service that starts the app on boot. The container listens on
`127.0.0.1:8000` only, so all public traffic goes through NGINX.

## 🎯 Demo Conversations

### Quote flow
```
User: Get me a quote for a Maruti Swift, 2 years old, 1 NCB
Bot:  [calls vehicle_lookup, calculate_premium]
      Here are 3 plans for your Maruti Swift (2 years old, IDV ₹5,20,000):

      🛡️ Third Party Only         ₹4,030 (mandatory minimum)
      🛡️ Comprehensive            ₹18,757 (recommended)
      🛡️ Comprehensive + Zero-Dep ₹20,966 (best protection)

      Which plan would you like to proceed with?
```

### Claims flow with vision
```
User: [uploads damage photo] I hit a divider on the highway
Bot:  [calls analyze_damage_image, estimate_payout]
      I'm sorry to hear about that. Based on the photo:

      Damage assessed:
      • Front bumper: severe
      • Left headlight: moderate
      • Hood: minor

      Estimated repair cost: ₹68,000
      Estimated payout (after 30% dep + ₹2,500 deductible): ₹45,100
      Out of pocket: ₹22,900

      Since payout is < ₹50,000, no surveyor visit needed.
      Claim ID: CLM-2026-08123. We'll email next steps.
```

### Policy Q&A with RAG
```
User: What is NCB and how does it work?
Bot:  [calls policy_kb_search]
      NCB (No Claim Bonus) is a discount on your Own Damage premium for
      claim-free years. It starts at 20% after 1 year and goes up to 50%
      after 5 years. Per IRDAI rules:
      
      • Year 1: 20%
      • Year 2: 25%
      • Year 3: 35%
      • Year 4: 45%
      • Year 5+: 50%
      
      NCB belongs to you, not the car — you can transfer it to a new vehicle.
```

## 🧪 Testing

```bash
# Inside the running container (from deployment/docker):
sudo docker compose exec app python -m pytest tests/ -v
```

## 🔐 Configuration

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `OPENAI_API_KEY` | ✅ | — | OpenAI API access |
| `LANGSMITH_API_KEY` | optional | — | LangSmith tracing |
| `LANGSMITH_TRACING` | optional | `false` | Enable tracing |
| `CORS_ORIGINS` | optional | `*` | Comma-separated origins |
| `CHROMA_PERSIST_DIR` | optional | `/app/chroma_db` | ChromaDB persistence path |
| `OPENAI_CHAT_MODEL` | optional | `gpt-6-luna` | Chat model for every agent and the photo analysis |
| `OPENAI_REASONING_EFFORT` | optional | `none` | Reasoning level for reasoning models; leave empty for other models |

**Why gpt-6-luna?** OpenAI shuts down `gpt-4o` and `gpt-4o-mini` in the API on 14 Jan 2027 and lists
`gpt-6-luna` as the replacement for both. It is a reasoning model, so it rejects `temperature`;
`reasoning_effort="none"` keeps it about as fast as gpt-4o was.

## 💰 Cost Estimate

Running the demo for a month with moderate use (~50 conversations/day):
- OpenAI API: **under $1/month** — gpt-6-luna costs $0.10 / $0.50 per 1M input / output tokens,
  about 1/20th of gpt-4o
- AWS t3.medium EC2: **~$30/month**
- Total: **~$31/month**

A damage-photo analysis measured about **$0.00016** with gpt-6-luna (767 input + ~170 output
tokens), against $0.0038 for the same photo with gpt-4o.

## 🎓 What this demonstrates (capstone-level)

✅ **Hierarchical multi-agent** orchestration with LangGraph
✅ **Multimodal AI** — the model actually looks at the photo and classifies real damage
✅ **RAG** with proper chunking, embeddings, and grounded responses
✅ **Tool calling** with Pydantic-validated schemas
✅ **Stateful conversations** via LangGraph checkpointing
✅ **Domain modeling** — real Indian motor insurance math (IRDAI-compliant)
✅ **Production deployment** — Docker, NGINX, systemd, EC2
✅ **Clean separation** — agents, tools, services, frontend all isolated

## 📜 License

MIT — see [LICENSE](LICENSE).

## 🙏 Built by

Capstone project demonstrating production-grade agentic AI engineering.
