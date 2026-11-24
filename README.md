# Veritas Desk: Real-Time Fact Verification & Credibility Analysis

Veritas Desk is a high-reliability news claim verification system combining Google Search grounding, deterministic verification guardrails, transparent input classification, and auxiliary linguistic style analysis.

---

## 1. System Architecture

```text
                     User Input (Headline/Claim + Optional URL)
                                        │
                                        ▼
                     ┌──────────────────────────────────────┐
                     │     1. Transparent Input Gate        │
                     │         (backend/gate.py)            │
                     └──────────────────┬───────────────────┘
                                        │
             ┌──────────────────────────┴──────────────────────────┐
             │                                                     │
   [Non-Claim Types]                                       [Factual Claim]
   • question                                                      │
   • prediction                                                    │
   • opinion                                                       │
   • too_short                                                     │
             │                                                     ▼
             │                                  ┌──────────────────────────────────────┐
             │                                  │   2. Auxiliary Style Indicator       │
             │                                  │       (TF-IDF + LinearSVC)           │
             │                                  │ • Output: "neutral" vs "sensational" │
             │                                  │ • Note: Isolated from truth verdict  │
             │                                  └──────────────────┬───────────────────┘
             │                                                     │
             │                                                     ▼
             │                                  ┌──────────────────────────────────────┐
             │                                  │   3. Gemini Google Search Grounding  │
             │                                  │      (backend/verify_llm.py)         │
             │                                  │ • Injected server clock date         │
             │                                  │ • Google Search tool grounding       │
             │                                  │ • Grounding metadata URL validation  │
             │                                  │ • Structured JSON response schema    │
             │                                  └──────────────────┬───────────────────┘
             │                                                     │
             │                                                     ▼
             │                                  ┌──────────────────────────────────────┐
             │                                  │   4. Deterministic Guardrails        │
             │                                  │ • Empty evidence downgrade           │
             │                                  │ • Time sensitivity (2d / 7d window)  │
             │                                  │ • Stance corroboration check         │
             │                                  │ • Publisher diversity check          │
             │                                  │ • Satire / disreputable URL penalty  │
             │                                  └──────────────────┬───────────────────┘
             │                                                     │
             ▼                                                     ▼
┌──────────────────────────────────────┐        ┌──────────────────────────────────────┐
│  Early Exit: NOT_A_CLAIM Verdict     │        │  Persistent SQLite Cache (history.db)│
│  (Latency < 10ms, No LLM Call)       │        │  (6h time-sensitive / 24h standard)  │
└──────────────────┬───────────────────┘        └──────────────────┬───────────────────┘
                   │                                               │
                   └───────────────────────┬───────────────────────┘
                                           ▼
                            Client JSON Response & UI Render
```

---

## 2. Why the Previous Approach Failed & How This Architecture Fixes It

| # | Root Cause in Previous Version | Failure Mode | How Veritas Desk Fixes It |
|---|---|---|---|
| **1** | Questions, forecasts, and opinions scored as claims | Asking *"Will it rain today in X?"* received 80+ "Likely real" because it was grammatically structured like formal news. | **Input Gate (`backend/gate.py`)**: Interrogatives, modal auxiliaries, future markers (`will`, `going to`), and first-person opinion indicators are routed to `NOT_A_CLAIM` in <10ms without calling external models. |
| **2** | Retrieved articles never filtered by publication date | Outdated articles from months or years ago were cited as "corroborating" real-time weather or breaking events. | **Programmatic Date Injection + Time-Sensitivity Guardrail**: Today's ISO date is injected into the system prompt from the server clock. Claims with `"today"`, `"now"`, or `"this morning"` reject evidence older than 2 days. |
| **3** | Topical similarity counted as factual support | An article merely mentioning "Hyderabad" or "rain" in the past was treated as corroboration for today's forecast. | **Grounded Fact-Checking + Stance Agreement**: Gemini assesses factual stance (`supports`, `refutes`, `neutral`). Guardrail Rule 3 downgrades `SUPPORTED` verdicts lacking direct supporting stance. |
| **4** | TF-IDF style classifier output fed directly into truth score | Well-written falsehoods or satire formatted in formal journalistic style were scored as "Likely real". | **Strict Separation of Style and Truth**: TF-IDF LinearSVC output is strictly isolated into an informational `style_indicator` field (`neutral` vs `sensational`). It never affects verdict or confidence. |

---

## 3. Quickstart

### Prerequisites
- Python 3.10+ (tested on Python 3.14)
- Google Gemini API Key (optional for gate testing; required for live Google Search grounding)

### Environment Setup

1. Clone repository and set up virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Configure environment variables:
   Copy `.env.example` to `.env`:
   ```powershell
   cp .env.example .env
   ```
   Edit `.env`:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   GEMINI_MODEL=gemini-2.5-flash
   ADMIN_API_KEY=your_secure_admin_key
   ENABLE_LEARNING=false
   ```
   *(Note: `.env` is ignored in `.gitignore` to prevent credential leaks).*

### Running the Application

```powershell
.\venv\Scripts\python backend/app.py
```
Open **`http://127.0.0.1:5000`** in your browser.

### Running the Test Suite

```powershell
.\venv\Scripts\pytest
```

### Running the 30-Input Evaluation Benchmark

```powershell
.\venv\Scripts\python backend/scripts/eval_demo.py
```
This runs 30 representative test cases across all categories and saves the benchmark metrics to `models/demo_eval.md`.

---

## 4. API Specification

### `POST /api/analyze`

#### Request Payload
```json
{
  "text": "The Reserve Bank of India kept repo rates steady at 6.5 percent",
  "url": "https://www.thehindu.com"
}
```

#### Response Payload (`200 OK`)
```json
{
  "claim": "The Reserve Bank of India kept repo rates steady at 6.5 percent",
  "input_type": "claim",
  "verdict": "SUPPORTED",
  "confidence": "high",
  "reasoning": "The Reserve Bank of India Monetary Policy Committee kept the benchmark repo rate unchanged at 6.5 percent as confirmed by multiple official news publications.",
  "evidence": [
    {
      "url": "https://www.thehindu.com/business/rbi-monetary-policy",
      "publisher": "The Hindu",
      "title": "RBI keeps repo rate unchanged at 6.5%",
      "published_date": "2024-02-08",
      "stance": "supports",
      "key_sentence": "The Monetary Policy Committee decided unanimously to keep the policy repo rate unchanged at 6.5%."
    },
    {
      "url": "https://reuters.com/world/india/rbi-holds-rates",
      "publisher": "Reuters",
      "title": "India central bank leaves key rate unchanged",
      "published_date": "2024-02-08",
      "stance": "supports",
      "key_sentence": "The Reserve Bank of India left its key repo rate steady at 6.50%."
    }
  ],
  "guardrail_notes": [],
  "style_indicator": {
    "label": "neutral",
    "score": 0.884,
    "note": "Writing style only, not a measure of truth."
  },
  "source_domain": "thehindu.com",
  "source_reputation": "trusted",
  "cached": false,
  "checked_at": "2026-11-21T12:00:00Z",
  "latency_ms": 1140
}
```

#### Non-Claim Early Return Example (`POST /api/analyze`)
**Input:** `"Will it rain today in Hyderabad?"`
```json
{
  "claim": "Will it rain today in Hyderabad?",
  "input_type": "question",
  "verdict": "NOT_A_CLAIM",
  "confidence": "high",
  "reasoning": "This is a question, not a factual claim. Rephrase as a verifiable assertion (e.g. 'It rained in Hyderabad today').",
  "evidence": [],
  "guardrail_notes": [
    "Input classified as question; bypassed external verification."
  ],
  "style_indicator": {
    "label": "neutral",
    "score": 0.512,
    "note": "Writing style only, not a measure of truth."
  },
  "source_domain": null,
  "source_reputation": "unknown",
  "cached": false,
  "checked_at": "2026-11-21T12:00:00Z",
  "latency_ms": 8
}
```

---

## 5. Repository Structure

```text
fake-news-detector/
├── backend/
│   ├── app.py               # Flask application with /api/analyze, /api/history, rate limiting
│   ├── gate.py              # Input classification gate (claims vs questions/predictions/opinions)
│   ├── verify_llm.py        # Gemini verification with Google Search grounding & deterministic guardrails
│   ├── db.py                # SQLite WAL database management, verify_cache, and history persistence
│   ├── preprocess.py        # Text normalization and feature tokenization
│   ├── explain.py           # Token attribution and linguistic feature extraction
│   ├── learn.py             # Safeguarded administrative retraining routines
│   ├── sources.json         # Source reliability dictionary including known satire domains
│   └── scripts/
│       └── eval_demo.py     # 30-input evaluation script generating models/demo_eval.md
├── frontend/
│   ├── index.html           # Semantic verification interface with accessible badges and cards
│   ├── app.js               # Reactive client-side rendering for verdicts, evidence, and guardrails
│   └── style.css            # Dark/light responsive styling, verdict color palette, evidence cards
├── models/
│   ├── model.joblib         # Serialized baseline TF-IDF LinearSVC model bundle (for style indicator)
│   ├── metrics.json         # Baseline model performance metrics
│   └── demo_eval.md         # Generated 30-input benchmark evaluation report
├── tests/
│   ├── test_gate.py         # Unit tests for input classification gate and edge cases
│   ├── test_guardrails.py   # Unit tests for the 5 deterministic guardrails
│   ├── test_verify_llm.py   # Unit tests with mocked Gemini client (hallucination, timeout, caching)
│   ├── test_api.py          # Integration tests for Flask endpoints and contract schemas
│   ├── test_live_gemini.py  # Optional live end-to-end integration test (skipped if key unset)
│   ├── test_fusion.py       # Bayesian fusion unit tests
│   ├── test_explain.py      # Token attribution unit tests
│   ├── test_preprocess.py   # Preprocessing and cleaning unit tests
│   ├── test_regressions.py  # Regression tests for historic failure modes
│   ├── test_retriever.py    # RSS retriever tests
│   └── test_stance.py       # Stance detection tests
├── .env.example             # Template for required environment variables
├── requirements.txt         # Production dependencies (including google-genai, pydantic)
└── README.md                # Project architecture, quickstart, and API documentation
```
