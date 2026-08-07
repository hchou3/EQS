---
name: project-architecture
description: EQS project structure — FastAPI backend (api/) and Next.js 14 frontend (react/my-app/). Key layers and relationships.
metadata:
  type: project
---

EQS is a clinical AI bias-detection platform with two separate apps:

**Backend (api/)**
- Entry: `api/api.py` — FastAPI app mounting 5 routers
- Routers: `csv_upload`, `bias_metrics`, `csv_train`, `csv_shap`, `job_polling`
- Config singleton: `api/config.py` → `settings` object (pydantic-settings)
- Session store: `api/routes/sessions.py` → in-memory `SessionStore` singleton
- Job store: `api/routes/jobs.py` → in-memory `JobStore` singleton
- AI layer: `api/ai/csv/` — `csv_processing.py` (CSVData class + bias analysis), `csv_training.py` (LightGBM + SHAP), `classes.py` (dataclasses), `statistics.py` (dev-only diagnostics re-export)
- LLM: `api/ai/pretraining_tools/pretraining.py` (Gemini via `google.genai`), `prompt.py` (EHR + CSV prompts)
- Dead modules: `api/ai/ehr_processing.py`, `api/ai/ehr_labels.py` — NLP-based EHR bias detection; never imported by any live route

**Frontend (react/my-app/)**
- Entry: `app/page.tsx` renders `<Header>` + `<Dashboard>`
- Dashboard orchestrates FileUpload → Chatbox flow; stores chats/messages in local state
- API client: `lib/api/chat.ts` — **all stubs**, no real backend calls wired yet
- Components each have a barrel `index.ts`

**Known issues to fix (from this audit):**
- `_bundle_key()` is duplicated in `csv_train.py` and `csv_shap.py`
- `pydantic_settings` re-imported in `csv_upload.py` (should use `from ..config import settings` only)
- `MODELS` list in `pretraining.py` is unused
- `GroupDisparityPoint` and `ErrorResponse` in `models.py` have zero callers in routes
- `DisparityComparison` dataclass in `classes.py` has zero callers
- `statistics.py` is never imported by any route
- `summarise_dataset()`, `get_consolidated_report()`, `analyze_dataframe()` — only in module, no live route callers
- `SessionStore.purge_expired()` defined but never scheduled/called
- EHR modules use bare `from ehr_labels import ...` (broken import path) and have 15+ debug print() calls
- Frontend `getConversationHistory` exported but only appears in a comment; chat.ts is entirely stubs
- `react/.env` contains API keys for Mistral, Google, OpenAI — not used by the Next.js app (keys live in `api/.env` too)
- `FileUpload` accepts `.csv,.json,.txt,.pdf` but the backend only accepts `.csv`
