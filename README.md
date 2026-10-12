# testing
# Sauna Fitness Guru

A working Streamlit application for CSV-only sauna history, deterministic analytics, and an optional local Ollama analyst. Exactly four tabs: Data Upload, Consolidated View, Historical Performance, and AI Analyst (Chat).

## Setup and launch

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if it is not already available. From this project directory:

```sh
uv sync
uv run streamlit run app.py
```

`uv sync` creates the project-local `.venv`, uses `.python-version` (Python 3.12), and installs the locked runtime and development dependencies. All project commands use uv; no pip or other environment manager is needed. Open the local URL Streamlit prints, normally http://localhost:8501. Analytics need no API key.

To enable AI, install [Ollama](https://ollama.com/download), start its desktop app (or run `ollama serve`), and install a model:

```sh
ollama pull qwen2.5:7b
cp -n .env.example .env
```

```dotenv
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b
APP_TIMEZONE=America/Edmonton
```

If `.env` already exists, update its Ollama settings using the values above; the copy command preserves it. No API key is required. The default `qwen2.5:7b` is a local model with approximately 4.7 GB weights. Allow additional memory for inference and the context window. You can select another installed model with `OLLAMA_MODEL`; it must support tool calling and structured JSON outputs.

Environment variables take precedence over local `.env` values and Streamlit secrets. Alternatively create `.streamlit/secrets.toml`:

```toml
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:7b"
```

The sidebar allows host and model changes. Enable AI requests there, then explicitly generate a summary or send a chat question. The integration uses Ollama's native [`/api/chat`](https://docs.ollama.com/api/chat) endpoint, [tool calls](https://docs.ollama.com/capabilities/tool-calling), and JSON Schema output via `format`. Only local HTTP loopback hosts are accepted; Ollama must run on the same machine as Streamlit. Remote servers and cloud fallback are not supported. Existing `OPENAI_*` settings are ignored and can be removed from your local configuration.

## CSV contract

Required headers, in any order:

```csv
date,Duration (min),Sauna Temp,Heart Rate Pre-Sauna,Heart Rate Post-Sauna,Steam rounds
```

- Each row is a session; several sessions on one date are allowed.
- Dates must be exact ISO `YYYY-MM-DD`, real calendar dates, supported by pandas nanosecond timestamps, and no later than today's date in `APP_TIMEZONE`.
- Duration and both heart-rate fields must be finite, positive numbers. Duration is minutes; heart rate is bpm.
- Temperature must be numeric and finite. Choose Celsius or Fahrenheit **before processing**; units are never inferred. Temperatures are normalized internally to Celsius. The separate sidebar display setting changes chart, card, and AI units without reinterpreting the file.
- Steam rounds must be a nonnegative integer. Missing values, ambiguous dates, invalid numbers, invalid UTF-8, mismatched field counts, malformed CSV, and empty files are rejected with CSV line numbers and reasons. No invalid session is silently dropped, repaired, or imputed.
- Header whitespace and the UTF-8 BOM are handled. Extra columns appear in the read-only preview but are excluded from analytics and all AI requests.
- Exact duplicates are compared using all original column values, including extra columns. They trigger a warning and remain unless **Remove exact duplicates** is checked. Numerically equivalent but differently written rows are not exact duplicates.

Download the synthetic Celsius sample from Data Upload, or use `data/sample_sauna_history.csv`. It is not automatically loaded. Upload a single CSV, select its interpretation and duplicate option, then click **Validate and use CSV**. A successful upload replaces the active dataset and resets chat, summary, and filter. A failed replacement preserves the last valid history and clearly identifies it. The attempted upload preview remains separate from the active history. **Clear uploaded data** removes the active history, preview, chat, and summary from session state.

## Views and calculations

The shared sidebar date filter applies only to Consolidated View and Historical Performance and defaults to the whole uploaded history. Endpoints are inclusive. Empty results display an empty state without fake averages. Chat uses the full active history unless a question requests a period.

All numerical calculations run in Python. The `utils/portfolio_math.py` filename is retained as requested; it contains sauna calculations, not financial calculations.

- Counts count rows (including retained duplicates), not unique dates. Total minutes sums session durations.
- All averages are arithmetic means across individual sessions, never means of period means. Historical cards display one decimal place while calculations retain precision.
- Paired heart-rate change is each session's post reading minus its pre reading in bpm; positive values mean the post reading was higher. The mean uses all paired sessions.
- Weekly groups are actual Monday dates, with Sunday as the final day. December and January sessions can belong to the same week, and weeks in different years remain distinct. Monthly groups start on the first day of the month.
- The trend selector shows one metric and unit per chart. Periods without sessions show zero for counts and total minutes, and missing averages. Tooltips include contributing session counts and partial-period status.
- **Maximum sessions per week** is the observed largest count within the selected range, not a frequency recommendation. All ties are shown and filter-truncated weeks are labeled partial.
- Consistency context includes weeks with/without records and the longest interval between distinct recorded session dates. Missing records do not prove that no sauna use occurred.
- Comparison queries report both inclusive periods, sample sizes, session averages, and Python-calculated differences (second period minus first). Changes in average readings over time are distinguished from within-session pre/post differences. Post-session measurement timing is unknown.

For an independent check, the sample's September 2026 has **6 sessions, 93 minutes, average duration 15.5 min, average temperature 80°C/176°F, average pre/post heart rates 64.5/84.5 bpm, mean paired change +20 bpm, and average steam rounds 10/6**. The week beginning December 29, 2025 has **4 sessions**; it includes January 1 and 4, 2026. Automated tests assert these values.

## AI and source grounding

AI makes requests only from **Generate summary**, **Refresh summary**, or a submitted chat message. The sidebar privacy acknowledgment must be checked first. Summary requests return 2–3 sentences through structured output; the result survives reruns and is invalidated by changes in data, units, range, model, reference content, or today's date. Source links appear outside the summary.

`data/sauna_reference.json` is a concise, paraphrased implementation-time snapshot of [FoundMyFitness's sauna topic](https://www.foundmyfitness.com/topics/sauna), reviewed October 10, 2026; the page displayed an August 10, 2026 update. It records section names, claims, limitations, and a study DOI checked in the source bibliography (publisher full text was unavailable). The actual snapshot excerpts, not just a URL, accompany requests. It is bundled locally, so runtime source retrieval is unnecessary and analytics do not depend on a website being available. Missing or invalid reference files disable research recommendations through the AI instructions and show a disclosure.

The analyst separates personal observations from general research, distinguishes associations from causation, and avoids applying findings indiscriminately across sauna types or people. It does not diagnose, prescribe treatment, infer improved cardiovascular fitness, predict health outcomes, or promise disease-risk reductions. Six-month suggestions emphasize consistent tracking, standardized measurement timing, and reviewing tolerance; they do not automatically increase heat, duration, or frequency. The recommendation window runs from the application's current local date to the date six calendar months later, with month-end adjustment.

A named month without a year uses the only year containing that month, or asks for a year when multiple years contain it. Requested dates without records are explicitly identified. The model receives Python-verified raw fields, metrics, date coverage, weekly/monthly aggregates, and paired differences. It explains results rather than calculating answers itself.

For large datasets, context explicitly switches to structured summaries and predefined `query_data` functions: session pages, scoped statistics, weekly/monthly pages, named-month resolution, and period comparisons. Pages contain at most 100 records; every row and metric remains queryable and omission is disclosed. The function name and arguments are validated; generated Python, SQL, and shell code are never executed. Requests allow at most six model rounds, three queries per round, a 100,000-character initial budget and a 140,000-character subsequent budget. Oversized conversations or broad multi-page tasks return an actionable message to narrow the question or clear chat, never silently truncate history. Custom model context limits may be lower than this budget and produce a handled API error.

## Privacy, reliability, and limitations

Uploaded files and chat live in Streamlit session memory, without disk persistence or shared caches. Refreshing/disconnecting can lose session state. No API keys, raw personal data, or full payloads are logged. Secrets, `.env`, `.venv`, uploads, and personal CSV files are excluded by `.gitignore`. Streamlit usage telemetry is disabled. Personal files outside those patterns must also be kept out of version control.

Before the first AI request, the app explains that relevant measurements, computed metrics, and messages are sent to the local Ollama server. Data-query responses can include session rows. The HTTP client disables environment proxies and redirects. With a locally installed model, inference stays on the machine running Streamlit. Clearing app state resets chat and summaries; Ollama's own process and configuration are managed separately. Protect any hosted deployment with suitable access controls.

Ollama requests use a 120-second HTTP timeout with no automatic retries, allowing time for local model loading. Connection failures, timeouts, missing models, unsupported capabilities, incomplete responses, and malformed output show safe messages; analytics stay usable. Start Ollama and run `ollama list` to check installed model names. If generation is slow, choose a smaller model. The app sizes the context window to each request (4,096–65,536 tokens); local memory requirements increase with context size. The default Qwen2.5 model does not use a separate thinking phase; gpt-oss models use low reasoning effort. There is no automatic live research refresh. AI answers can still be mistaken, so review claims against the verified data. Tests mock Ollama and do not need a running model. Upload size is capped at 20 MB. Numerical inputs use floating-point arithmetic; extraordinarily large values outside practical session measurements can exceed arithmetic representability.

## Verification and project layout

```sh
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Tests cover schema/value rejection, unit conversion, dates and future dates, exact duplicate handling, several sessions per date, year-crossing weeks, ties/partial weeks, session-weighted means, paired differences, empty filters, verified month/period queries, large-data pagination, reference fallback, local host validation, mocked API failures, Ollama tool-call round trips, summary invalidation, four-tab UI rendering, explicit request triggering, failed replacements, and clearing state.

`app.py` owns configuration, shared state/settings, and tab orchestration. `components/` renders the four tabs. `utils/data_processing.py` validates and filters CSV data. `utils/portfolio_math.py` calculates all analytics. `utils/llm_agent.py` owns reference loading, verified context, prompts, query validation, and API handling. `data/` contains only the synthetic example and reference snapshot. Dependencies are committed through `pyproject.toml`, `uv.lock`, and `.python-version`.
