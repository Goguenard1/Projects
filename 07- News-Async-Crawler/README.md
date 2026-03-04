# 🌍 Climate & Energy News Scraper

Automated pipeline for extracting article headlines from 6 major climate and energy news sources using async web crawling and LLM-powered extraction.

## Overview

This project scrapes article titles from specialized climate/energy websites and structures them into clean, exportable data. It combines headless browser crawling with LLM-based content extraction to intelligently identify real article headlines while ignoring navigation, ads, and other page noise.

### Sources

| Site | URL |
|------|-----|
| Bloomberg Green | bloomberg.com/green |
| Carbon Pulse | carbon-pulse.com |
| Climate Change News | climatechangenews.com |
| PV Tech | pv-tech.org |
| SEforALL | seforall.org/news-and-events/news |
| Africa Energy | africa-energy.com/news-centre |

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Target URLs │ ──▶ │  Async Web   │ ──▶ │  LLM-based   │ ──▶ │  Structured  │
│  (6 sources) │     │  Crawling    │     │  Extraction  │     │  Output      │
└──────────────┘     │  (Playwright)│     │  (Pydantic)  │     │  (.txt/.json)│
                     └──────────────┘     └──────────────┘     └──────────────┘
```

**Key components:**

- **AsyncWebCrawler** — headless Chromium browser via Playwright for JavaScript-rendered pages
- **LLMExtractionStrategy** — sends scraped content to an LLM to identify and extract article titles
- **Pydantic schema** — validates and structures the extracted data
- **Rate limit handling** — configurable pauses between requests to respect API quotas

## Tech Stack

- **Python 3.10+**
- **Crawl4AI** — async web crawling framework
- **Playwright** — headless browser automation
- **Pydantic** — data validation
- **LiteLLM** — unified LLM API interface
- **asyncio** — asynchronous orchestration

## Supported LLM Providers

The pipeline works with any LiteLLM-compatible provider. Tested with:

| Provider | Model | Free Tier |
|----------|-------|-----------|
| Groq | `groq/llama-3.1-8b-instant` | ✅ 131K TPM |
| Groq | `groq/llama-3.3-70b-versatile` | ✅ 12K TPM |
| Google | `gemini/gemini-2.5-flash-preview-04-17` | ✅ 250K TPM |
| Cerebras | `cerebras/llama-3.3-70b` | ✅ |

## Installation

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/climate-news-scraper.git
cd climate-news-scraper

# Install dependencies
pip install crawl4ai playwright pydantic litellm

# Install browser
playwright install chromium

# Run Crawl4AI setup
crawl4ai-setup
```

## Configuration

Set your LLM API key in the script or as an environment variable:

```python
# Option 1: In the script
GROQ_API_KEY = "your-api-key-here"

# Option 2: Environment variable
import os
os.environ["GROQ_API_KEY"] = "your-api-key-here"
```

Update the `LLMConfig` to match your provider:

```python
llm_config = LLMConfig(
    provider="groq/llama-3.1-8b-instant",  # or gemini/gemini-2.5-flash-preview-04-17
    api_token=GROQ_API_KEY
)
```

## Usage

```bash
python extract_all_sites.py
```

Or in a Jupyter notebook:

```python
# Windows event loop fix (required)
import asyncio, sys
if sys.platform == 'win32':
    loop = asyncio.ProactorEventLoop()
    asyncio.set_event_loop(loop)

# Run the script
%run extract_all_sites.py
```

## Output

Results are saved to `titres_extraits.txt`:

```
============================================================
SOURCE : carbon-pulse.com
URL : https://carbon-pulse.com/
============================================================

1. EU carbon price hits new monthly high amid...
2. Australia launches consultation on carbon...
3. ...
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `UnicodeEncodeError` on Windows | Run `chcp 65001` or set `PYTHONIOENCODING=utf-8` |
| LLM rate limit errors | Increase `asyncio.sleep()` pause or switch to a model with higher TPM |
| Bloomberg returns 0 titles | Expected — Bloomberg uses a paywall that blocks scrapers |
| Garbled emoji in terminal | Cosmetic only — does not affect functionality |

## Project Structure

```
climate-news-scraper/
├── extract_all_sites.py    # Main extraction pipeline
├── titres_extraits.txt     # Output file (generated)
├── README.md
└── requirements.txt
```

## License

MIT
