# 📰 Climate & Energy Newsletter Pipeline — Africa / Ivory Coast

End-to-end automated pipeline for building a weekly newsletter on climate and energy news, with a focus on Africa and Côte d'Ivoire. From web scraping to AI-powered article selection and newsletter generation.

## Overview

This project automates the full editorial workflow of a specialized newsletter:

1. **Extract** article titles from 5 climate/energy news sources
2. **Score** each article on regional relevance and media resonance
3. **Classify** articles using a trained LDA model (keep/reject)
4. **Visualize** the thematic landscape with NLP word clouds
5. **Generate** a complete newsletter draft in French using an LLM

## Architecture

```
                          ┌─────────────────┐
                          │  5 News Sources  │
                          │  (Carbon Pulse,  │
                          │  PV Tech, etc.)  │
                          └────────┬────────┘
                                   │
                          ┌────────▼────────┐
                          │  STEP I: Extract │
                          │  Crawl4AI +      │
                          │  Playwright      │
                          └────────┬────────┘
                                   │
                          titres_extraits.txt
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                     │
     ┌────────▼────────┐ ┌────────▼────────┐           │
     │ STEP II.1: LLM  │ │ STEP II.2: Web  │           │
     │ Region scoring  │ │ DuckDuckGo      │           │
     │ (Groq API)      │ │ outlet counting │           │
     └────────┬────────┘ └────────┬────────┘           │
              │                    │                     │
              └──────────┬─────────┘                     │
                         │                               │
                 scores_final.csv                        │
                         │                               │
            ┌────────────┼────────────┐                  │
            │            │            │                  │
   ┌────────▼───┐ ┌──────▼─────┐ ┌───▼──────────┐      │
   │ STEP III:  │ │ STEP IV:   │ │ STEP V:      │      │
   │ LDA Model  │ │ WordCloud  │ │ Newsletter   │      │
   │ Train/     │ │ NLP Viz    │ │ Generation   │      │
   │ Predict    │ │            │ │ (Groq + DDG) │      │
   └────────────┘ └────────────┘ └──────────────┘
```

## Scoring System

Each article receives a score out of 100 based on two criteria:

### Region Relevance (0–50)

| Score | Region |
|-------|--------|
| 50 | Côte d'Ivoire (Abidjan, CIPREL, CI-Energies) |
| 40 | West Africa / Francophone Africa (ECOWAS, UEMOA) |
| 30 | Africa continent (AfDB, African Union) |
| 20 | Global South / developing countries |
| 10 | Global / developed countries |

### Media Resonance (0–50)

Measured via **real web search** (DuckDuckGo), not LLM guessing. The pipeline searches each article title and counts coverage by major outlets:

| Outlets found | Score |
|--------------|-------|
| 5+ major outlets (Reuters, BBC, Bloomberg...) | 50 |
| 3–4 outlets | 40 |
| 1–2 outlets | 30 |
| Trade media only | 20 |
| Original source only | 10 |

**Tracked outlets:** Reuters, Bloomberg, BBC, CNN, Al Jazeera, Le Monde, The Guardian, Financial Times, Jeune Afrique, RFI, France 24, The Africa Report, World Bank, IEA, IRENA, and more.

## LDA Classification

A **Linear Discriminant Analysis** model learns from the scored articles to automate future selections:

- **Training:** Top 10 articles = "kept", rest = "rejected"
- **Features:** region_score, resonance_score, web_resonance
- **Evaluation:** Train/test split (70/30), accuracy, precision, recall, F1, ROC AUC, Leave-One-Out CV
- **Output:** For each new article → KEEP/REJECT prediction with probability %

The model improves over time as more newsletters are produced and labeled.

## Tech Stack

| Technology | Usage |
|-----------|-------|
| **Python 3** | Core language |
| **Crawl4AI** | Async web crawling with Playwright |
| **Ollama** | Local LLM inference (Qwen3, Gemma3) |
| **Groq API** | Cloud LLM for scoring and generation |
| **LiteLLM** | Unified LLM API interface |
| **DuckDuckGo Search** | Web search for media resonance |
| **Scikit-learn** | LDA model training and evaluation |
| **Pandas** | Data manipulation |
| **WordCloud** | NLP visualization |
| **Matplotlib / Seaborn** | Plots and charts |

## Installation

```bash
pip install crawl4ai playwright pydantic litellm duckduckgo-search scikit-learn pandas matplotlib seaborn wordcloud

# Install browser for Crawl4AI
playwright install chromium
crawl4ai-setup

# Optional: local LLM via Ollama
# Download from https://ollama.com
ollama pull qwen3:4b
```

## Usage

1. Run the extraction script from the terminal:
```bash
   python -u extract_all_sites.py
```
2. Open `Projet_Newsletter.ipynb` in JupyterLab
3. Run all cells sequentially — scoring, LDA, word cloud, and newsletter generation are all inside the notebook
## Project Structure

```
07-News-Async-Crawler/
│
├── Projet_Newsletter.ipynb       # Main notebook (full pipeline)
├── extract_all_sites.py          # Step I  — Web scraping
├── score_step1_llm.py            # Step II.1 — LLM scoring
├── score_step2_web_resonance.py  # Step II.2 — Web search resonance
├── step3_lda.py                  # Step III — LDA classification
│
├── titres_extraits.txt           # Extracted article titles
├── scores_llm.csv                # LLM scores (region + resonance)
├── scores_final.csv              # Final scores with web resonance
│
├── lda_model.pkl                 # Trained LDA model
├── lda_analysis.png              # LDA score distribution + feature importance
├── lda_scatter.png               # Region vs web resonance scatter plot
├── wordcloud_kept.png            # Word cloud of selected articles
├── newsletter.md                 # Generated newsletter (French)
│
└── README.md
```

## Sample Output

### Scoring CSV

| Title | Region | Resonance | Web | Final |
|-------|--------|-----------|-----|-------|
| South African rare earths project aims to rival Chinese model | 30 | 15 | 40 | 68 |
| Gulf oil and gas crisis sparks calls for renewables investment | 20 | 20 | 50 | 72 |
| New summit in Colombia seeks to revive stalled UN talks | 10 | 25 | 50 | 68 |

### LDA Prediction

```
🏆 RECOMMENDED FOR NEWSLETTER:
  1. [92.3%] Gulf oil crisis sparks renewables investment calls
  2. [87.1%] South African rare earths vs Chinese model
  3. [84.5%] Colombia summit to revive UN fossil fuel talks

❌ REJECTED:
  [12.4%] Austria targets carbon cost pass-through in power prices
  [8.7%]  California climate disclosure rule could reorder rankings
```

## Configuration

### LLM Providers

The pipeline supports multiple providers via LiteLLM:

```python
# Groq (cloud, fast, free tier)
LLMConfig(provider="groq/llama-3.1-8b-instant", api_token="...")

# Ollama (local, no rate limits)
LLMConfig(provider="ollama/qwen3:4b", api_token="ollama", base_url="http://localhost:11434")

# Google Gemini (cloud, high TPM)
LLMConfig(provider="gemini/gemini-2.5-flash-preview-04-17", api_token="...")
```

### Adding New Sources

Edit `WEBSITES` in `extract_all_sites.py`:

```python
WEBSITES = [
    "https://carbon-pulse.com/",
    "https://www.climatechangenews.com/",
    "https://www.pv-tech.org/",
    "https://www.seforall.org/news-and-events/news",
    "https://www.africa-energy.com/news-centre",
    "https://your-new-source.com/news"  # Add here
]
```

## License

MIT
