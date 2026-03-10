# =============================================================================
# EXTRACTION DES TITRES - CLIMAT/ÉNERGIE (V3 - MINIMALISTE)
#
# Approche simple :
#   1. CSS d'abord (gratuit, instantané)
#   2. Si CSS échoue → LLM via Crawl4AI tel quel, sans filtre ni parsing custom
#   3. Modèle : gemma3:1b (léger)
# =============================================================================

import asyncio
import sys

if sys.platform == 'win32':
    asyncio.set_event_loop(asyncio.ProactorEventLoop())

from crawl4ai import (
    AsyncWebCrawler, CrawlerRunConfig, BrowserConfig,
    CacheMode, LLMConfig, JsonCssExtractionStrategy
)
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from pydantic import BaseModel, Field
from typing import List
import json
import time

# -----------------------------------------------------------------------------
# MODÈLE LLM
# -----------------------------------------------------------------------------
LLM = LLMConfig(
    provider="ollama/gemma3:1b",
    api_token="ollama",
    base_url="http://localhost:11434"
)

# -----------------------------------------------------------------------------
# SCHÉMA PYDANTIC (ce qu'on veut que le LLM retourne)
# -----------------------------------------------------------------------------
class Article(BaseModel):
    title: str = Field(..., description="The article headline")

class ArticleList(BaseModel):
    articles: List[Article] = Field(..., description="List of articles")

# -----------------------------------------------------------------------------
# SITES + SÉLECTEURS CSS
# -----------------------------------------------------------------------------
SITES = [
    {
        "url": "https://carbon-pulse.com/",
        "name": "Carbon Pulse",
        "css": {
            "name": "Articles", "baseSelector": "article, .post, [class*='article']",
            "fields": [{"name": "title", "selector": "h2 a, h3 a, h2, h3", "type": "text"}]
        }
    },
    {
        "url": "https://www.climatechangenews.com/",
        "name": "Climate Change News",
        "css": {
            "name": "Articles", "baseSelector": "article, .post, [class*='story']",
            "fields": [{"name": "title", "selector": "h2 a, h3 a, h2, h3", "type": "text"}]
        }
    },
    {
        "url": "https://www.pv-tech.org/",
        "name": "PV Tech",
        "css": {
            "name": "Articles", "baseSelector": "h2",
            "fields": [{"name": "title", "selector": "a", "type": "text"}]
        }
    },
    {
        "url": "https://www.seforall.org/news-and-events/news",
        "name": "SEforALL",
        "css": {
            "name": "Articles", "baseSelector": ".views-row, [class*='card']",
            "fields": [{"name": "title", "selector": "h4 a, h3 a, h2 a", "type": "text"}]
        }
    },
    {
        "url": "https://www.africa-energy.com/news-centre",
        "name": "Africa Energy",
        "css": {
            "name": "Articles", "baseSelector": "article, [class*='news'], [class*='card']",
            "fields": [{"name": "title", "selector": "h2 a, h3 a, h2, h3", "type": "text"}]
        }
    }
]

# -----------------------------------------------------------------------------
# BROWSER
# -----------------------------------------------------------------------------
BROWSER = BrowserConfig(headless=True, text_mode=True, verbose=False)

# -----------------------------------------------------------------------------
# EXTRACTION
# -----------------------------------------------------------------------------
async def extract(crawler, site):
    url = site["url"]
    name = site["name"]

    print(f"\n{'='*60}")
    print(f"🌐 {name} — {url}")
    print(f"{'='*60}")

    # ── PHASE 1 : CSS ──
    print("  📋 CSS...")
    try:
        result = await crawler.arun(
            url=url,
            config=CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                page_timeout=45000,
                extraction_strategy=JsonCssExtractionStrategy(site["css"])
            )
        )
        if result.success and result.extracted_content:
            data = json.loads(result.extracted_content)
            titles = [d["title"].strip() for d in data if d.get("title", "").strip() and len(d["title"].strip()) > 20]
            # Dédupliquer
            seen = set()
            unique = []
            for t in titles:
                if t.lower() not in seen:
                    seen.add(t.lower())
                    unique.append(t)
            if len(unique) >= 3:
                print(f"  ✅ CSS : {len(unique)} titres")
                return {"name": name, "url": url, "titles": unique, "method": "CSS", "ok": True}
            else:
                print(f"  ⚠️ CSS : {len(unique)} titres seulement → LLM")
    except Exception as e:
        print(f"  ⚠️ CSS erreur : {e} → LLM")

    # ── PHASE 2 : LLM ──
    print("  🤖 LLM (gemma3:1b)...")
    try:
        t0 = time.time()
        result = await crawler.arun(
            url=url,
            config=CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                page_timeout=45000,
                extraction_strategy=LLMExtractionStrategy(
                    llm_config=LLM,
                    schema=ArticleList.model_json_schema(),
                    extraction_type="schema",
                    instruction="Extract all news article titles from this page. Return only real news headlines, ignore navigation and ads.",
                    extra_args={"temperature": 0.0, "max_tokens": 2000}
                )
            )
        )
        dt = time.time() - t0

        if result.success and result.extracted_content:
            print(f"  ⏱️ LLM : {dt:.1f}s")

            # Debug : ce que le LLM a reçu
            if hasattr(result, 'markdown') and result.markdown:
                md = getattr(result.markdown, 'raw_markdown', '') or ''
                print(f"  📄 Contenu envoyé : {len(md)} chars, ~{len(md.split())} mots")

            # Debug : réponse brute
            print(f"  📤 Réponse brute ({len(result.extracted_content)} chars) :")
            preview = result.extracted_content[:500]
            for line in preview.split('\n')[:10]:
                print(f"     │ {line[:90]}")

            # Parser — on laisse simple
            raw = json.loads(result.extracted_content)
            articles = []
            if isinstance(raw, list):
                for item in raw:
                    if isinstance(item, dict) and 'articles' in item:
                        articles.extend(item['articles'])
                    elif isinstance(item, dict) and 'title' in item:
                        articles.append(item)
            elif isinstance(raw, dict) and 'articles' in raw:
                articles = raw['articles']

            titles = [a['title'].strip() for a in articles if a.get('title', '').strip()]
            print(f"  ✅ LLM : {len(titles)} titres en {dt:.1f}s")
            return {"name": name, "url": url, "titles": titles, "method": "LLM", "ok": True}
        else:
            print(f"  ❌ LLM échoué")
            return {"name": name, "url": url, "titles": [], "method": "LLM", "ok": False}

    except Exception as e:
        print(f"  ❌ LLM erreur : {e}")
        return {"name": name, "url": url, "titles": [], "method": "LLM", "ok": False}

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
async def main():
    print("\n🚀 EXTRACTION CLIMAT/ÉNERGIE — CSS → LLM (gemma3:1b)\n")
    results = []
    t0 = time.time()

    async with AsyncWebCrawler(config=BROWSER) as crawler:
        for i, site in enumerate(SITES, 1):
            print(f"\n📍 Site {i}/{len(SITES)}")
            r = await extract(crawler, site)
            results.append(r)
            if i < len(SITES):
                await asyncio.sleep(2)

    dt = time.time() - t0

    # ── RÉSUMÉ ──
    print(f"\n\n{'='*60}")
    print(f"📊 RÉSUMÉ — {dt:.0f}s total")
    print(f"{'='*60}")

    total = 0
    for r in results:
        icon = "⚡" if r["method"] == "CSS" else "🤖"
        n = len(r["titles"])
        total += n
        status = f"{n} titres" if r["ok"] else "ÉCHEC"
        print(f"\n  {icon} {r['name']} [{r['method']}] — {status}")
        for i, t in enumerate(r["titles"], 1):
            print(f"     {i:2}. {t}")

    print(f"\n  📈 Total : {total} titres | ⏱️ {dt:.0f}s")

    # Sauvegarder
    with open("titres_extraits.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open("titres_extraits.txt", "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"\n{'='*40}\n{r['name']} [{r['method']}]\n{'='*40}\n")
            for i, t in enumerate(r["titles"], 1):
                f.write(f"{i}. {t}\n")
    print(f"  💾 Sauvegardé : titres_extraits.json + .txt")

if __name__ == "__main__":
    asyncio.run(main())
