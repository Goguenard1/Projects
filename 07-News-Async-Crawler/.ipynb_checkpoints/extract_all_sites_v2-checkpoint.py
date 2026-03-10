# =============================================================================
# EXTRACTION DES TITRES - MULTI-SITES CLIMAT/ÉNERGIE (V2 - OPTIMISÉ + DEBUG)
# 5 sites : Carbon Pulse, Climate Change News, PV Tech, SEforALL, Africa Energy
#
# STRATÉGIE :
#   1. Extraction CSS d'abord (instantané, gratuit)
#   2. Si CSS < 3 titres → fallback LLM avec pré-filtrage
#   3. Debug complet : prompt, réponse brute, tokens/chunks dans le terminal
#
# =============================================================================

# -----------------------------------------------------------------------------
# ÉTAPE 1 : Configuration de l'event loop pour Windows
# -----------------------------------------------------------------------------
import asyncio
import sys

if sys.platform == 'win32':
    loop = asyncio.ProactorEventLoop()
    asyncio.set_event_loop(loop)
    print("✅ Étape 1 : ProactorEventLoop configuré pour Windows")
else:
    print("✅ Étape 1 : Système non-Windows")

# -----------------------------------------------------------------------------
# ÉTAPE 2 : Imports
# -----------------------------------------------------------------------------
from crawl4ai import (
    AsyncWebCrawler, CrawlerRunConfig, BrowserConfig,
    CacheMode, LLMConfig, JsonCssExtractionStrategy
)
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from pydantic import BaseModel, Field
from typing import List
import json
import time
import textwrap

print("✅ Étape 2 : Imports réussis")

# -----------------------------------------------------------------------------
# ÉTAPE 3 : Configuration des sites avec sélecteurs CSS PRÉCIS
# -----------------------------------------------------------------------------
# Sélecteurs basés sur l'analyse du HTML réel de chaque site.
# PV Tech : les articles sont dans des <h2> avec liens directs
# SEforALL : structure Drupal avec .views-row et h4 > a
# Carbon Pulse, CCN, Africa Energy : 403 en fetch simple,
#   → sélecteurs génériques larges, le LLM prendra le relais si besoin

WEBSITES = [
    {
        "url": "https://carbon-pulse.com/",
        "name": "Carbon Pulse",
        "schema": {
            "name": "Carbon Pulse Articles",
            "baseSelector": "article, .post, .entry, [class*='article'], [class*='post-item'], [class*='story']",
            "fields": [
                {"name": "title", "selector": "h2 a, h3 a, .entry-title a, .post-title a, h2, h3", "type": "text"},
                {"name": "link", "selector": "h2 a, h3 a, .entry-title a, .post-title a, a[href]", "type": "attribute", "attribute": "href"}
            ]
        }
    },
    {
        "url": "https://www.climatechangenews.com/",
        "name": "Climate Change News",
        "schema": {
            "name": "CCN Articles",
            "baseSelector": "article, .post, [class*='article'], [class*='story'], [class*='card'], .entry",
            "fields": [
                {"name": "title", "selector": "h2 a, h3 a, .entry-title a, h2, h3, a[rel='bookmark']", "type": "text"},
                {"name": "link", "selector": "h2 a, h3 a, .entry-title a, a[rel='bookmark']", "type": "attribute", "attribute": "href"}
            ]
        }
    },
    {
        # PV Tech : structure confirmée par analyse HTML réelle
        # Les titres sont dans des h2 > a[href] pointant vers /pv-tech.org/...
        "url": "https://www.pv-tech.org/",
        "name": "PV Tech",
        "schema": {
            "name": "PV Tech Articles",
            "baseSelector": "h2",
            "fields": [
                {"name": "title", "selector": "a", "type": "text"},
                {"name": "link", "selector": "a", "type": "attribute", "attribute": "href"}
            ]
        }
    },
    {
        # SEforALL : structure Drupal confirmée par analyse HTML réelle
        # Les articles sont des blocs avec image + h4 > a pour le titre
        "url": "https://www.seforall.org/news-and-events/news",
        "name": "SEforALL",
        "schema": {
            "name": "SEforALL Articles",
            "baseSelector": ".views-row, [class*='node--type-news'], [class*='teaser'], [class*='card']",
            "fields": [
                {"name": "title", "selector": "h4 a, h3 a, h2 a, a[href*='/news/']", "type": "text"},
                {"name": "link", "selector": "h4 a, h3 a, h2 a, a[href*='/news/']", "type": "attribute", "attribute": "href"}
            ]
        }
    },
    {
        "url": "https://www.africa-energy.com/news-centre",
        "name": "Africa Energy",
        "schema": {
            "name": "Africa Energy Articles",
            "baseSelector": "article, .post, [class*='news'], [class*='article'], [class*='card'], [class*='item'], .views-row",
            "fields": [
                {"name": "title", "selector": "h2 a, h3 a, h4 a, h2, h3, a[href]", "type": "text"},
                {"name": "link", "selector": "h2 a, h3 a, h4 a, a[href]", "type": "attribute", "attribute": "href"}
            ]
        }
    }
]

print(f"✅ Étape 3 : {len(WEBSITES)} sites configurés")

# -----------------------------------------------------------------------------
# ÉTAPE 4 : Schéma Pydantic (pour le fallback LLM)
# -----------------------------------------------------------------------------
class Article(BaseModel):
    title: str = Field(..., description="The article headline/title")

class ArticleList(BaseModel):
    articles: List[Article] = Field(..., description="List of news articles")

print("✅ Étape 4 : Schéma Pydantic défini")

# -----------------------------------------------------------------------------
# ÉTAPE 5 : Configuration LLM (Ollama / qwen3:4b)
# -----------------------------------------------------------------------------
llm_config = LLMConfig(
    provider="ollama/qwen3:4b",
    api_token="ollama",
    base_url="http://localhost:11434"
)
print("✅ Étape 5 : LLMConfig créé (qwen3:4b)")

# -----------------------------------------------------------------------------
# ÉTAPE 6 : Configurations du crawler
# -----------------------------------------------------------------------------

# --- Browser : text_mode pour performance ---
browser_config = BrowserConfig(
    headless=True,
    text_mode=True,       # Pas d'images/CSS/fonts → plus rapide
    verbose=False
)

# --- Markdown Generator avec PruningContentFilter ---
md_generator = DefaultMarkdownGenerator(
    content_filter=PruningContentFilter(
        threshold=0.4,
        threshold_type="fixed",
        min_word_threshold=3
    )
)

# Le prompt LLM — compact, anglais, /no_think pour qwen3
LLM_INSTRUCTION = """/no_think
Extract all news article titles from this page.
Return ONLY real news/press article headlines.
Ignore: menus, ads, navigation links, buttons, footers, sidebars, author names, dates, categories.
"""

print("✅ Étape 6 : Configurations prêtes")

# -----------------------------------------------------------------------------
# ÉTAPE 7 : Fonctions utilitaires
# -----------------------------------------------------------------------------

def print_separator(char="─", length=70):
    print(char * length)

def print_debug_header(label):
    print(f"\n  {'🔍 DEBUG':} {label}")
    print_separator("·", 60)

def clean_css_results(raw_data, site_name):
    """Nettoie les résultats CSS : déduplique et filtre le bruit."""
    titles = []
    seen = set()
    
    noise_patterns = [
        "menu", "home", "about", "contact", "subscribe", "login", "sign up",
        "search", "read more", "load more", "next", "previous",
        "cookie", "privacy", "terms", "follow us", "share", "tweet",
        "facebook", "twitter", "linkedin", "instagram", "youtube",
        "newsletter", "copyright", "all rights", "powered by",
        "skip to content", "back to top", "close", "accept",
        "advertising", "sign in", "my account", "premium",
        "subscribe to", "market research", "events", "webinars",
        "special reports", "sponsored", "solar media",
    ]
    
    for item in raw_data:
        title = item.get("title", "").strip()
        
        if not title:
            continue
        if len(title) < 20:
            continue
        if len(title) > 300:
            continue
        if title.lower() in seen:
            continue
        if any(noise in title.lower() for noise in noise_patterns):
            continue
        if title.count(" ") < 3:      # Moins de 4 mots
            continue
        
        seen.add(title.lower())
        titles.append(title)
    
    return titles


def parse_llm_results(extracted_content):
    """Parse les résultats JSON du LLM."""
    try:
        extracted = json.loads(extracted_content)
        
        if isinstance(extracted, list):
            if len(extracted) > 0 and isinstance(extracted[0], dict) and 'articles' in extracted[0]:
                articles = extracted[0]['articles']
            else:
                articles = extracted
        elif isinstance(extracted, dict):
            articles = extracted.get('articles', [])
        else:
            articles = []
        
        titles = []
        seen = set()
        for article in articles:
            if isinstance(article, dict):
                title = article.get('title', '').strip()
            else:
                title = str(article).strip()
            
            if title and len(title) > 15 and title.lower() not in seen:
                seen.add(title.lower())
                titles.append(title)
        
        return titles
    except json.JSONDecodeError as e:
        print(f"    ❌ Erreur parsing JSON : {e}")
        return []


# -----------------------------------------------------------------------------
# ÉTAPE 8 : Fonction d'extraction principale (hybride + debug)
# -----------------------------------------------------------------------------

async def extract_titles_from_site(crawler, site_info):
    """
    Extraction hybride avec debug complet dans le terminal.
    Phase 1 : CSS (rapide)
    Phase 2 : LLM fallback (avec logs du prompt, réponse, tokens)
    """
    url = site_info["url"]
    name = site_info["name"]
    
    print(f"\n{'═'*70}")
    print(f"  🌐 {name}")
    print(f"  {url}")
    print(f"{'═'*70}")
    
    # ══════════════════════════════════════════════
    # PHASE 1 : Extraction CSS
    # ══════════════════════════════════════════════
    print(f"\n  📋 PHASE 1 — Extraction CSS (sans LLM)")
    print_separator()
    
    try:
        css_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            word_count_threshold=3,
            excluded_tags=["script", "style", "nav", "footer", "header", "aside", "form"],
            exclude_external_links=True,
            exclude_external_images=True,
            exclude_social_media_links=True,
            remove_overlay_elements=True,
            page_timeout=45000,
            extraction_strategy=JsonCssExtractionStrategy(site_info["schema"])
        )
        
        result = await crawler.arun(url=url, config=css_config)
        
        if result.success and result.extracted_content:
            raw_data = json.loads(result.extracted_content)
            print(f"  CSS brut : {len(raw_data)} éléments trouvés")
            
            titles = clean_css_results(raw_data, name)
            print(f"  Après filtrage : {len(titles)} titres valides")
            
            if len(titles) >= 3:
                print(f"  ✅ CSS suffisant ({len(titles)} ≥ 3) → pas besoin du LLM")
                return {
                    "url": url, "name": name, "titles": titles,
                    "method": "CSS", "success": True
                }
            else:
                print(f"  ⚠️  CSS insuffisant ({len(titles)} < 3) → fallback LLM")
                # Debug : montrer ce que CSS a trouvé
                if titles:
                    print(f"  Titres CSS trouvés :")
                    for t in titles:
                        print(f"    • {t[:80]}...")
        else:
            if result.success:
                print(f"  ⚠️  Crawl OK mais pas de contenu extrait → fallback LLM")
            else:
                error_msg = getattr(result, 'error_message', 'Unknown')
                print(f"  ⚠️  Crawl échoué : {error_msg} → fallback LLM")
                
    except Exception as e:
        print(f"  ⚠️  Erreur CSS : {e} → fallback LLM")
    
    # ══════════════════════════════════════════════
    # PHASE 2 : Fallback LLM (avec debug complet)
    # ══════════════════════════════════════════════
    print(f"\n  🤖 PHASE 2 — Extraction LLM (qwen3:4b)")
    print_separator()
    
    try:
        # Créer la stratégie LLM
        llm_strategy = LLMExtractionStrategy(
            llm_config=llm_config,
            schema=ArticleList.model_json_schema(),
            extraction_type="schema",
            instruction=LLM_INSTRUCTION,
            input_format="fit_markdown",
            chunk_token_threshold=1500,
            overlap_rate=0.05,
            apply_chunking=True,
            extra_args={
                "temperature": 0.0,
                "max_tokens": 2000
            }
        )
        
        llm_run_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            word_count_threshold=3,
            excluded_tags=["script", "style", "nav", "footer", "header", "aside", "form"],
            exclude_external_links=True,
            exclude_external_images=True,
            exclude_social_media_links=True,
            remove_overlay_elements=True,
            page_timeout=45000,
            markdown_generator=md_generator,
            extraction_strategy=llm_strategy
        )
        
        # --- CRAWL ---
        print(f"  ⏳ Crawl + extraction en cours...")
        llm_start = time.time()
        result = await crawler.arun(url=url, config=llm_run_config)
        llm_elapsed = time.time() - llm_start
        print(f"  ⏱️  Temps LLM total : {llm_elapsed:.1f}s")
        
        if result.success:
            # ─────────────────────────────────────
            # DEBUG : Contenu envoyé au LLM
            # ─────────────────────────────────────
            print_debug_header("CONTENU ENVOYÉ AU LLM (fit_markdown)")
            
            # Récupérer le fit_markdown si disponible
            fit_md = None
            if hasattr(result, 'markdown') and result.markdown:
                if hasattr(result.markdown, 'fit_markdown') and result.markdown.fit_markdown:
                    fit_md = result.markdown.fit_markdown
                elif hasattr(result.markdown, 'raw_markdown'):
                    fit_md = result.markdown.raw_markdown
            
            if fit_md:
                md_lines = fit_md.strip().split('\n')
                md_tokens_approx = len(fit_md.split())  # approximation grossière
                print(f"    Taille : {len(fit_md)} chars, ~{md_tokens_approx} mots, {len(md_lines)} lignes")
                print(f"    ── Début (premières 15 lignes) ──")
                for line in md_lines[:15]:
                    print(f"    │ {line[:100]}")
                if len(md_lines) > 15:
                    print(f"    │ ... ({len(md_lines) - 15} lignes supplémentaires)")
                print(f"    ── Fin ──")
            else:
                print(f"    (fit_markdown non disponible)")
            
            # ─────────────────────────────────────
            # DEBUG : Prompt envoyé
            # ─────────────────────────────────────
            print_debug_header("PROMPT / INSTRUCTION")
            for line in LLM_INSTRUCTION.strip().split('\n'):
                print(f"    │ {line}")
            
            # ─────────────────────────────────────
            # DEBUG : Tokens & Chunks
            # ─────────────────────────────────────
            print_debug_header("TOKENS & CHUNKS")
            try:
                # show_usage() affiche dans stdout, on le capture
                usage_info = llm_strategy.show_usage()
                if usage_info:
                    print(f"    {usage_info}")
                else:
                    print(f"    (usage info affiché ci-dessus par show_usage)")
            except Exception as e:
                print(f"    (show_usage non disponible : {e})")
            
            # ─────────────────────────────────────
            # DEBUG : Réponse brute du LLM
            # ─────────────────────────────────────
            print_debug_header("RÉPONSE BRUTE DU LLM")
            
            if result.extracted_content:
                raw_response = result.extracted_content
                print(f"    Taille réponse : {len(raw_response)} chars")
                # Afficher la réponse (tronquée si trop longue)
                if len(raw_response) > 2000:
                    print(f"    ── Début (2000 premiers chars) ──")
                    print(textwrap.indent(raw_response[:2000], "    │ "))
                    print(f"    │ ... (tronqué, {len(raw_response) - 2000} chars restants)")
                else:
                    print(textwrap.indent(raw_response, "    │ "))
                print(f"    ── Fin ──")
                
                # Parser les résultats
                titles = parse_llm_results(raw_response)
                
                if titles:
                    print(f"\n  ✅ LLM : {len(titles)} titres extraits en {llm_elapsed:.1f}s")
                    return {
                        "url": url, "name": name, "titles": titles,
                        "method": "LLM", "success": True,
                        "llm_time": llm_elapsed
                    }
                else:
                    print(f"\n  ❌ LLM : réponse reçue mais 0 titres parsés")
                    return {
                        "url": url, "name": name, "titles": [],
                        "method": "LLM", "success": False,
                        "error": "No titles parsed from LLM response"
                    }
            else:
                print(f"    (aucun contenu extrait)")
                return {
                    "url": url, "name": name, "titles": [],
                    "method": "LLM", "success": False,
                    "error": "No extracted_content"
                }
        else:
            error_msg = getattr(result, 'error_message', 'Unknown')
            print(f"  ❌ Crawl LLM échoué : {error_msg}")
            return {
                "url": url, "name": name, "titles": [],
                "method": "LLM", "success": False,
                "error": error_msg
            }
            
    except Exception as e:
        print(f"  ❌ Exception LLM : {e}")
        import traceback
        traceback.print_exc()
        return {
            "url": url, "name": name, "titles": [],
            "method": "LLM", "success": False, "error": str(e)
        }


# -----------------------------------------------------------------------------
# ÉTAPE 9 : Fonction principale
# -----------------------------------------------------------------------------
async def main():
    all_results = []
    start_time = time.time()
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        for i, site in enumerate(WEBSITES, 1):
            print(f"\n{'▓'*70}")
            print(f"  📍 Site {i}/{len(WEBSITES)}")
            print(f"{'▓'*70}")
            
            result = await extract_titles_from_site(crawler, site)
            all_results.append(result)
            
            if i < len(WEBSITES):
                print(f"\n  ⏳ Pause de 3 secondes...")
                await asyncio.sleep(3)
    
    elapsed = time.time() - start_time
    return all_results, elapsed


# -----------------------------------------------------------------------------
# ÉTAPE 10 : Exécution et résumé
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "█" * 70)
    print("  🚀 EXTRACTION MULTI-SITES — CLIMAT/ÉNERGIE")
    print("  Stratégie : CSS → LLM fallback (qwen3:4b)")
    print("  Mode debug : prompt + réponse + tokens")
    print("█" * 70)
    
    results, elapsed = asyncio.run(main())
    
    # ═══════════════════════════════════════
    # RÉSUMÉ FINAL
    # ═══════════════════════════════════════
    print("\n" + "═" * 70)
    print("  📊 RÉSUMÉ FINAL")
    print("═" * 70)
    
    total_titles = 0
    css_sites = 0
    llm_sites = 0
    failed_sites = 0
    
    for result in results:
        name = result["name"]
        method = result.get("method", "?")
        
        if result["success"]:
            count = len(result["titles"])
            total_titles += count
            
            if method == "CSS":
                css_sites += 1
                icon = "⚡"
            else:
                llm_sites += 1
                llm_time = result.get("llm_time", 0)
                icon = "🤖"
            
            time_str = f" ({result.get('llm_time', 0):.1f}s)" if method == "LLM" else ""
            print(f"\n  {icon} {name} [{method}]{time_str} : {count} titres")
            print("  " + "─" * 50)
            for i, title in enumerate(result["titles"], 1):
                print(f"     {i:2}. {title}")
        else:
            failed_sites += 1
            print(f"\n  ❌ {name} : ÉCHEC — {result.get('error', 'Unknown')}")
    
    print(f"\n{'═'*70}")
    print(f"  📈 TOTAL : {total_titles} titres de {len(WEBSITES)} sites")
    print(f"  ⚡ CSS : {css_sites} sites  |  🤖 LLM : {llm_sites} sites  |  ❌ Échecs : {failed_sites}")
    print(f"  ⏱️  Temps total : {elapsed:.1f}s")
    print(f"{'═'*70}")
    
    # Sauvegarder .txt
    output_file = "titres_extraits.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("TITRES D'ARTICLES — ACTUALITÉS CLIMAT/ÉNERGIE\n")
        f.write(f"Méthode : Hybride CSS → LLM fallback (qwen3:4b)\n")
        f.write(f"Date : {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write("=" * 60 + "\n\n")
        
        for result in results:
            f.write(f"\n{'='*60}\n")
            f.write(f"SOURCE : {result['name']} [{result.get('method', '?')}]\n")
            f.write(f"URL : {result['url']}\n")
            f.write(f"{'='*60}\n\n")
            
            if result["success"] and result["titles"]:
                for i, title in enumerate(result["titles"], 1):
                    f.write(f"{i}. {title}\n")
            else:
                f.write(f"Aucun titre extrait — {result.get('error', '')}\n")
        
        f.write(f"\n\nTOTAL : {total_titles} titres en {elapsed:.1f}s\n")
    
    # Sauvegarder .json
    output_json = "titres_extraits.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n  💾 Fichiers sauvegardés :")
    print(f"     • {output_file}")
    print(f"     • {output_json}")
    print(f"  ✅ TERMINÉ")
