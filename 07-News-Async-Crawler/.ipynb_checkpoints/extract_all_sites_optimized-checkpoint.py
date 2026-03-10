# =============================================================================
# EXTRACTION DES TITRES - MULTI-SITES CLIMAT/ÉNERGIE (OPTIMISÉ)
# 5 sites : Carbon Pulse, Climate Change News, PV Tech, SEforALL, Africa Energy
#
# STRATÉGIE D'OPTIMISATION :
# --------------------------
# 1. APPROCHE HYBRIDE : CSS d'abord (gratuit, instantané), LLM en fallback
#    → Les sélecteurs CSS extraient les titres sans aucun appel LLM
#    → Le LLM n'intervient QUE si les CSS échouent (0 titre trouvé)
#
# 2. PRÉ-FILTRAGE DU CONTENU AVANT LLM :
#    - PruningContentFilter : élimine le bruit (nav, pubs, footers) AVANT
#      que le contenu arrive au LLM → moins de tokens à traiter
#    - input_format="fit_markdown" : le LLM reçoit le markdown filtré,
#      pas le HTML brut → réduction massive des tokens
#    - excluded_tags : supprime script, style, nav, footer, header, aside
#    - exclude_external_links/images : nettoie encore plus
#
# 3. OPTIMISATIONS BROWSER :
#    - text_mode=True : pas de chargement d'images/CSS/fonts → plus rapide
#    - headless=True : pas de rendu graphique
#
# 4. OPTIMISATIONS LLM (quand utilisé en fallback) :
#    - chunk_token_threshold réduit : adapté à un petit modèle local
#    - apply_chunking=True : découpe intelligente du contenu
#    - temperature=0.0 : réponses déterministes
#    - Prompt en anglais + /no_think pour Qwen3
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

print("✅ Étape 2 : Imports réussis")

# -----------------------------------------------------------------------------
# ÉTAPE 3 : Configuration des sites avec leurs sélecteurs CSS
# -----------------------------------------------------------------------------
# Chaque site a un schéma CSS spécifique pour extraction SANS LLM.
# Le champ "css_scope" permet de cibler la zone de contenu principal.
# Si les sélecteurs CSS ne trouvent rien → fallback LLM automatique.

WEBSITES = [
    {
        "url": "https://carbon-pulse.com/",
        "name": "Carbon Pulse",
        "css_scope": None,  # pas de scope spécifique, on prend tout
        "schema": {
            "name": "Carbon Pulse Articles",
            "baseSelector": "article, .post, .entry, .article-item, .story, [class*='article'], [class*='post'], h2 a, h3 a",
            "fields": [
                {"name": "title", "selector": "h2 a, h3 a, h2, h3, .entry-title a, .post-title a, a", "type": "text"},
                {"name": "link", "selector": "h2 a, h3 a, .entry-title a, .post-title a, a", "type": "attribute", "attribute": "href"}
            ]
        }
    },
    {
        "url": "https://www.climatechangenews.com/",
        "name": "Climate Change News",
        "css_scope": None,
        "schema": {
            "name": "CCN Articles",
            "baseSelector": "article, .post, .entry, [class*='article'], [class*='story'], [class*='card']",
            "fields": [
                {"name": "title", "selector": "h2 a, h3 a, h2, h3, .entry-title a, .post-title a, a", "type": "text"},
                {"name": "link", "selector": "h2 a, h3 a, .entry-title a, a", "type": "attribute", "attribute": "href"}
            ]
        }
    },
    {
        "url": "https://www.pv-tech.org/",
        "name": "PV Tech",
        "css_scope": None,
        "schema": {
            "name": "PV Tech Articles",
            "baseSelector": "article, .post, [class*='article'], [class*='card'], [class*='story'], [class*='teaser']",
            "fields": [
                {"name": "title", "selector": "h2 a, h3 a, h2, h3, .entry-title a, a", "type": "text"},
                {"name": "link", "selector": "h2 a, h3 a, a", "type": "attribute", "attribute": "href"}
            ]
        }
    },
    {
        "url": "https://www.seforall.org/news-and-events/news",
        "name": "SEforALL",
        "css_scope": None,
        "schema": {
            "name": "SEforALL Articles",
            "baseSelector": "article, .node, .views-row, [class*='card'], [class*='news'], [class*='teaser']",
            "fields": [
                {"name": "title", "selector": "h2 a, h3 a, h2, h3, .field-title a, a", "type": "text"},
                {"name": "link", "selector": "h2 a, h3 a, a", "type": "attribute", "attribute": "href"}
            ]
        }
    },
    {
        "url": "https://www.africa-energy.com/news-centre",
        "name": "Africa Energy",
        "css_scope": None,
        "schema": {
            "name": "Africa Energy Articles",
            "baseSelector": "article, .post, [class*='news'], [class*='article'], [class*='card'], [class*='item']",
            "fields": [
                {"name": "title", "selector": "h2 a, h3 a, h2, h3, a", "type": "text"},
                {"name": "link", "selector": "h2 a, h3 a, a", "type": "attribute", "attribute": "href"}
            ]
        }
    }
]

print(f"✅ Étape 3 : {len(WEBSITES)} sites configurés avec sélecteurs CSS")

# -----------------------------------------------------------------------------
# ÉTAPE 4 : Schéma Pydantic (pour le fallback LLM uniquement)
# -----------------------------------------------------------------------------
class Article(BaseModel):
    title: str = Field(..., description="The article headline/title")

class ArticleList(BaseModel):
    articles: List[Article] = Field(..., description="List of news articles")

print("✅ Étape 4 : Schéma Pydantic défini")

# -----------------------------------------------------------------------------
# ÉTAPE 5 : Configuration du LLM local (Ollama) - FALLBACK UNIQUEMENT
# -----------------------------------------------------------------------------
llm_config = LLMConfig(
    provider="ollama/qwen3:4b",
    api_token="ollama",
    base_url="http://localhost:11434"
)
print("✅ Étape 5 : LLMConfig créé (fallback)")

# -----------------------------------------------------------------------------
# ÉTAPE 6 : Configurations du crawler
# -----------------------------------------------------------------------------

# --- Browser : mode texte pour performance maximale ---
browser_config = BrowserConfig(
    headless=True,
    text_mode=True,       # Ne charge PAS images/CSS/fonts → beaucoup plus rapide
    verbose=False
)

# --- Markdown Generator avec PruningContentFilter ---
# Élimine le bruit AVANT que le contenu n'arrive au LLM
md_generator = DefaultMarkdownGenerator(
    content_filter=PruningContentFilter(
        threshold=0.4,          # Filtre modéré (0=tout garder, 1=tout supprimer)
        threshold_type="fixed",
        min_word_threshold=3    # Ignore les blocs de moins de 3 mots
    )
)

# --- Config de base pour le crawl (partagée) ---
def make_css_config(schema):
    """Crée une config pour l'extraction CSS (rapide, sans LLM)"""
    return CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        word_count_threshold=3,
        excluded_tags=["script", "style", "nav", "footer", "header", "aside", "form"],
        exclude_external_links=True,
        exclude_external_images=True,
        exclude_social_media_links=True,
        remove_overlay_elements=True,
        page_timeout=45000,
        extraction_strategy=JsonCssExtractionStrategy(schema)
    )

def make_llm_config():
    """Crée une config pour l'extraction LLM (fallback, plus lent)"""
    llm_strategy = LLMExtractionStrategy(
        llm_config=llm_config,
        schema=ArticleList.model_json_schema(),
        extraction_type="schema",
        # Prompt optimisé : anglais + /no_think + très direct
        instruction="""/no_think
Extract all news article titles from this page.
Return ONLY real news/press article headlines.
Ignore: menus, ads, navigation links, buttons, footers, sidebars, author names, dates.
""",
        # --- Optimisations clés pour petit modèle local ---
        input_format="fit_markdown",    # Utilise le markdown filtré par PruningContentFilter
        chunk_token_threshold=1500,     # Petits chunks adaptés à qwen3:4b
        overlap_rate=0.05,              # Léger overlap pour ne rien perdre
        apply_chunking=True,            # Active le découpage intelligent
        extra_args={
            "temperature": 0.0,         # Déterministe
            "max_tokens": 2000          # Suffisant pour les titres
        }
    )
    return CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        word_count_threshold=3,
        excluded_tags=["script", "style", "nav", "footer", "header", "aside", "form"],
        exclude_external_links=True,
        exclude_external_images=True,
        exclude_social_media_links=True,
        remove_overlay_elements=True,
        page_timeout=45000,
        markdown_generator=md_generator,    # PruningContentFilter actif
        extraction_strategy=llm_strategy
    )

print("✅ Étape 6 : Configurations crawler prêtes (CSS + LLM fallback)")

# -----------------------------------------------------------------------------
# ÉTAPE 7 : Fonctions d'extraction
# -----------------------------------------------------------------------------

def clean_css_results(raw_data, site_name):
    """
    Nettoie les résultats CSS : déduplique, filtre les titres trop courts
    ou qui ressemblent à de la navigation.
    """
    titles = []
    seen = set()
    
    # Mots-clés de navigation/UI à ignorer
    noise_words = {
        "menu", "home", "about", "contact", "subscribe", "login", "sign up",
        "search", "more", "read more", "load more", "next", "previous",
        "cookie", "privacy", "terms", "follow us", "share", "tweet",
        "facebook", "twitter", "linkedin", "instagram", "youtube",
        "newsletter", "copyright", "all rights", "powered by",
        "skip to content", "back to top", "close", "accept",
    }
    
    for item in raw_data:
        title = item.get("title", "").strip()
        
        # Filtres de qualité
        if not title:
            continue
        if len(title) < 15:           # Trop court pour un vrai titre
            continue
        if len(title) > 300:          # Trop long, probablement pas un titre
            continue
        if title.lower() in seen:     # Doublon
            continue
        if any(noise in title.lower() for noise in noise_words):
            continue
        if title.count(" ") < 2:      # Moins de 3 mots → probablement pas un titre
            continue
        
        seen.add(title.lower())
        titles.append(title)
    
    return titles


def parse_llm_results(extracted_content):
    """Parse les résultats du LLM (même logique que l'original, améliorée)"""
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
            
            if title and len(title) > 10 and title.lower() not in seen:
                seen.add(title.lower())
                titles.append(title)
        
        return titles
    except json.JSONDecodeError as e:
        print(f"  ❌ Erreur JSON LLM : {e}")
        return []


async def extract_titles_from_site(crawler, site_info):
    """
    Extraction hybride : CSS d'abord, LLM en fallback.
    """
    url = site_info["url"]
    name = site_info["name"]
    
    print(f"\n{'='*60}")
    print(f"🌐 {name} : {url}")
    print(f"{'='*60}")
    
    # ========================================
    # PHASE 1 : Tentative CSS (rapide, gratuit)
    # ========================================
    print(f"  📋 Phase 1 : Extraction CSS...")
    try:
        css_config = make_css_config(site_info["schema"])
        result = await crawler.arun(url=url, config=css_config)
        
        if result.success and result.extracted_content:
            raw_data = json.loads(result.extracted_content)
            titles = clean_css_results(raw_data, name)
            
            if len(titles) >= 3:  # Seuil de confiance : au moins 3 titres
                print(f"  ✅ CSS réussi : {len(titles)} titres extraits (pas de LLM nécessaire)")
                return {"url": url, "name": name, "titles": titles, "method": "CSS", "success": True}
            else:
                print(f"  ⚠️  CSS : seulement {len(titles)} titres → passage au LLM")
        else:
            print(f"  ⚠️  CSS : crawl échoué ou pas de contenu → passage au LLM")
            
    except Exception as e:
        print(f"  ⚠️  CSS erreur : {e} → passage au LLM")
    
    # ========================================
    # PHASE 2 : Fallback LLM (plus lent, mais plus intelligent)
    # ========================================
    print(f"  🤖 Phase 2 : Extraction LLM (fallback)...")
    try:
        llm_run_config = make_llm_config()
        result = await crawler.arun(url=url, config=llm_run_config)
        
        if result.success and result.extracted_content:
            titles = parse_llm_results(result.extracted_content)
            
            if titles:
                print(f"  ✅ LLM réussi : {len(titles)} titres extraits")
                return {"url": url, "name": name, "titles": titles, "method": "LLM", "success": True}
            else:
                print(f"  ❌ LLM : aucun titre extrait du contenu")
                return {"url": url, "name": name, "titles": [], "method": "LLM", "success": False, "error": "No titles parsed"}
        else:
            error_msg = getattr(result, 'error_message', 'Unknown error')
            print(f"  ❌ LLM crawl échoué : {error_msg}")
            return {"url": url, "name": name, "titles": [], "method": "LLM", "success": False, "error": error_msg}
            
    except Exception as e:
        print(f"  ❌ LLM exception : {e}")
        return {"url": url, "name": name, "titles": [], "method": "LLM", "success": False, "error": str(e)}


# -----------------------------------------------------------------------------
# ÉTAPE 8 : Fonction principale
# -----------------------------------------------------------------------------
async def main():
    """Parcourt tous les sites avec l'approche hybride CSS → LLM"""
    
    all_results = []
    start_time = time.time()
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        for i, site in enumerate(WEBSITES, 1):
            print(f"\n📍 Site {i}/{len(WEBSITES)}")
            
            result = await extract_titles_from_site(crawler, site)
            all_results.append(result)
            
            # Pause entre les sites
            if i < len(WEBSITES):
                print("  ⏳ Pause de 2 secondes...")
                await asyncio.sleep(2)
    
    elapsed = time.time() - start_time
    return all_results, elapsed


# -----------------------------------------------------------------------------
# ÉTAPE 9 : Exécution et affichage
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 EXTRACTION MULTI-SITES - CLIMAT/ÉNERGIE (OPTIMISÉ)")
    print("   Stratégie : CSS d'abord → LLM en fallback")
    print("=" * 60)
    
    results, elapsed = asyncio.run(main())
    
    # Résumé final
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ FINAL")
    print("=" * 60)
    
    total_titles = 0
    css_count = 0
    llm_count = 0
    
    for result in results:
        name = result["name"]
        method = result.get("method", "?")
        
        if result["success"]:
            count = len(result["titles"])
            total_titles += count
            
            if method == "CSS":
                css_count += 1
                emoji = "⚡"  # Rapide
            else:
                llm_count += 1
                emoji = "🤖"  # LLM
            
            print(f"\n{emoji} {name} [{method}] : {count} titres")
            print("-" * 40)
            for i, title in enumerate(result["titles"], 1):
                print(f"   {i:2}. {title}")
        else:
            print(f"\n❌ {name} : ÉCHEC - {result.get('error', 'Unknown')}")
    
    print("\n" + "=" * 60)
    print(f"📈 TOTAL : {total_titles} titres extraits de {len(WEBSITES)} sites")
    print(f"⚡ Méthode CSS : {css_count} sites (rapide, sans LLM)")
    print(f"🤖 Méthode LLM : {llm_count} sites (fallback)")
    print(f"⏱️  Temps total : {elapsed:.1f} secondes")
    print("=" * 60)
    
    # Sauvegarder
    output_file = "titres_extraits.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("TITRES D'ARTICLES - ACTUALITÉS CLIMAT/ÉNERGIE\n")
        f.write(f"Méthode : Hybride CSS + LLM fallback\n")
        f.write("=" * 60 + "\n\n")
        
        for result in results:
            name = result["name"]
            url = result["url"]
            method = result.get("method", "?")
            
            f.write(f"\n{'='*60}\n")
            f.write(f"SOURCE : {name} [{method}]\n")
            f.write(f"URL : {url}\n")
            f.write(f"{'='*60}\n\n")
            
            if result["success"] and result["titles"]:
                for i, title in enumerate(result["titles"], 1):
                    f.write(f"{i}. {title}\n")
            else:
                f.write("Aucun titre extrait\n")
        
        f.write(f"\n\nTOTAL : {total_titles} titres\n")
        f.write(f"Temps : {elapsed:.1f}s\n")
    
    print(f"\n💾 Résultats sauvegardés dans : {output_file}")
    
    # Sauvegarder aussi en JSON pour usage programmatique
    output_json = "titres_extraits.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Résultats JSON dans : {output_json}")
    print("✅ TERMINÉ")
