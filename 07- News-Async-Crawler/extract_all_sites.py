# =============================================================================
# EXTRACTION DES TITRES - MULTI-SITES CLIMAT/ÉNERGIE
# 6 sites : Bloomberg Green, Carbon Pulse, Climate Change News, 
#           PV Tech, SEforALL, Africa Energy
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
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, BrowserConfig, CacheMode, LLMConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from pydantic import BaseModel, Field
from typing import List
import json
import time

print("✅ Étape 2 : Imports réussis")

# -----------------------------------------------------------------------------
# ÉTAPE 3 : Liste des sites à scraper
# -----------------------------------------------------------------------------
WEBSITES = [
    "https://www.bloomberg.com/green",
    "https://carbon-pulse.com/",
    "https://www.climatechangenews.com/",
    "https://www.pv-tech.org/",
    "https://www.seforall.org/news-and-events/news",
    "https://www.africa-energy.com/news-centre"
]

print(f"✅ Étape 3 : {len(WEBSITES)} sites à scraper")

# -----------------------------------------------------------------------------
# ÉTAPE 4 : Définir le schéma Pydantic
# -----------------------------------------------------------------------------
class Article(BaseModel):
    title: str = Field(..., description="Le titre de l'article")

class ArticleList(BaseModel):
    articles: List[Article] = Field(..., description="Liste des articles")

print("✅ Étape 4 : Schéma Pydantic défini")

# -----------------------------------------------------------------------------
# ÉTAPE 5 : Configurer Groq
# -----------------------------------------------------------------------------
GROQ_API_KEY = "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"

llm_config = LLMConfig(
    provider="groq/meta-llama/llama-guard-4-12b",
    api_token=GROQ_API_KEY
)

print("✅ Étape 5 : LLMConfig GROQ créé")

# -----------------------------------------------------------------------------
# ÉTAPE 6 : Configurer l'extraction et le crawler
# -----------------------------------------------------------------------------
extraction_strategy = LLMExtractionStrategy(
    llm_config=llm_config,
    schema=ArticleList.model_json_schema(),
    extraction_type="schema",
    instruction="""
    Extrais tous les titres d'articles d'actualité de cette page.
    
    Retourne uniquement les vrais titres d'articles de presse/news.
    Ignore : menus, publicités, liens de navigation, boutons, footers.
    """,
    extra_args={
        "temperature": 0.0,
        "max_tokens": 3000
    }
)

browser_config = BrowserConfig(
    headless=True,
    verbose=False
)

crawler_config = CrawlerRunConfig(
    cache_mode=CacheMode.BYPASS,
    word_count_threshold=5,
    page_timeout=60000,
    extraction_strategy=extraction_strategy
)

print("✅ Étape 6 : Crawler et extraction configurés")

# -----------------------------------------------------------------------------
# ÉTAPE 7 : Fonction pour extraire les titres d'un site
# -----------------------------------------------------------------------------
async def extract_titles_from_site(crawler, url):
    """Extrait les titres d'articles d'un site"""
    
    print(f"\n{'='*60}")
    print(f"🌐 Crawling : {url}")
    print(f"{'='*60}")
    
    try:
        result = await crawler.arun(url=url, config=crawler_config)
        
        if result.success:
            print("✅ Crawl réussi !")
            
            if result.extracted_content:
                try:
                    extracted = json.loads(result.extracted_content)
                    
                    # Récupérer les articles
                    if isinstance(extracted, list):
                        if len(extracted) > 0 and 'articles' in extracted[0]:
                            articles = extracted[0]['articles']
                        else:
                            articles = extracted
                    elif isinstance(extracted, dict):
                        articles = extracted.get('articles', [])
                    else:
                        articles = []
                    
                    # Extraire les titres
                    titles = []
                    for article in articles:
                        if isinstance(article, dict):
                            title = article.get('title', '')
                        else:
                            title = str(article)
                        if title:
                            titles.append(title)
                    
                    print(f"🎯 {len(titles)} titres extraits")
                    return {"url": url, "titles": titles, "success": True}
                    
                except json.JSONDecodeError as e:
                    print(f"❌ Erreur JSON : {e}")
                    return {"url": url, "titles": [], "success": False, "error": str(e)}
            else:
                print("❌ Pas de contenu extrait")
                return {"url": url, "titles": [], "success": False, "error": "No content"}
        else:
            print(f"❌ Crawl échoué : {result.error_message}")
            return {"url": url, "titles": [], "success": False, "error": result.error_message}
            
    except Exception as e:
        print(f"❌ Exception : {e}")
        return {"url": url, "titles": [], "success": False, "error": str(e)}

# -----------------------------------------------------------------------------
# ÉTAPE 8 : Fonction principale
# -----------------------------------------------------------------------------
async def main():
    """Fonction principale qui parcourt tous les sites"""
    
    all_results = []
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        for i, url in enumerate(WEBSITES, 1):
            print(f"\n📍 Site {i}/{len(WEBSITES)}")
            
            result = await extract_titles_from_site(crawler, url)
            all_results.append(result)
            
            # Pause entre les sites pour éviter les rate limits
            if i < len(WEBSITES):
                print("⏳ Pause de 60 secondes...")
                await asyncio.sleep(60)
    
    return all_results

# -----------------------------------------------------------------------------
# ÉTAPE 9 : Exécution et affichage des résultats
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 EXTRACTION MULTI-SITES - ACTUALITÉS CLIMAT/ÉNERGIE")
    print("=" * 60)
    
    # Exécuter
    results = asyncio.run(main())
    
    # Résumé final
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ FINAL")
    print("=" * 60)
    
    total_titles = 0
    
    for result in results:
        url = result["url"]
        # Extraire juste le nom du site
        site_name = url.replace("https://", "").replace("www.", "").split("/")[0]
        
        if result["success"]:
            count = len(result["titles"])
            total_titles += count
            print(f"\n✅ {site_name} : {count} titres")
            print("-" * 40)
            for i, title in enumerate(result["titles"], 1):
                print(f"   {i:2}. {title}")
        else:
            print(f"\n❌ {site_name} : ÉCHEC - {result.get('error', 'Unknown')}")
    
    print("\n" + "=" * 60)
    print(f"📈 TOTAL : {total_titles} titres extraits de {len(WEBSITES)} sites")
    print("=" * 60)
    
    # Sauvegarder dans un fichier
    output_file = "titres_extraits.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("TITRES D'ARTICLES - ACTUALITÉS CLIMAT/ÉNERGIE\n")
        f.write("=" * 60 + "\n\n")
        
        for result in results:
            url = result["url"]
            site_name = url.replace("https://", "").replace("www.", "").split("/")[0]
            
            f.write(f"\n{'='*60}\n")
            f.write(f"SOURCE : {site_name}\n")
            f.write(f"URL : {url}\n")
            f.write(f"{'='*60}\n\n")
            
            if result["success"] and result["titles"]:
                for i, title in enumerate(result["titles"], 1):
                    f.write(f"{i}. {title}\n")
            else:
                f.write("Aucun titre extrait\n")
        
        f.write(f"\n\nTOTAL : {total_titles} titres\n")
    
    print(f"\n💾 Résultats sauvegardés dans : {output_file}")
    print("✅ TERMINÉ")
