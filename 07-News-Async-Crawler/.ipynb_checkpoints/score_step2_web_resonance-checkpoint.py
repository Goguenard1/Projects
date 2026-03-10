# =============================================================================
# STEP 2 : WEB SEARCH RESONANCE — How much is each article talked about?
# Input  : scores_llm.csv (from Step 1)
# Output : scores_final.csv (with web_resonance + final_score filled)
# =============================================================================

import os
import csv
import time
import json
import re
from litellm import completion

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
GROQ_API_KEY = "YOUR_GROQ_API_KEY_HERE"  # <-- Replace with your key
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

INPUT_FILE = "scores_llm.csv"
OUTPUT_FILE = "scores_final.csv"

MODEL = "groq/llama-3.1-8b-instant"
PAUSE_BETWEEN_CALLS = 6  # 6 seconds = max 10 RPM, safe margin

# Major outlets we check against
MAJOR_OUTLETS = [
    "Reuters", "Bloomberg", "BBC", "CNN", "Al Jazeera",
    "Le Monde", "The Guardian", "Financial Times", "AFP",
    "Jeune Afrique", "RFI", "France 24",
    "New York Times", "Washington Post",
    "Africa Report", "African Business",
    "CNBC", "The Economist",
    "UN News", "World Bank", "IEA", "IRENA"
]

# -----------------------------------------------------------------------------
# STEP 1 : Read the CSV from Step 1
# -----------------------------------------------------------------------------
def read_scores_csv(filepath):
    """Read the scores_llm.csv file"""
    articles = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            articles.append(row)
    return articles

# -----------------------------------------------------------------------------
# STEP 2 : Ask LLM to estimate web resonance
# -----------------------------------------------------------------------------
RESONANCE_PROMPT = """You are a media analyst. Estimate how widely this news story has been covered by major international media.

TITLE: "{title}"
SOURCE: {source}

SCORING:
25 = 5+ major outlets covered this (COP summits, UN events, disasters)
20 = 3-4 major outlets (World Bank announcements, major deals)
15 = 1-2 major outlets + trade media (new regulations, reports)
10 = mainly trade/specialized media only
5 = only the original source

Respond with ONLY this JSON on a single line, nothing else:
{{"coverage": "low", "outlets": "Reuters and BBC", "score": 10}}
"""

def estimate_web_resonance(title, source):
    """Ask the LLM to estimate how widely covered this story is"""
    
    prompt = RESONANCE_PROMPT.format(title=title, source=source)
    
    try:
        response = completion(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=150
        )
        
        content = response.choices[0].message.content.strip()
        
        # Remove markdown code fences if present
        content = content.replace("```json", "").replace("```", "").strip()
        
        # Try to parse the full response as JSON first
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # Fallback: try to extract score with regex
            score_match = re.search(r'"score"\s*:\s*(\d+)', content)
            coverage_match = re.search(r'"coverage"\s*:\s*"(\w+)"', content)
            outlets_match = re.search(r'"outlets"\s*:\s*"([^"]*)"', content)
            
            if score_match:
                data = {
                    "score": int(score_match.group(1)),
                    "coverage": coverage_match.group(1) if coverage_match else "unknown",
                    "outlets": outlets_match.group(1) if outlets_match else ""
                }
            else:
                print(f"  ⚠️  Could not parse response: {content[:80]}")
                return {
                    "web_resonance": 0,
                    "coverage": "parse_error",
                    "likely_outlets": "",
                    "reasoning": content[:80],
                    "error": "JSON parse failed"
                }
        
        return {
            "web_resonance": int(data.get("score", 0)),
            "coverage": data.get("coverage", "unknown"),
            "likely_outlets": data.get("outlets", ""),
            "reasoning": "",
            "error": None
        }
            
    except Exception as e:
        error_msg = str(e)[:100]
        
        # Handle rate limit — wait and retry once
        if "rate_limit" in error_msg.lower() or "429" in error_msg:
            print(f"  ⏳ Rate limit hit, waiting 30 seconds...")
            time.sleep(30)
            try:
                response = completion(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=150
                )
                content = response.choices[0].message.content.strip()
                content = content.replace("```json", "").replace("```", "").strip()
                data = json.loads(content)
                return {
                    "web_resonance": int(data.get("score", 0)),
                    "coverage": data.get("coverage", "unknown"),
                    "likely_outlets": data.get("outlets", ""),
                    "reasoning": "",
                    "error": None
                }
            except Exception as e2:
                error_msg = str(e2)[:100]
        
        print(f"  ❌ API error: {error_msg}")
        return {
            "web_resonance": 0,
            "coverage": "error",
            "likely_outlets": "",
            "reasoning": error_msg,
            "error": error_msg
        }

# -----------------------------------------------------------------------------
# STEP 3 : Calculate final score
# -----------------------------------------------------------------------------
def calculate_final_score(llm_total, web_resonance):
    """
    Final score = LLM total (region + resonance estimate) + web resonance
    Max = 50 (region) + 25 (resonance estimate) + 25 (web resonance) = 100
    """
    try:
        return int(llm_total) + int(web_resonance)
    except (ValueError, TypeError):
        return 0

# -----------------------------------------------------------------------------
# STEP 4 : Main
# -----------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("🔍 STEP 2 : WEB SEARCH RESONANCE SCORING")
    print("=" * 60)
    
    # Read input
    print(f"\n📄 Reading {INPUT_FILE}...")
    articles = read_scores_csv(INPUT_FILE)
    print(f"✅ {len(articles)} articles loaded\n")
    
    if not articles:
        print("❌ No articles found. Check your input file.")
        return
    
    # Process each article
    results = []
    
    for i, article in enumerate(articles, 1):
        title = article["title"]
        source = article["source"]
        llm_total = article.get("llm_total", 0)
        
        print(f"[{i:3}/{len(articles)}] {title[:65]}...")
        
        # Get web resonance estimate
        resonance = estimate_web_resonance(title, source)
        
        # Calculate final score
        web_score = resonance["web_resonance"]
        final = calculate_final_score(llm_total, web_score)
        
        # Update article with new data
        article["web_resonance"] = web_score
        article["final_score"] = final
        
        if resonance["error"]:
            print(f"         ⚠️  Error: {resonance['error']}")
        else:
            outlets_str = resonance["likely_outlets"] if resonance["likely_outlets"] else "none"
            print(f"         → Web Resonance: {web_score}/25 ({resonance['coverage']})")
            print(f"         → Likely outlets: {outlets_str}")
            print(f"         → Final Score: {final}/100")
        
        results.append(article)
        
        # Pause between calls
        if i < len(articles):
            time.sleep(PAUSE_BETWEEN_CALLS)
    
    # Save to CSV
    print(f"\n💾 Saving results to {OUTPUT_FILE}...")
    
    fieldnames = [
        "title", "source",
        "region_score", "region_label",
        "resonance_score", "resonance_label",
        "llm_total",
        "web_resonance", "final_score", "kept"
    ]
    
    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 FINAL SCORING SUMMARY")
    print("=" * 60)
    
    scored = [r for r in results if int(r.get("final_score", 0)) > 0]
    if scored:
        avg_final = sum(int(r["final_score"]) for r in scored) / len(scored)
        
        # Articles above threshold
        above_60 = [r for r in scored if int(r["final_score"]) >= 60]
        above_50 = [r for r in scored if int(r["final_score"]) >= 50]
        
        print(f"✅ {len(scored)}/{len(results)} articles scored")
        print(f"📈 Average Final Score: {avg_final:.1f}/100")
        print(f"🎯 Articles scoring ≥60: {len(above_60)}")
        print(f"🎯 Articles scoring ≥50: {len(above_50)}")
        
        # Top 15
        top_15 = sorted(scored, key=lambda x: int(x["final_score"]), reverse=True)[:15]
        print(f"\n🏆 TOP 15 ARTICLES FOR NEWSLETTER:")
        print("-" * 60)
        for i, r in enumerate(top_15, 1):
            print(f"  {i:2}. [{r['final_score']:>3}/100] {r['title'][:55]}")
            print(f"      Region: {r['region_score']}/50 | LLM Res: {r['resonance_score']}/25 | Web: {r['web_resonance']}/25")
            print(f"      Source: {r['source']}")
        
        # Newsletter candidates (≥60)
        if above_60:
            print(f"\n📰 NEWSLETTER CANDIDATES (score ≥ 60):")
            print("-" * 60)
            for r in sorted(above_60, key=lambda x: int(x["final_score"]), reverse=True):
                print(f"  [{r['final_score']:>3}] {r['title'][:70]}")
    
    print(f"\n✅ Done! Final scores saved to {OUTPUT_FILE}")
    print("📝 Next: Open the CSV, fill the 'kept' column (Yes/No) after your newsletter")
    print("         This builds your training data for future LDA classification")


if __name__ == "__main__":
    main()
