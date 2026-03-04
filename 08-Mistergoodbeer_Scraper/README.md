# 🍺 MisterGoodBeer Paris — Web Scraper

Scraping automatisé du site [MisterGoodBeer](https://www.mistergoodbeer.com) pour extraire les données des bars pas chers de Paris : nom, adresse, prix et horaires de Happy Hour, pour les 20 arrondissements.

## Objectif

Constituer une base de données complète des bars référencés sur MisterGoodBeer Paris, en combinant :

1. **Les données de listing** — nom, adresse, prix depuis les pages de classement par arrondissement
2. **Les données détaillées** — horaires de Happy Hour extraits depuis la page individuelle de chaque bar
3. **La fusion** — jointure des deux datasets sur le nom du bar pour obtenir un fichier unique enrichi

## Architecture du pipeline

```
Pages listing (×20 arrondissements)          Pages individuelles (×N bars)
┌─────────────────────────────┐              ┌──────────────────────────┐
│  /top-bars-pas-chers-paris  │              │  /bars/{nom-du-bar}      │
│  - Nom                      │   ──────▶    │  - Happy Hour            │
│  - Adresse                  │  ouverture   │  - Détails               │
│  - Prix                     │  nouvel      └──────────────────────────┘
└─────────────────────────────┘  onglet
              │                                        │
              ▼                                        ▼
        bars.xlsx                            happy_hours_paris.csv/xlsx
              │                                        │
              └──────────────┬─────────────────────────┘
                             ▼
                    pd.merge(on='Nom')
                             │
                             ▼
                    bars_complet.xlsx
```

## Fonctionnalités

- **Scraping multi-arrondissements** — boucle automatique sur les 20 arrondissements de Paris
- **Navigation dynamique** — ouverture de chaque fiche bar dans un nouvel onglet pour extraire les Happy Hours
- **Scroll dynamique** — gestion du lazy loading avec scroll progressif
- **Gestion d'erreurs** — try/except sur chaque bar pour ne pas interrompre le pipeline en cas de données manquantes
- **Fusion de datasets** — jointure Pandas des données de listing et des données détaillées
- **Export multi-format** — sortie en Excel (.xlsx) et CSV

## Stack technique

| Technologie | Usage |
|-------------|-------|
| **Python 3** | Langage principal |
| **Selenium** | Automatisation navigateur (Firefox/Geckodriver) |
| **Pandas** | Manipulation et fusion des données |
| **WebDriverWait** | Attentes explicites pour le contenu dynamique |
| **openpyxl** | Export Excel |

## Installation

```bash
# Installer les dépendances
pip install selenium pandas openpyxl

# Geckodriver (Firefox) doit être installé
# macOS
brew install geckodriver

# Linux
sudo apt install firefox-geckodriver

# Windows — télécharger depuis https://github.com/mozilla/geckodriver/releases
```

## Utilisation

Ouvrir le notebook dans Jupyter :

```bash
jupyter notebook Mistergoodbeer_Scrap.ipynb
```

### Cellules principales

| Cellule | Description |
|---------|-------------|
| **Cell 10** | Scraping complet des 20 arrondissements (nom, adresse, prix) |
| **Cell 23** | Scraping des Happy Hours via navigation sur chaque fiche bar |
| **Cell 24** | Conversion CSV → Excel |
| **Cell 25** | Fusion des deux fichiers sur la colonne `Nom` |

## Structure des données

### bars.xlsx

| Colonne | Description |
|---------|-------------|
| Arrondissement | Ex : `1e`, `2e`, ... `20e` |
| Nom | Nom du bar |
| Adresse | Adresse complète à Paris |
| Prix | Prix de la pinte (à partir de X€) |

### happy_hours_paris.xlsx

| Colonne | Description |
|---------|-------------|
| Nom | Nom du bar |
| Happy Hour | Horaires du Happy Hour |

### bars_complet.xlsx (fusion)

| Colonne | Description |
|---------|-------------|
| Arrondissement | Arrondissement |
| Nom | Nom du bar |
| Adresse | Adresse |
| Prix | Prix de la pinte |
| Happy Hour | Horaires HH (ou "Non renseigné") |

## Difficultés rencontrées

Le notebook contient plusieurs itérations du scraper, reflétant une démarche itérative face aux défis techniques :

- **Sélecteurs CSS instables** — le site utilise des classes générées (ex: `H2Bars_name__AsCPG`) qui changent entre les déploiements → passage aux XPath absolus puis dynamiques
- **Contenu chargé dynamiquement** — certains bars n'apparaissent qu'après scroll → ajout de `scrollTo()` + `sleep()`
- **Extraction du prix** — format variable selon les bars → parsing avec `split("à partir de")` et `split("€")`
- **Navigation multi-onglets** — gestion du `switch_to.window` pour revenir à la page de listing après chaque extraction

## Fichiers

```
mistergoodbeer-scraper/
├── Mistergoodbeer_Scrap.ipynb   # Notebook principal
├── bars.xlsx                     # Données de listing
├── happy_hours_paris.csv         # Happy Hours (CSV)
├── happy_hours_paris.xlsx        # Happy Hours (Excel)
├── bars_complet.xlsx             # Dataset final fusionné
└── README.md
```

## License

MIT
