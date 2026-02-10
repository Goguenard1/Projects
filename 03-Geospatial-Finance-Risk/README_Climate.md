# Climate Risk Assessment for Financial Portfolios 🌍

A comprehensive geospatial and quantitative analysis project evaluating climate risk exposure of a financial asset portfolio in metropolitan France, integrating IPCC climate scenarios, geospatial data processing, and sector-specific vulnerability scoring.

## 📊 Project Overview

This project was developed as part of a **Data Challenge** in collaboration with **Caisse des Dépôts et Consignations (CDC)**, France's public financial institution, to assess and quantify climate risk exposure of financial assets across France.

**Challenge Objective**: Develop a robust methodology to measure asset portfolio exposure to climate hazards by integrating geospatial data, climate projections, and financial characteristics.

**Academic Context**: Master 2 IREF - Economic Risks and Data Science  
**Authors**: Anas Baji, Enora Friant, Mohamed Rekik  
**Partner**: Caisse des Dépôts et Consignations

## 🎯 Key Objectives

1. **Quantify climate risk exposure** for a portfolio of financial assets in metropolitan France
2. **Integrate multi-temporal climate projections** using IPCC scenarios across three time horizons
3. **Develop a vulnerability scoring system** accounting for:
   - Asset location (municipality-level)
   - Business sector activity
   - Loan maturity/investment horizon
   - Climate hazard intensity
4. **Leverage geospatial analysis** (QGIS) to map and assess risk exposure

## 🌡️ Climate Hazards Analyzed

The project evaluates **four major climate hazards** affecting French municipalities:

| Hazard | Icon | Indicator | Data Source |
|--------|------|-----------|-------------|
| **Drought** | 🌞 | NORCWBC | DRIAS Climate Portal |
| **Floods** | 🌊 | NORPAV | DRIAS + Georisques |
| **Wildfires** | 🔥 | NORIFMxAV | DRIAS |
| **Heat Waves** | 🌡️ | NORTXHWD | DRIAS Climate Portal |

## 📅 Temporal Framework - IPCC Climate Scenarios

The analysis incorporates **three temporal horizons** aligned with IPCC climate projections:

- **Horizon 1 (H1)**: 2021-2050 - Near-term climate impacts
- **Horizon 2 (H2)**: 2051-2070 - Mid-century projections  
- **Horizon 3 (H3)**: 2071-2100 - Long-term climate scenarios

Assets are matched to appropriate horizons based on **loan maturity dates**, ensuring forward-looking risk assessment.

## 🗺️ Data Sources & Integration

### Primary Data Sources

1. **DRIAS Climate Portal** ([drias-climat.fr](http://www.drias-climat.fr/))
   - Climate hazard indicators (NOR indices)
   - Multi-scenario projections (RCP/SSP scenarios)
   - Municipality-level climate data

2. **Georisques** ([georisques.gouv.fr](https://www.georisques.gouv.fr/))
   - Flood-prone municipality database
   - Historical natural disaster records

3. **Financial Portfolio Data**
   - Asset location (INSEE municipality codes)
   - Loan amount (exposure in million EUR)
   - Business sector classification
   - Loan maturity dates

### Geospatial Data Processing

- **Tool**: QGIS (Quantum GIS)
- **Process**: 
  - Import and layer climate hazard shapefiles
  - Join climate indicators to municipality boundaries
  - Extract risk metrics per INSEE code
  - Export processed data for quantitative analysis

## 🛠️ Methodology

### Phase 1: Data Collection & Preparation

1. **Geospatial Data Extraction (QGIS)**
   - Import DRIAS climate layers for all hazards
   - Process three temporal horizons (H1, H2, H3)
   - Join hazard indicators to French municipality geometries
   - Export municipality-level climate risk metrics

2. **Portfolio-Climate Data Integration**
   - Merge financial portfolio with climate indicators via INSEE codes
   - Match assets to appropriate temporal horizon based on loan maturity
   - Validate data consistency and completeness

### Phase 2: Risk Scoring Framework

#### 1. **Climate Indicator Normalization**
Normalize DRIAS indicators (NORCWBC, NORIFMxAV, NORPAV, NORTXHWD) to [0,1] scale using MinMax transformation:

```
NOR_normalized = (NOR - NOR_min) / (NOR_max - NOR_min)
```

#### 2. **Sector-Specific Vulnerability Weighting**

Each economic sector receives differential risk scores (0, 0.5, 1) based on hazard sensitivity:

| Sector | Drought | Wildfires | Floods | Heat Waves |
|--------|---------|-----------|--------|------------|
| **Construction** | 1.0 | 0.5 | 0.5 | 0.5 |
| **Residential/Care Homes** | 0.5 | 0.0 | 1.0 | 1.0 |
| **Agriculture** | 1.0 | 1.0 | 1.0 | 0.5 |
| **Industry** | 0.5 | 0.5 | 0.5 | 1.0 |
| **Services** | 0.0 | 0.0 | 0.5 | 0.5 |

*Scores based on sector-specific climate vulnerability literature and expert assessment*

#### 3. **Financial Exposure Integration**

Normalize loan amounts to account for portfolio concentration:

```
Exposure_normalized = (Loan_Amount - Loan_Min) / (Loan_Max - Loan_Min)
```

#### 4. **Composite Vulnerability Score**

For each asset and each hazard:

```
Vulnerability_Score = NOR_normalized × Sector_Weight × Exposure_normalized
```

This produces **four distinct scores per asset** representing exposure to each climate hazard.

#### 5. **Geospatial Adjustments**

**Wildfire Risk Correction**: Assets in non-forested municipalities receive a wildfire score of 0, regardless of other factors.

**Flood Risk Enhancement**: Integration with Georisques database to flag high-risk flood zones.

### Phase 3: Stochastic Extreme Event Modeling

To capture the **tail risk** of extreme climate events, we implement a **jump process** for floods and wildfires:

```python
# Jump process for extreme events
jump_probability = 0.05  # 5% chance of extreme event
jump_multiplier = np.random.uniform(1.5, 3.0)  # 1.5x to 3x severity increase

if np.random.random() < jump_probability:
    risk_score = risk_score * jump_multiplier
```

This models the **non-linear, catastrophic nature** of certain climate events.

## 📈 Key Results & Deliverables

### Outputs

1. **Municipality-Level Climate Risk Database**
   - 36,000+ French municipalities
   - Four climate hazards per municipality
   - Three temporal horizons

2. **Portfolio Vulnerability Scores**
   - Individual asset-level scores for each hazard
   - Aggregated portfolio exposure metrics
   - Sector-specific risk concentrations

3. **Risk Visualizations**
   - Heatmaps of high-risk municipalities
   - Portfolio concentration by hazard type
   - Temporal evolution of risk (H1 → H3)
   - Before/after comparison of risk adjustments

4. **Decision Support Metrics**
   - Assets ranked by total climate vulnerability
   - High-risk asset identification (top 10%)
   - Sector diversification recommendations

### Key Insights

- **Sectoral Heterogeneity**: Agriculture and construction show highest drought sensitivity
- **Geographic Concentration**: Southern France exhibits elevated wildfire and heat wave risks
- **Temporal Escalation**: Risk scores increase 15-40% from H1 to H3 across most hazards
- **Portfolio Implications**: X% of assets in high-risk zones require enhanced monitoring

## 🛠️ Technologies & Tools

### Geospatial Analysis
```
QGIS                    # Spatial data processing & visualization
GeoPandas               # Python geospatial operations
Shapely                 # Geometric operations
```

### Data Analysis & Modeling
```python
pandas, numpy           # Data manipulation
scikit-learn            # Normalization, scaling
scipy                   # Statistical analysis
```

### Visualization
```python
matplotlib, seaborn     # Statistical plots
plotly                  # Interactive visualizations
```

### Climate Data Sources
```
DRIAS                   # French climate projections
Georisques              # Natural hazard database
INSEE                   # Geographic identifiers
```

## 📊 Project Structure

### Section I: Data Collection & Preparation
1. QGIS geospatial data extraction
2. Climate indicator processing (DRIAS)
3. Portfolio-climate data integration
4. Temporal horizon matching

### Section II: Risk Scoring Framework
1. Climate indicator integration (NORIFMxAV, NORCWBC, NORPAV, NORTXHWD)
2. Sector vulnerability scoring
3. Financial exposure normalization
4. Composite vulnerability calculation
5. Geospatial adjustments (forest coverage, flood zones)
6. Stochastic extreme event modeling

### Section III: Results & Visualization
1. Portfolio risk distribution analysis
2. High-risk asset identification
3. Temporal risk evolution
4. Sector-specific insights
5. Interactive risk maps

## 🔑 Key Methodological Innovations

1. **Multi-Horizon Climate Integration**: Matching asset maturity to appropriate IPCC scenarios
2. **Sector-Differentiated Scoring**: Evidence-based vulnerability weights by economic activity
3. **Geospatial Correction**: Municipality-specific adjustments (forest cover, topography)
4. **Stochastic Tail Risk**: Jump process modeling for extreme events
5. **Financial-Climate Linkage**: Explicit connection between loan exposure and climate risk

## 📚 Climate Science Background

### DRIAS NOR Indicators

**NOR (Nombre d'Occurrences de Référence)** indices represent the frequency or intensity of climate events relative to a reference period (1976-2005):

- **NORCWBC**: Consecutive days without rain (drought indicator)
- **NORIFMxAV**: Fire weather index combining temperature, humidity, wind
- **NORPAV**: Precipitation anomaly (flood indicator)
- **NORTXHWD**: Heat wave duration and intensity

### IPCC Scenarios

The project uses **Representative Concentration Pathways (RCP)** or **Shared Socioeconomic Pathways (SSP)** scenarios reflecting different greenhouse gas emission trajectories.

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn geopandas shapely
```

### Data Requirements
1. **Financial Portfolio**: Excel/CSV with columns:
   - `Localisation (Code INSEE)` - Municipality code
   - `Encours (million EUR)` - Loan amount
   - `Secteur d'activité` - Business sector
   - `Maturité du prêt` - Loan maturity year

2. **Climate Data** (extracted via QGIS):
   - `feux-de-forets-H1.csv`, `H2.csv`, `H3.csv` - Wildfire indicators
   - `secheresse-H1.csv`, `H2.csv`, `H3.csv` - Drought indicators
   - `inondations-H1.csv`, `H2.csv`, `H3.csv` - Flood indicators
   - `chaleur-H1.csv`, `H2.csv`, `H3.csv` - Heat wave indicators

3. **Auxiliary Data**:
   - Non-forested municipalities list
   - Georisques flood-prone zones

### Running the Analysis
```python
jupyter notebook DataChallengeF.ipynb
```

Execute cells sequentially through all sections.

## 📊 Use Cases & Applications

### For Financial Institutions
- **Portfolio stress testing** under climate scenarios
- **Risk-adjusted asset pricing** incorporating climate factors
- **Capital allocation** based on climate vulnerability
- **Regulatory compliance** (TCFD, SFDR climate risk disclosure)

### For Public Policy
- **Infrastructure investment** prioritization in high-risk zones
- **Climate adaptation** fund allocation
- **Regional development** planning under climate constraints

### For Asset Managers
- **ESG integration** with quantitative climate metrics
- **Sector rotation** strategies based on climate transition
- **Green bond** eligibility assessment

## 💡 Future Enhancements

1. **Transition Risk Integration**: Combine physical risk with policy/technology transition risks
2. **Dynamic Modeling**: Time-series forecasting of risk evolution
3. **Network Effects**: Model supply chain climate contagion
4. **Machine Learning**: Predictive models for climate-financial correlations
5. **Real-Time Updates**: API integration with DRIAS for continuous monitoring
6. **Interactive Dashboard**: Web-based tool for portfolio managers

## 📄 Academic Context

This project demonstrates the practical application of:
- **Climate science** (IPCC scenarios, hazard modeling)
- **Geospatial analysis** (GIS, spatial joins)
- **Financial risk management** (exposure quantification, scoring)
- **Data science** (normalization, stochastic modeling)
- **Sustainable finance** (climate risk integration)

## 👥 Contributors

- **Anas Baji** - Methodology Design, Data Processing, Scoring Framework
- **Enora Friant** - Geospatial Analysis, QGIS Processing, Visualization
- **Mohamed Rekik** - Climate Data Integration, Statistical Analysis

## 🏆 Partner Organization

**Caisse des Dépôts et Consignations (CDC)**
- France's public financial institution
- Leading investor in sustainable and territorial development
- Pioneer in climate risk integration for public finance

## 📚 References

- **DRIAS Portal**: [http://www.drias-climat.fr/](http://www.drias-climat.fr/)
- **Georisques**: [https://www.georisques.gouv.fr/](https://www.georisques.gouv.fr/)
- **IPCC Reports**: Climate Change Assessment Reports
- **TCFD Recommendations**: Task Force on Climate-related Financial Disclosures

## 📄 License

This project is part of academic coursework for M2 IREF program in collaboration with Caisse des Dépôts et Consignations.

---

**Note**: This project showcases the integration of climate science, geospatial analysis, and financial risk management for real-world sustainable finance applications. The methodology aligns with emerging regulatory frameworks (EU Taxonomy, SFDR) requiring climate risk quantification.

🌍 **Climate Action through Data Science**
