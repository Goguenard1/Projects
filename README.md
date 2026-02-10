# Flight Delay Prediction using PySpark & Machine Learning

A comprehensive data science project analyzing US domestic flight data from 1994 to predict flight delays using multiple machine learning algorithms and big data processing with PySpark.

## 📊 Project Overview

This project performs exploratory data analysis and builds predictive models to classify flight delays using historical airline data. A flight is considered delayed if its arrival delay exceeds 15 minutes. The project demonstrates end-to-end machine learning pipeline development, from data preprocessing to model evaluation.

**Academic Context**: M2 IREF - Economic Risks and Data Science  
**Authors**: Juliette Mingot & Anas Baji

## 🎯 Objectives

- Analyze patterns in flight delays across different temporal and spatial dimensions
- Engineer features from flight schedules, routes, and carrier information
- Build and compare multiple classification models for delay prediction
- Evaluate model performance using industry-standard metrics

## 📁 Dataset

- **Source**: 1994 US domestic flight records
- **Key Features**:
  - Temporal: Month, Day of Month, Day of Week, Scheduled Departure/Arrival Times
  - Spatial: Origin and Destination Airports
  - Flight Details: Distance, Carrier (Airline)
  - Target: Arrival Delay (binarized to 0/1 based on 15-minute threshold)

## 🛠️ Technologies & Tools

- **Big Data Processing**: Apache Spark (PySpark)
- **Data Analysis**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Machine Learning**: 
  - PySpark MLlib (Logistic Regression, Random Forest, Gradient Boosting, Multilayer Perceptron)
  - Scikit-learn (for additional preprocessing and metrics)

## 🔍 Key Analysis Components

### Exploratory Data Analysis
- Temporal delay patterns by hour, day, and month
- Airport-specific delay statistics
- Carrier performance analysis
- Distance vs. delay correlations

### Feature Engineering
- One-hot encoding of categorical variables (airports, carriers)
- Time-based feature extraction
- Feature vectorization for model training

### Machine Learning Models

| Model | Type | Key Parameters |
|-------|------|----------------|
| **Logistic Regression** | Linear baseline | - |
| **Random Forest** | Ensemble | 10 trees, max depth 5 |
| **Multilayer Perceptron** | Neural Network | Custom architecture |
| **Gradient Boosting Trees** | Ensemble | Max bins: 250 |

### Model Evaluation
- **Metrics**: Accuracy, AUC-ROC
- **Visualization**: ROC curves for each model
- **Comparison**: Performance benchmarking across algorithms

## 📈 Results

Models are evaluated using:
- Binary classification accuracy
- Area Under ROC Curve (AUC)
- ROC curve visualization for model comparison

The Gradient Boosting classifier shows strong performance with comprehensive feature utilization and handles the imbalanced nature of flight delay data effectively.

## 🚀 Getting Started

### Prerequisites
```bash
pip install pyspark pandas numpy matplotlib seaborn scikit-learn
```

### Running the Notebook
1. Ensure you have the `1994.csv.bz2` dataset in your working directory
2. Open `PF_Mingot_Baji.ipynb` in Jupyter Notebook or JupyterLab
3. Run cells sequentially to reproduce the analysis

### Data Requirements
- The notebook expects compressed flight data in `.csv.bz2` format
- Minimum required columns: Month, DayofMonth, DayOfWeek, CRSDepTime, CRSArrTime, ArrDelay, Distance, Origin, Dest, UniqueCarrier

## 📊 Project Structure

The notebook is organized into 21 questions covering:
1. Data loading and initial exploration
2. Feature selection and renaming
3. Data cleaning and preprocessing
4. Exploratory visualizations
5. Missing value handling
6. Feature encoding for categorical variables
7. Model training and hyperparameter configuration
8. Model evaluation and comparison
9. Results visualization and interpretation

## 🔑 Key Insights

- Flight delays exhibit clear temporal patterns throughout the day
- Certain airports and routes are more prone to delays
- Ensemble methods (Random Forest, Gradient Boosting) outperform simpler linear models
- Feature engineering from time and location data significantly improves prediction accuracy

## 📝 Future Improvements

- Incorporate weather data for enhanced predictions
- Implement cross-validation for robust model selection
- Explore deep learning architectures (LSTM for sequential patterns)
- Add hyperparameter tuning with grid search or Bayesian optimization
- Deploy model as a real-time prediction service

## 👥 Contributors

- **Juliette Mingot** - Data Analysis & Modeling
- **Anas Baji** - Data Analysis & Modeling

## 📄 License

This project is part of academic coursework for M2 IREF program.

---

**Note**: This project uses historical data for educational purposes and demonstrates practical application of big data processing and machine learning techniques in the aviation industry.
