# Machine Learning Regression & Optimization Homework

A comprehensive machine learning project covering cross-validation techniques, genetic algorithm optimization, and real-world rental price prediction using multiple regression methods.

## 📊 Project Overview

This project is divided into three main components demonstrating fundamental and advanced machine learning concepts, from model selection via cross-validation to genetic algorithm optimization and practical price prediction modeling.

**Academic Context**: M2 IREF - Economic Risks and Data Science  
**Authors**: Juliette Mabileau & Anas Baji

## 🎯 Project Components

### 1. Cross-Validation & Model Selection
Implementation of k-fold cross-validation (k=10) to evaluate and compare regression models with optimal hyperparameter selection.

**Models Evaluated:**
- **Polynomial Regression**: Testing degrees 1-10 to find optimal model complexity
- **k-Nearest Neighbors (k-NN) Regression**: Comparing different values of k

**Objective**: Minimize validation MSE while avoiding overfitting through systematic model comparison.

### 2. Global Minimum Optimization with Genetic Algorithms
Implementation of a genetic algorithm to find the global minimum of the Six-Hump Camelback function, a classic optimization benchmark.

**Key Components:**
- Population-based stochastic optimization
- Natural selection mechanisms (selection, crossover, mutation)
- Fitness evaluation and convergence analysis
- Hyperparameter tuning (population size, mutation probability, generation count)

**Achievement**: Solution within 0.01 of the exact minimum using population size of 1000 and mutation probability of 0.1.

### 3. Airbnb London Rental Price Prediction
End-to-end machine learning pipeline for predicting rental prices using real-world Airbnb data from London.

#### Data Pipeline
- Data cleaning and missing value treatment
- Categorical variable encoding
- Statistical testing (location-based analysis)
- Correlation analysis and multicollinearity detection
- Dimensionality reduction techniques

#### Machine Learning Models

| Model | Regularization | Feature Selection | Purpose |
|-------|---------------|-------------------|---------|
| **Random Forest** | - | GridSearchCV | Ensemble baseline |
| **k-NN Regression** | Lasso | Cross-validation | Distance-based prediction |
| **Linear Regression (OLS)** | Ridge/Lasso | Genetic Algorithm | Interpretable baseline |
| **Ridge Regression** | L2 | Grid Search | Handle multicollinearity |
| **Lasso Regression** | L1 | LassoCV | Feature selection |

## 🛠️ Technologies & Libraries

### Core ML & Data Science
```python
- pandas, numpy           # Data manipulation
- scikit-learn            # Machine learning models
- statsmodels             # Statistical modeling (OLS)
- matplotlib, seaborn     # Visualization
```

### Specialized Libraries
```python
- DEAP                    # Genetic algorithm framework
- PolynomialFeatures      # Feature engineering
- GridSearchCV, KFold     # Hyperparameter tuning
```

## 📈 Key Methods & Techniques

### Cross-Validation Strategy
- **10-Fold Cross-Validation**: Robust model evaluation with shuffled data
- **MSE Calculation**: Mean Squared Error as primary performance metric
- **Degree Selection**: Systematic polynomial degree comparison (1-10)

### Regularization Techniques
- **Ridge Regression**: L2 regularization to handle multicollinearity
- **Lasso Regression**: L1 regularization for automatic feature selection
- **LassoCV**: Cross-validated Lasso for optimal alpha selection

### Feature Engineering & Selection
- One-hot encoding for categorical variables
- Statistical hypothesis testing on location features
- Genetic algorithm-based feature selection using BIC criterion
- Correlation analysis to identify redundant features

### Genetic Algorithm Optimization
**Implementation Details:**
- DEAP (Distributed Evolutionary Algorithms in Python) framework
- Fitness function based on BIC (Bayesian Information Criterion)
- Three genetic operators: selection, crossover, mutation
- Parameter tuning through systematic experimentation

## 📊 Model Evaluation & Results

### Performance Metrics
- **Mean Squared Error (MSE)**: Primary regression metric
- **R² Score**: Explained variance in cross-validation
- **BIC (Bayesian Information Criterion)**: Model complexity penalty

### Visualization & Analysis
- Density plots comparing predicted vs actual values
- MSE comparison across all models
- Feature importance analysis
- Convergence plots for genetic algorithm

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy scikit-learn statsmodels matplotlib seaborn deap
```

### Running the Notebook

1. **Data Requirements:**
   - `data_crossval.csv` - Cross-validation exercise data
   - `airbnb_london_homework.csv` - Airbnb London rental data

2. **Execution:**
   ```python
   jupyter notebook Mabileau_Baji.ipynb
   ```

3. **Sections:**
   - Section 1: Cross-validation exercises (independent execution)
   - Section 2: Genetic algorithm optimization (independent execution)
   - Section 3: Airbnb price prediction (requires dataset)

## 📝 Project Structure

### Part 1: Cross-Validation
1. Data importation and cleaning
2. Train-test split (80-20)
3. Polynomial regression with degree selection
4. k-NN regression with neighbor optimization
5. Model comparison and visualization

### Part 2: Global Minimum Optimization
1. Camelback function definition
2. Genetic algorithm implementation using DEAP
3. Fitness evaluation and selection mechanisms
4. Hyperparameter tuning experiments
5. Convergence analysis and visualization

### Part 3: Rental Price Prediction
1. **Data Exploration:**
   - Missing value analysis
   - Descriptive statistics
   - Visual data exploration
   - Categorical variable transformation

2. **Feature Engineering:**
   - Location-based statistical tests
   - Correlation and multicollinearity analysis
   - Dimensionality reduction

3. **Model Training:**
   - Random Forest with GridSearchCV
   - k-NN with cross-validated k selection
   - OLS regression baseline
   - Ridge/Lasso regularization

4. **Feature Selection:**
   - Lasso-based automatic selection
   - Genetic algorithm for optimal feature subset
   - BIC criterion for model selection

5. **Model Comparison:**
   - MSE evaluation on test set
   - Density plots of predictions
   - Performance benchmarking

## 🔑 Key Findings

### Cross-Validation Insights
- Optimal polynomial degree identified through systematic CV
- k-NN performance sensitive to neighborhood size
- MSE trends reveal bias-variance tradeoff

### Genetic Algorithm Performance
- Successfully minimizes camelback function to within 0.01 tolerance
- Population size and mutation rate significantly impact convergence
- Demonstrates effectiveness of evolutionary optimization

### Rental Price Prediction
- Multiple models show comparable performance
- Feature selection improves interpretability without sacrificing accuracy
- Regularization techniques effectively handle multicollinearity
- Genetic algorithm provides competitive feature subset selection

## 📊 Statistical Methods

### Hypothesis Testing
- Location-based price difference testing
- Statistical significance of categorical features

### Model Validation
- K-fold cross-validation for robust estimates
- Train-test split for final model evaluation
- MSE as consistent performance metric across all models

## 💡 Learning Outcomes

This project demonstrates:
- Systematic model selection methodology
- Proper cross-validation implementation
- Regularization techniques for real-world data
- Genetic algorithms for optimization and feature selection
- Complete ML pipeline from data cleaning to model deployment
- Comparative analysis of multiple regression techniques

## 🔮 Future Enhancements

- Implement ensemble stacking for improved predictions
- Add neural network regression models
- Incorporate external data sources (neighborhood demographics, transportation)
- Deploy best model as REST API
- Implement A/B testing framework for model updates
- Add time-series components for seasonal pricing

## 👥 Contributors

- **Juliette Mabileau** - Model Development & Analysis
- **Anas Baji** - Model Development & Analysis

## 📚 References

- DEAP Documentation: [https://deap.readthedocs.io/](https://deap.readthedocs.io/)
- Scikit-learn User Guide: [https://scikit-learn.org/](https://scikit-learn.org/)
- Six-Hump Camelback Function: Classic optimization benchmark

## 📄 License

This project is part of academic coursework for M2 IREF program.

---

**Note**: This project showcases practical applications of machine learning regression techniques, evolutionary algorithms, and systematic model evaluation methodologies for real-world predictive modeling tasks.
