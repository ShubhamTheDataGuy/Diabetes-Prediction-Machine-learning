# Diabetes Prediction Using Machine Learning

This project focuses on building and evaluating machine learning models to predict diabetes based on a dataset of patient information. It explores multiple classification algorithms and compares their performance.

## Table of Contents
- [Dataset](#dataset)
- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Dataset
The dataset used for this project is `diabetes.csv`, which contains various health-related attributes of patients and their diabetes diagnosis status. Key attributes include:
- **Pregnancies**: Number of times pregnant
- **Glucose**: Plasma glucose concentration
- **BloodPressure**: Diastolic blood pressure (mm Hg)
- **SkinThickness**: Triceps skin fold thickness (mm)
- **Insulin**: 2-Hour serum insulin (mu U/ml)
- **BMI**: Body mass index (weight in kg/(height in m)^2)
- **DiabetesPedigreeFunction**: Diabetes pedigree function
- **Age**: Age in years
- **Outcome**: Class label (0: Non-diabetic, 1: Diabetic)

## Project Overview
The project involves:
1. **Data Preprocessing**: Cleaning and scaling the dataset.
2. **Exploratory Data Analysis (EDA)**: Visualizing key patterns and relationships in the data.
3. **Model Building**: Training and evaluating various classification models, including:
   - Logistic Regression
   - K-Nearest Neighbors (KNN)
   - Support Vector Machines (SVM)
   - Decision Trees
   - Random Forest
  
4. **Model Evaluation**: Comparing performance metrics such as accuracy.

## Technologies Used
The following tools and libraries were used:
- **Python**: Programming language
- **Pandas & NumPy**: Data manipulation and numerical operations
- **Seaborn & Matplotlib**: Data visualization
- **scikit-learn**: Machine learning models and evaluation


## Getting Started
### Prerequisites
Ensure you have Python installed and the following libraries:
```bash
pip install pandas numpy seaborn matplotlib scikit-learn lightgbm
```

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/diabetes-prediction.git
   cd diabetes-prediction
   ```
2. Place the `diabetes.csv` file in the project directory.
3. Open the `diabetes_ml.ipynb` notebook in Jupyter or Google Colab.

## Usage
1. Run the cells in the Jupyter notebook to:
   - Load and preprocess the data.
   - Visualize data relationships.
   - Train and evaluate machine learning models.
2. Experiment with hyperparameters to optimize model performance.

## Results
The following models were evaluated on accuracy and other metrics:
- **Logistic Regression**: [84.86%]
- **KNN**: [84.07%]
- **SVM**: [85.39%]
- **Random Forest**: [88.15%]


Detailed performance metrics are available in the notebook.


