Neonatal Child Health Risk Prediction using Explainable AI
📌 Project Overview

This project develops a Machine Learning-based Neonatal Child Health Risk Prediction System that predicts the health-risk profile of a newborn based on important neonatal health and clinical parameters.

A Random Forest Classifier is used to learn the relationship between neonatal health characteristics and the corresponding risk category. To make the model more transparent and understandable, SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations) are integrated to explain the predictions.

The project demonstrates how Explainable AI (XAI) can be applied to a neonatal child-health prediction problem.

🎯 Objectives
Predict neonatal health risk using machine learning.
Preprocess neonatal health parameters for model training.
Train a Random Forest Classification model.
Evaluate model performance using accuracy, precision, recall, F1-score, and ROC-AUC.
Predict the risk category of an individual neonatal observation.
Use SHAP to explain the contribution of individual features.
Use LIME to provide local explanations for individual predictions.
Identify the neonatal health parameters that have the greatest influence on predictions.
🧠 Technologies Used
Technology	Purpose
Python	Programming language
Pandas	Data processing
NumPy	Numerical operations
Matplotlib	Data visualization
Scikit-learn	Machine learning
SHAP	Global and local explainability
LIME	Local explainability
Jupyter Notebook	Development environment
Streamlit	Application interface
GitHub	Version control and project hosting
📊 Dataset

The project uses a Neonatal Health Dataset containing neonatal health parameters and their corresponding risk classification.

For this educational implementation, the included dataset is synthetic, meaning the records are generated for demonstrating the machine-learning and Explainable AI workflow.

Important Features

The model uses neonatal parameters such as:

Gestational Age
Birth Weight
Apgar Score
Oxygen Saturation
Heart Rate
Temperature
Respiratory Rate
Sex
Delivery Type
Feeding Type
NICU Admission
Target Variable

High Risk

The target variable represents the predicted neonatal health-risk category.

0 → Low Risk


1 → High Risk
🔄 Machine Learning Workflow
                 Neonatal Health Dataset
                           │
                           ▼
                   Data Preprocessing
                           │
                           ▼
                 Feature Preparation
                           │
                           ▼
              Categorical Encoding
                           │
                           ▼
                  Train-Test Split
                           │
                           ▼
              Random Forest Classifier
                           │
                           ▼
                  Risk Prediction
                           │
                 ┌─────────┴─────────┐
                 ▼                   ▼
          Model Evaluation          XAI
       Accuracy / Precision      ┌────┴────┐
       Recall / F1 / ROC-AUC     ▼         ▼
                               SHAP      LIME
                                 │         │
                                 └────┬────┘
                                      ▼
                           Explainable Prediction
🤖 Machine Learning Model
Random Forest Classifier

The project uses a Random Forest Classifier for predicting neonatal health risk.

Random Forest is an ensemble learning algorithm that combines multiple decision trees to produce a classification prediction.

It is suitable for this project because neonatal health risk can depend on complex relationships between multiple parameters such as:

Birth weight
Gestational age
Apgar score
Oxygen saturation
Heart rate
Temperature
Respiratory rate
Model Configuration

The model is configured using:

RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=4,
    class_weight="balanced",
    random_state=42
)

The model produces both a predicted class and a probability for the high-risk class.

📈 Model Evaluation

The model is evaluated using multiple classification metrics.

Accuracy

Accuracy measures the percentage of predictions that are correctly classified.

Accuracy =
Correct Predictions / Total Predictions

Higher accuracy indicates better overall classification performance.

Precision

Precision measures how many observations predicted as high risk are actually high risk.

Precision =
True Positives / (True Positives + False Positives)
Recall

Recall measures how many actual high-risk observations were correctly identified.

Recall =
True Positives / (True Positives + False Negatives)
F1-Score

F1-score combines precision and recall into a single metric.

F1 = 2 × (Precision × Recall)
     / (Precision + Recall)
ROC-AUC

ROC-AUC measures how effectively the model distinguishes between the two risk categories.

A value closer to 1 indicates stronger classification performance.

🔍 Explainable AI with SHAP
What is SHAP?

SHAP (SHapley Additive exPlanations) is an Explainable AI technique used to explain the contribution of individual features to a machine-learning prediction.

SHAP is based on Shapley values from cooperative game theory.

In this project, SHAP is integrated with the Random Forest model to understand why a particular neonatal observation receives a particular risk prediction.

📊 SHAP Interpretation

A SHAP value indicates the contribution of a feature toward the model prediction.

Positive SHAP Value
        ↓
Pushes prediction toward High Risk


Negative SHAP Value
        ↓
Pushes prediction toward Low Risk

Features with larger absolute SHAP values have a greater influence on the model prediction.

For example:

Feature                    Contribution
----------------------------------------
Gestational Age              +0.42
Birth Weight                 +0.31
Oxygen Saturation            +0.27
Apgar Score                  +0.21
Heart Rate                   +0.14
Temperature                  -0.08

The exact SHAP values depend on the trained model and dataset.

📉 SHAP Visualizations

The project generates multiple SHAP visualizations.

1. SHAP Summary Plot

The SHAP summary plot provides an overall view of feature importance across multiple test observations.

It helps identify which neonatal parameters have the greatest influence on the model.

2. SHAP Feature Importance Plot

The SHAP feature-importance plot ranks features according to their average absolute SHAP contribution.

Features with larger values have greater overall influence on the model.

3. Individual SHAP Explanation

SHAP can also be used to explain one individual neonatal prediction.

It shows how the features of that observation contributed toward the final predicted risk.

🔍 Explainable AI with LIME
What is LIME?

LIME (Local Interpretable Model-agnostic Explanations) is an Explainable AI technique that explains an individual machine-learning prediction.

LIME creates small variations around a selected observation and observes how the machine-learning model responds to those changes.

It then creates a simpler local surrogate model to approximate the original model's decision around that observation.

📊 LIME Interpretation

LIME produces local feature weights.

Positive Weight
       ↓
Supports the predicted class


Negative Weight
       ↓
Opposes the predicted class

For example:

Feature                    LIME Weight
--------------------------------------
Low Oxygen Saturation         +0.38
Low Birth Weight              +0.29
Low Apgar Score               +0.24
Gestational Age               +0.18
Heart Rate                    +0.09

The exact values depend on the selected observation and trained model.

📊 SHAP vs LIME
Aspect	SHAP	LIME
Full Name	SHapley Additive exPlanations	Local Interpretable Model-agnostic Explanations
Main Purpose	Feature contribution	Local prediction explanation
Explanation Type	Global and local	Mainly local
Approach	Shapley-value based	Local surrogate model
Model Specificity	TreeExplainer optimized for tree models	Model-agnostic
Output	SHAP values	Feature weights
Usage in Project	Explain Random Forest predictions	Explain individual neonatal predictions

Both methods provide complementary explanations of the machine-learning model.

📁 Project Structure
Neonatal-Health-SHAP-LIME/
│
├── neonatal_health_explainability.ipynb
│
├── data/
│   └── neonatal_health_synthetic.csv
│
├── src/
│   ├── train_model.py
│   ├── explain_model.py
│   └── lime_local.py
│
├── results/
│   ├── shap_summary.png
│   ├── shap_feature_importance.png
│   ├── lime_local_explanation.png
│   ├── confusion_matrix.png
│   └── test_predictions.csv
│
├── app.py
│
├── requirements.txt
│
├── README.md
│
└── LICENSE
⚙️ Installation
1. Clone the Repository
git clone https://github.com/Kabivalavan/neonatal-health-shap-lime.git

Move into the project directory:

cd neonatal-health-shap-lime
2. Install Required Libraries
pip install pandas numpy matplotlib scikit-learn shap lime joblib streamlit jupyter
3. Start Jupyter Notebook
jupyter notebook

Open:

neonatal_health_explainability.ipynb

Then execute the cells sequentially.

▶️ How to Run
Step 1

Clone or download the repository.

Step 2

Install the required Python libraries.

Step 3

Open the Jupyter Notebook.

Step 4

Run the data-preprocessing cells.

Step 5

Train the Random Forest model.

Step 6

Evaluate the model using classification metrics.

Step 7

Generate SHAP explanations.

Step 8

Generate LIME explanations.

Step 9

View the generated visualizations.

Step 10

Run the Streamlit application if required:

streamlit run app.py
🧪 Example Prediction

The trained model can predict the risk category of a neonatal observation based on its health parameters.

Example input:

Gestational Age       : 34 weeks
Birth Weight          : 2.1 kg
Apgar Score            : 6
Oxygen Saturation      : 91%
Heart Rate             : 148 bpm
Temperature             : 36.2 °C
Respiratory Rate        : 58 breaths/min

The model generates a predicted neonatal risk category.

For example:

Predicted Class:
High Risk


High-Risk Probability:
Model dependent

SHAP and LIME can then explain which input features contributed most strongly to this prediction.

📌 Key Features
Machine Learning
Random Forest Classification
Train-test split
Risk prediction
Probability prediction
Classification evaluation
Confusion matrix
Explainable AI
SHAP TreeExplainer
SHAP summary plot
SHAP feature importance
Individual SHAP explanation
LIME local explanation
Feature contribution analysis
Data Processing
Missing-value handling
Numerical feature processing
Categorical feature encoding
Feature scaling where required
Train/test preprocessing pipeline
📚 Project Applications

This project can be useful for demonstrating:

🏥 Neonatal healthcare analytics
📊 Health-risk prediction research
🤖 Explainable machine learning
🔍 Model interpretation
📈 Clinical-data analytics research
🧪 Machine-learning education
🧠 Explainable AI applications
✅ Advantages
Provides a machine-learning-based neonatal risk prediction system.
Uses multiple neonatal health parameters.
Random Forest can model nonlinear relationships.
SHAP explains feature contributions.
LIME provides local explanations for individual predictions.
Provides both prediction and explanation.
Helps understand the behavior of the machine-learning model.
Can be extended with other classification algorithms.
⚠️ Limitations
The dataset used in this project is synthetic and intended for educational demonstration.
The model has not been clinically validated.
Predictions should not be used for diagnosis or treatment decisions.
SHAP and LIME explain model behavior but do not establish medical causation.
A real healthcare deployment would require appropriate clinical datasets, validation, privacy protection, fairness analysis, and professional oversight.
🚀 Future Enhancements

The project can be improved by:

Replacing the synthetic dataset with an approved, de-identified neonatal dataset.
Comparing Random Forest with XGBoost, Gradient Boosting, Logistic Regression and other models.
Hyperparameter optimization using GridSearchCV or RandomizedSearchCV.
Adding ROC and Precision-Recall curves.
Performing cross-validation.
Adding additional SHAP visualizations.
Improving LIME explanations.
Developing a complete Streamlit interface.
Deploying the model as a REST API.
Adding model monitoring and versioning.
Performing fairness and subgroup-performance analysis.
🧑‍💻 Author

Kabivalavan K

Register No: 727823TUIT060

Class & Section: IV IT - E

Course: Explainable AI

Course Code: 231T010

GitHub: https://github.com/Kabivalavan

📜 License

MIT License

⭐ Acknowledgement

This project demonstrates the application of Machine Learning and Explainable AI to neonatal child-health risk prediction.

The integration of SHAP and LIME makes the machine-learning model more transparent by explaining how individual neonatal health parameters contribute to the predicted risk.

The project is developed for academic and educational purposes and demonstrates the use of Explainable AI techniques in a healthcare-related machine-learning scenario.
