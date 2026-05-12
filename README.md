\# CardioRisk Predict: Cardiovascular Disease Risk Prediction 🏥

\![Python\](https://img.shields.io/badge/python-3.8+-blue.svg)
\![Streamlit\](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)
\![Machine
Learning\](https://img.shields.io/badge/ML-CatBoost-green.svg)

\##  Project Overview This project aims to predict the risk of
cardiovascular disease (CVD) using clinical and lifestyle data. By
leveraging advanced machine learning algorithms like \*\*CatBoost\*\*,
we provide a data-driven approach to early screening, which is more
effective than traditional single-threshold medical checks.

\*\*Key Achievement:\*\* Our final model reached an \*\*ROC-AUC of
0.8042\*\*.

\##  Team Members (Group CSS 324) \* \*\*Inkar Tlegenova\*\* \*
\*\*Zhuldyzay Kalka\*\* \* \*\*Aiken Otargali\*\* \* \*University:\* SDU
University, Spring 2026

\##  Interactive Demo We have developed an interactive web application
using \*\*Streamlit\*\*. \* \*\*Live Prediction:\*\* Users can input
their biometrics (age, weight, blood pressure) and lifestyle habits to
get a real-time risk assessment. \* \*\*Visualizations:\*\* Includes EDA
charts and model performance comparisons.

\##  Dataset & Preprocessing \* \*\*Source:\*\* Kaggle (70,000 patient
records). \* \*\*Cleaning:\*\* Handled outliers in height/weight and
corrected physiologically impossible blood pressure values (e.g.,
\$ap\\\_hi \< ap\\\_lo\$). \* \*\*Feature Engineering:\*\* \* \`BMI\`
(Body Mass Index) calculated from height/weight. \* \`age_years\`
converted from days for better interpretability.

\##  Technologies Used \* \*\*Languages:\*\* Python \*
\*\*Libraries:\*\* Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn \*
\*\*Models:\*\* CatBoost (Best), XGBoost, LightGBM, Random Forest,
Logistic Regression \* \*\*Deployment:\*\* Streamlit

\##  Model Performance \| Model \| Accuracy \| Precision \| Recall \|
ROC-AUC \| \| :\-\-- \| :\-\-- \| :\-\-- \| :\-\-- \| :\-\-- \| \|
\*\*CatBoost\*\* \| \*\*73.56%\*\* \| \*\*75.64%\*\* \| \*\*68.63%\*\*
\| \*\*0.8042\*\* \| \| Gradient Boosting \| 73.00% \| 74.47% \| 67.37%
\| 0.8029 \| \| Random Forest \| 72.00% \| 73.00% \| 67.00% \| 0.7707 \|

\##  How to Run 1. \*\*Clone the repository:\*\* \`\`\`bash git clone
https://github.com/ВАШ_НИК/CardioRisk-Predict.git \`\`\` 2. \*\*Install
dependencies:\*\* \`\`\`bash pip install -r requirements.txt \`\`\` 3.
\*\*Run the Streamlit App:\*\* \`\`\`bash streamlit run app.py \`\`\`

\##  Repository Structure \* \`MLFINAL total.ipynb\` - Full data
analysis and model training pipeline. \*
\`cleaned_cardiovascular_dataset.csv\` - Preprocessed data used for
modeling. \* \`CardioRisk_Report.pdf\` - Detailed project documentation.
\* \`app.py\` - Streamlit application code. \*
\`CHD_preprocessed.csv\` - Secondary dataset for unseen data testing.
