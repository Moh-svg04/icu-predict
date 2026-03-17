ICU Outcomes Predictor: From Theory to Implementation
📌 Project Overview
This project is a functional implementation of a scientific study presented in our research poster: "Can AI Predict ICU Outcomes Better Than Doctors' Scoring Systems?".

While the initial research focused on analyzing existing literature and benchmarks (XGBoost vs. APACHE II), I decided to go one step further by personally conducting the experiment. This application processes real-world clinical data to predict patient mortality and Length of Stay (LOS).

🎓 Academic Context
Origin: Scientific research poster for an oral examination.

Institution: UFR des Sciences et Techniques - Université Le Havre Normandie.

Authors: Mohamed Gueye & Ilyes Hamzaoui.

Goal: Demonstrate the superiority of Machine Learning (XGBoost) over traditional medical scores like APACHE II.

🔬 The Experiment
Inspired by the MIMIC-IV dataset (67,748 patients), this tool reproduces the predictive pipeline described in the poster:

Data Extraction: Processing vital signs (HR, MAP, SpO2) and lab values (Lactate, GCS, Creatinine) from the MIMIC-IV database.

Model Training: Implementing an XGBoost regressor and classifier.

Performance: Aiming for an AUC of 0.918, significantly outperforming the 0.73 average of traditional scoring systems.

🛠️ Tech Stack
Language: Python 3.10+

Data Science: Pandas, Scikit-learn, XGBoost.

Web Interface: Streamlit (for real-time prediction and visualization).

Dataset: MIMIC-IV (PhysioNet).

🚀 How to Run
Clone the repo: git clone https://github.com/votre-repo/icu-predict.git

Install dependencies: pip install -r requirements.txt

Launch the App: streamlit run app.py
