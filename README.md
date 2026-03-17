# ICU Outcomes Predictor: From Theory to Implementation

## 📌 Project Overview
This project is a functional implementation and a personal reproduction of the scientific study presented in my research poster: **"Can AI Predict ICU Outcomes Better Than Doctors' Scoring Systems?"**. 

Initially, this work started as a **scientific presentation in English** about a research topic that fascinated me. After presenting the existing studies and the theoretical potential of AI in intensive care, I decided to go further and **personally conduct the experiment** to see if I could replicate the state-of-the-art results using real clinical data.

## 🎓 Academic Context
* **Origin:** Scientific research project & poster for an oral examination.
* **Institution:** UFR des Sciences et Techniques - Université Le Havre Normandie.
* **Authors:** Mohamed Gueye & Ilyes Hamzaoui.
* **Core Objective:** To bridge the gap between theoretical research and practical implementation by building a machine learning pipeline from scratch.

## 🖼️ Research Poster
Below is the original poster that inspired this implementation:

![Research Poster](main.poster.png)
*(Note: Replace 'poster_image.png' with the actual path to your image file in the repository)*

## 🔬 The Experiment & Results
Inspired by the MIMIC-IV dataset, this tool reproduces the predictive pipeline described in the poster:
1.  **Data Extraction:** Processing vital signs (Heart Rate, MAP, SpO2) and critical lab values (Lactate, GCS, etc.).
2.  **Model Training:** Implementation of an **XGBoost** classifier for mortality and a regressor for stay duration.
3.  **Performance:** By conducting the study myself, I aimed to reach the **0.918 AUC** benchmark, demonstrating a significant improvement over traditional medical scores (APACHE II).

## 🛠️ Tech Stack
* **Language:** Python 3.10+
* **Machine Learning:** Pandas, Scikit-learn, XGBoost.
* **Web Interface:** Streamlit.
* **Dataset:** MIMIC-IV (PhysioNet).

## 🚀 How to Run
1. `git clone https://github.com/votre-username/icu-predict.git`
2. `pip install -r requirements.txt`
3. `streamlit run app.py`
