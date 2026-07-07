# KD-Engine-Multimodal-AI
A multimodal machine learning web app that predicts diseases from patient symptoms and classifies unstructured clinical notes

KD-Engine: Multimodal Clinical AI
Live Demo: [Insert URL Here]

Project Overview:
The KD-Engine is a multimodal machine learning application designed to process both structured and unstructured clinical data. By integrating two specialized AI engines, this tool assists in diagnostic workflows by providing real-time predictions and confidence analysis.

Core Features:

-Symptom Analysis: A structured data engine utilizing an XGBoost model to predict diseases based on over 400 selectable symptoms.

-Clinical Report NLP: An unstructured data engine using TF-IDF vectorization and Logistic Regression to categorize medical dictations into their respective specialties.

-Confidence Metrics: Real-time probability visualization that allows users to interpret the AI's diagnostic certainty.

Technical Stack:

Language: Python

Frontend: Streamlit

Machine Learning: XGBoost, Scikit-Learn

Data Processing: Pandas, NumPy

Deployment: GitHub & Streamlit Community Cloud
