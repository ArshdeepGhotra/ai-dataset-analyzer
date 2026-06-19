# AI Dataset Analyzer

AI Dataset Analyzer is a Streamlit web application that helps users upload, analyze, clean, visualize, and manage datasets.

## Features

- User authentication
- CSV and Excel dataset upload
- Dataset overview with rows, columns, missing values, and duplicates
- Column-level analysis
- Data preprocessing tools
- Interactive visualizations
- Machine learning model recommendations
- Dataset history dashboard
- Saved dataset reopening
- SQLite database support

## Tech Stack

- Python
- Streamlit
- Pandas
- Plotly
- Scikit-learn
- SQLite
- Git and GitHub

## Project Structure

```text
ai-dataset-analyzer/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── modules/
│   ├── database.py
│   ├── dataset_analyzer.py
│   ├── preprocessing.py
│   ├── visualizer.py
│   ├── ml_recommender.py
│   └── history_dashboard.py
├── auth/
├── utils/
└── database/

