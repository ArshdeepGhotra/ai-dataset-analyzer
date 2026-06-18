# AI Dataset Analyzer

AI Dataset Analyzer is a Streamlit-based web application that helps users upload, analyze, clean, visualize, and save dataset history in one place.

The goal of this project is to make dataset analysis easier for users who may not have advanced programming or data science knowledge.

## Features

- User signup and login system
- CSV and Excel file upload
- Dataset overview
- Missing value detection
- Duplicate row detection
- Dataset summary statistics
- Data preprocessing tools
- Interactive data visualizations
- Basic machine learning recommendations
- Saved dataset upload history
- View saved dataset records
- Delete individual saved dataset records
- Clear full dataset history

## Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- Scikit-learn
- SQLite
- OpenPyXL

## Project Structure

```text
ai-dataset-analyzer/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── modules/
│   ├── database.py
│   ├── dataset_analyzer.py
│   ├── preprocessing.py
│   ├── visualizations.py
│   └── ml_recommender.py
│
└── database/
    └── dataset_history.db
```

## How to Run the Project

### 1. Clone the repository

```bash
git clone YOUR_REPOSITORY_LINK
```

### 2. Open the project folder

```bash
cd ai-dataset-analyzer
```

### 3. Create a virtual environment

```bash
python3 -m venv .venv
```

### 4. Activate the virtual environment

For macOS:

```bash
source .venv/bin/activate
```

For Windows:

```bash
.venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit app

```bash
streamlit run app.py
```

## MVP Scope

This project is currently an MVP. It focuses on the main features needed to upload a dataset, understand its structure, clean basic issues, visualize data, receive machine learning suggestions, and save dataset history.

## Future Improvements

- Cloud database integration
- PDF report export
- Advanced machine learning model training
- User profile page
- More advanced preprocessing options
- Streamlit Cloud deployment
- Better UI styling
- Downloadable analysis reports

## Author

Arshdeep Ghotra
