# AI Dataset Analyzer

A full-stack Streamlit web application for uploading, analyzing, cleaning, visualizing, and managing CSV and Excel datasets.

Built to make early-stage dataset exploration simpler for students, analysts, and anyone working with structured data.

---

## Features

### Secure User Accounts
- Create an account and log in securely
- Passwords are hashed before being stored
- Each user has their own saved dataset history

### Dataset Upload
- Upload CSV, XLSX, and XLS files
- View dataset previews immediately after upload
- Supports dataset validation and upload safety checks

### Dataset Overview
- Total rows and columns
- Missing-value count
- Duplicate-row count
- Numeric and text column counts
- Column data types
- Unique-value counts

### Data Quality Checks
- Detect missing values by column
- Identify duplicate rows
- Preview duplicate records
- Explore the full uploaded dataset

### Data Preprocessing
- Remove duplicate rows
- Fill numeric missing values with:
  - Mean
  - Median
  - Zero
- Fill text missing values with `Unknown`
- Drop rows containing missing values
- Compare dataset statistics before and after cleaning
- Export cleaned datasets as CSV files

### Interactive Visualizations
- Histograms
- Box plots
- Line charts
- Bar charts
- Scatter plots
- Correlation heatmaps

### ML Recommendations
- Select a potential target column
- Detect likely classification or regression tasks
- Receive suitable machine-learning model recommendations
- View dataset suitability information before training a model

### Dataset History Dashboard
- Automatically save uploaded dataset summaries
- Reopen previously saved datasets
- View upload history and dataset metrics
- Delete individual history records
- Clear saved dataset history

---

## Tech Stack

- **Python**
- **Streamlit**
- **Pandas**
- **Plotly**
- **Scikit-learn**
- **SQLite**
- **OpenPyXL**
- **Git and GitHub**

---

## Project Structure

```text
ai-dataset-analyzer/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── .streamlit/
│   └── config.toml
│
├── modules/
│   ├── database.py
│   ├── dataset_analyzer.py
│   ├── history_dashboard.py
│   ├── loader.py
│   ├── ml_recommender.py
│   ├── preprocessing.py
│   └── visualizer.py
│
├── auth/
├── database/
└── utils/



## Example Workflow

...

## Key Learning Outcomes

This project was built to strengthen practical skills in:

- Python application development
- Streamlit interface design
- Data analysis with Pandas
- Data cleaning and preprocessing
- Interactive visualization with Plotly
- SQLite database integration
- User authentication workflows
- Git and GitHub version control
- Deployable project structure

---

## Future Improvements

- Train and compare ML models directly in the app
- Add dataset profiling reports
- Add additional file formats such as JSON
- Add dark/light theme selection
- Allow users to rename saved datasets
- Add cloud database support for production deployment
- Build a React frontend with a Python API backend

---

## Author
ARSHDEEP GHOTRA

Bachelor's of Computing Science
Thomspson Rivers University