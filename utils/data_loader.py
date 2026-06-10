import pandas as pd


def load_dataset(uploaded_file):
    """
    Loads dataset based on uploaded file type.
    Supports CSV, Excel, JSON, and TXT files.
    """

    if uploaded_file is None:
        return None, "No file uploaded."

    file_name = uploaded_file.name.lower()

    try:
        if file_name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            return df, "CSV file loaded successfully."

        elif file_name.endswith(".xlsx") or file_name.endswith(".xls"):
            df = pd.read_excel(uploaded_file)
            return df, "Excel file loaded successfully."

        elif file_name.endswith(".json"):
            df = pd.read_json(uploaded_file)
            return df, "JSON file loaded successfully."

        elif file_name.endswith(".txt"):
            df = pd.read_csv(uploaded_file, delimiter=None, engine="python")
            return df, "TXT file loaded successfully."

        else:
            return None, "Unsupported file format. Please upload CSV, Excel, JSON, or TXT."

    except Exception as e:
        return None, f"Error loading file: {str(e)}"