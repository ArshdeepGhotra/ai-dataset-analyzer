import pandas as pd


def load_dataset(uploaded_file):
    """
    Load dataset based on file extension.
    Supports CSV and Excel files for MVP.
    """

    if uploaded_file is None:
        return None

    file_name = uploaded_file.name.lower()

    try:
        if file_name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            return df

        elif file_name.endswith(".xlsx") or file_name.endswith(".xls"):
            df = pd.read_excel(uploaded_file)
            return df

        else:
            raise ValueError("Unsupported file format. Please upload CSV or Excel file.")

    except Exception as e:
        raise RuntimeError(f"Error loading dataset: {e}")