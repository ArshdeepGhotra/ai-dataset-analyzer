import streamlit as st
import pandas as pd
from modules.loader import load_dataset


st.set_page_config(
    page_title="AI Dataset Analyzer",
    page_icon="📊",
    layout="wide"
)


st.title("📊 AI Dataset Analyzer")
st.write("Upload your dataset and get an instant overview.")


uploaded_file = st.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx", "xls"]
)


if uploaded_file is not None:
    try:
        df = load_dataset(uploaded_file)

        st.success("Dataset uploaded successfully!")

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        st.subheader("Basic Dataset Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Rows", df.shape[0])

        with col2:
            st.metric("Columns", df.shape[1])

        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())

        st.subheader("Column Details")

        column_info = pd.DataFrame({
            "Column Name": df.columns,
            "Data Type": df.dtypes.astype(str),
            "Missing Values": df.isnull().sum().values,
            "Unique Values": df.nunique().values
        })

        st.dataframe(column_info)

    except Exception as e:
        st.error(f"Something went wrong: {e}")

else:
    st.info("Please upload a CSV or Excel file to begin.")