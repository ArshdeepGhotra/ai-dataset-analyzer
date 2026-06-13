import streamlit as st
import pandas as pd

from modules.loader import load_dataset
from modules.dataset_analyzer import get_basic_info
from modules.preprocessing import (
    remove_duplicates,
    fill_numeric_missing,
    fill_text_missing,
    drop_missing_rows,
    get_csv_download
)

from modules.visualizer import render_visualizations


st.set_page_config(
    page_title="AI Dataset Analyzer",
    page_icon="📊",
    layout="wide"
)


# ---------------- CUSTOM CSS ----------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(
            135deg,
            #0f172a 0%,
            #111827 50%,
            #020617 100%
        );
        color: #f8fafc;
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 95%;
    }

    h1, h2, h3 {
        color: #f8fafc !important;
    }

    h1 {
        margin-bottom: 0.2rem !important;
    }

    h2, h3 {
        margin-top: 0.4rem !important;
        margin-bottom: 0.6rem !important;
    }

    p {
        margin-bottom: 0.5rem !important;
    }

    div[data-testid="stVerticalBlock"] {
        gap: 0.6rem;
    }

    .hero-box {
        background: rgba(15, 23, 42, 0.88);
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 18px;
        padding: 1rem 1.3rem;
        margin-bottom: 0.9rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
        animation: fadeUp 0.7s ease-out;
    }

    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(
            90deg,
            #38bdf8,
            #818cf8,
            #c084fc
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1rem;
        color: #cbd5e1;
        line-height: 1.5;
        margin-bottom: 0 !important;
    }

    div[data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.9);
        padding: 0.7rem 0.9rem;
        border-radius: 14px;
        border: 1px solid rgba(56, 189, 248, 0.2);
        box-shadow: 0 5px 16px rgba(0, 0, 0, 0.2);
    }

    div[data-testid="stMetric"] label {
        color: #cbd5e1 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #38bdf8 !important;
        font-weight: 800;
    }

    div[data-testid="stTabs"] {
        margin-top: 0.5rem;
    }

    button[data-baseweb="tab"] {
        background: rgba(30, 41, 59, 0.65);
        border-radius: 12px;
        margin-right: 0.35rem;
        padding: 0.4rem 0.8rem;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(15, 23, 42, 0.78);
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 16px;
        padding: 0.8rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
        animation: fadeUp 0.5s ease-out;
    }

    section[data-testid="stFileUploader"] {
        background: rgba(30, 41, 59, 0.7);
        padding: 0.7rem;
        border-radius: 14px;
        border: 1px dashed #38bdf8;
        margin-bottom: 0.5rem;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

    div[data-testid="stAlert"] {
        border-radius: 12px;
        padding: 0.55rem 0.8rem;
    }

    div[data-baseweb="select"] {
        border-radius: 12px;
    }

    div.stDownloadButton > button {
        background: linear-gradient(90deg, #2563eb, #7c3aed);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.65rem 1.1rem;
        font-weight: 700;
        transition: all 0.25s ease;
    }

    div.stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 22px rgba(124, 58, 237, 0.35);
    }

    details {
        background: rgba(15, 23, 42, 0.55) !important;
        border-radius: 14px !important;
    }

    footer {
        visibility: hidden;
    }

    @keyframes fadeUp {
        from {
            opacity: 0;
            transform: translateY(18px);
        }

        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------- HELPER FUNCTION ----------------
def get_dynamic_table_height(
    dataframe,
    max_height=360,
    min_height=115
):
    """
    Adjust table height based on the number of rows.

    This prevents unnecessary empty space when a table
    contains only a few records.
    """

    row_count = len(dataframe)

    if row_count == 0:
        return min_height

    row_height = 35
    header_height = 42
    padding = 18

    calculated_height = (
        header_height
        + (row_count * row_height)
        + padding
    )

    return min(calculated_height, max_height)


# ---------------- HERO SECTION ----------------
st.markdown(
    """
<div class="hero-box">
<h1 class="main-title">📊 AI Dataset Analyzer</h1>
<p class="subtitle">Upload a dataset, analyze its structure, detect missing values, clean the data, compare before-and-after changes, create interactive charts, and download the cleaned file.</p>
</div>
    """,
    unsafe_allow_html=True
)


# ---------------- FILE UPLOADER ----------------
uploaded_file = st.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx", "xls"]
)


if uploaded_file is not None:
    try:
        # Load and analyze the uploaded dataset
        df = load_dataset(uploaded_file)
        info = get_basic_info(df)

        st.success("Dataset uploaded successfully!")

        total_missing = int(
            df.isnull().sum().sum()
        )

        total_duplicates = int(
            df.duplicated().sum()
        )

        numeric_cols = int(
            len(
                df.select_dtypes(
                    include=["number"]
                ).columns
            )
        )

        text_cols = int(
            len(
                df.select_dtypes(
                    include=["object", "string", "category"]
                ).columns
            )
        )

        # Keep a cleaned version available for preprocessing
        # and visualization.
        cleaned_df = df.copy()

        # ---------------- TOP METRICS ----------------
        col1, col2, col3, col4, col5, col6 = st.columns(6)

        with col1:
            st.metric(
                "Rows",
                int(df.shape[0])
            )

        with col2:
            st.metric(
                "Columns",
                int(df.shape[1])
            )

        with col3:
            st.metric(
                "Missing",
                total_missing
            )

        with col4:
            st.metric(
                "Duplicates",
                total_duplicates
            )

        with col5:
            st.metric(
                "Numeric",
                numeric_cols
            )

        with col6:
            st.metric(
                "Text",
                text_cols
            )

        # ---------------- MAIN TABS ----------------
        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "📄 Overview",
                "🔎 Data Quality",
                "🧹 Preprocessing",
                "📈 Visualizations"
            ]
        )

        # ---------------- TAB 1: OVERVIEW ----------------
        with tab1:
            left_col, right_col = st.columns([2, 1])

            with left_col:
                with st.container(border=True):
                    st.subheader("Dataset Preview")

                    preview_df = df.head(8)

                    st.dataframe(
                        preview_df,
                        use_container_width=True,
                        height=get_dynamic_table_height(
                            preview_df,
                            max_height=340
                        )
                    )

            with right_col:
                with st.container(border=True):
                    st.subheader("Dataset Summary")

                    summary_table = pd.DataFrame(
                        {
                            "Metric": [
                                "Total Rows",
                                "Total Columns",
                                "Total Missing Values",
                                "Duplicate Rows",
                                "Numeric Columns",
                                "Text Columns"
                            ],
                            "Value": [
                                int(df.shape[0]),
                                int(df.shape[1]),
                                total_missing,
                                total_duplicates,
                                numeric_cols,
                                text_cols
                            ]
                        }
                    )

                    st.dataframe(
                        summary_table,
                        use_container_width=True,
                        height=get_dynamic_table_height(
                            summary_table,
                            max_height=300
                        )
                    )

            with st.expander(
                "View Column Details",
                expanded=False
            ):
                column_info = pd.DataFrame(
                    {
                        "Column Name": df.columns,
                        "Data Type": df.dtypes.astype(str),
                        "Missing Values": (
                            df.isnull().sum().values
                        ),
                        "Unique Values": (
                            df.nunique(dropna=True).values
                        )
                    }
                )

                st.dataframe(
                    column_info,
                    use_container_width=True,
                    height=get_dynamic_table_height(
                        column_info,
                        max_height=380
                    )
                )

        # ---------------- TAB 2: DATA QUALITY ----------------
        with tab2:
            quality_col1, quality_col2 = st.columns(2)

            with quality_col1:
                with st.container(border=True):
                    st.subheader("Missing Values")

                    if total_missing == 0:
                        st.success(
                            "No missing values found."
                        )

                    else:
                        st.warning(
                            f"Total missing values found: "
                            f"{total_missing}"
                        )

                        missing_info = pd.DataFrame(
                            {
                                "Column Name": df.columns,
                                "Missing Values": (
                                    df.isnull().sum().values
                                )
                            }
                        )

                        missing_info = missing_info[
                            missing_info["Missing Values"] > 0
                        ]

                        st.dataframe(
                            missing_info,
                            use_container_width=True,
                            height=get_dynamic_table_height(
                                missing_info,
                                max_height=260
                            )
                        )

            with quality_col2:
                with st.container(border=True):
                    st.subheader("Duplicate Rows")

                    if total_duplicates == 0:
                        st.success(
                            "No duplicate rows found."
                        )

                    else:
                        st.warning(
                            f"Duplicate rows found: "
                            f"{total_duplicates}"
                        )

                        duplicate_rows = df[
                            df.duplicated(keep=False)
                        ].head(10)

                        st.dataframe(
                            duplicate_rows,
                            use_container_width=True,
                            height=get_dynamic_table_height(
                                duplicate_rows,
                                max_height=300
                            )
                        )

            with st.expander(
                "View Full Dataset",
                expanded=False
            ):
                st.dataframe(
                    df,
                    use_container_width=True,
                    height=430
                )

        # ---------------- TAB 3: PREPROCESSING ----------------
        with tab3:
            with st.container(border=True):
                st.subheader("Data Preprocessing")

                preprocessing_option = st.selectbox(
                    "Choose a preprocessing operation:",
                    [
                        "None",
                        "Remove Duplicate Rows",
                        "Fill Numeric Missing Values with Mean",
                        "Fill Numeric Missing Values with Median",
                        "Fill Numeric Missing Values with Zero",
                        "Fill Text Missing Values with Unknown",
                        "Drop Rows with Missing Values"
                    ],
                    key="preprocessing_operation"
                )

                original_missing = int(
                    df.isnull().sum().sum()
                )

                original_duplicates = int(
                    df.duplicated().sum()
                )

                original_shape = df.shape

                if preprocessing_option == "Remove Duplicate Rows":
                    cleaned_df = remove_duplicates(
                        cleaned_df
                    )

                    st.success(
                        "Duplicate rows removed successfully."
                    )

                elif preprocessing_option == (
                    "Fill Numeric Missing Values with Mean"
                ):
                    cleaned_df = fill_numeric_missing(
                        cleaned_df,
                        method="mean"
                    )

                    st.success(
                        "Numeric missing values filled with mean."
                    )

                elif preprocessing_option == (
                    "Fill Numeric Missing Values with Median"
                ):
                    cleaned_df = fill_numeric_missing(
                        cleaned_df,
                        method="median"
                    )

                    st.success(
                        "Numeric missing values filled with median."
                    )

                elif preprocessing_option == (
                    "Fill Numeric Missing Values with Zero"
                ):
                    cleaned_df = fill_numeric_missing(
                        cleaned_df,
                        method="zero"
                    )

                    st.success(
                        "Numeric missing values filled with zero."
                    )

                elif preprocessing_option == (
                    "Fill Text Missing Values with Unknown"
                ):
                    cleaned_df = fill_text_missing(
                        cleaned_df,
                        value="Unknown"
                    )

                    st.success(
                        "Text missing values filled with 'Unknown'."
                    )

                elif preprocessing_option == (
                    "Drop Rows with Missing Values"
                ):
                    cleaned_df = drop_missing_rows(
                        cleaned_df
                    )

                    st.success(
                        "Rows with missing values dropped successfully."
                    )

            if preprocessing_option != "None":
                cleaned_missing = int(
                    cleaned_df.isnull().sum().sum()
                )

                cleaned_duplicates = int(
                    cleaned_df.duplicated().sum()
                )

                cleaned_shape = cleaned_df.shape

                compare_col, report_col = st.columns(2)

                with compare_col:
                    with st.container(border=True):
                        st.subheader("What Changed?")

                        comparison_df = pd.DataFrame(
                            {
                                "Metric": [
                                    "Rows",
                                    "Columns",
                                    "Total Missing Values",
                                    "Duplicate Rows"
                                ],
                                "Before": [
                                    int(original_shape[0]),
                                    int(original_shape[1]),
                                    original_missing,
                                    original_duplicates
                                ],
                                "After": [
                                    int(cleaned_shape[0]),
                                    int(cleaned_shape[1]),
                                    cleaned_missing,
                                    cleaned_duplicates
                                ]
                            }
                        )

                        st.dataframe(
                            comparison_df,
                            use_container_width=True,
                            height=get_dynamic_table_height(
                                comparison_df,
                                max_height=260
                            )
                        )

                with report_col:
                    with st.container(border=True):
                        st.subheader(
                            "Preprocessing Report"
                        )

                        missing_values_fixed = int(
                            original_missing
                            - cleaned_missing
                        )

                        duplicate_rows_removed = int(
                            original_duplicates
                            - cleaned_duplicates
                        )

                        rows_removed = int(
                            original_shape[0]
                            - cleaned_shape[0]
                        )

                        report_data = pd.DataFrame(
                            {
                                "Report Item": [
                                    "Operation Applied",
                                    "Rows Removed",
                                    "Missing Values Fixed",
                                    "Duplicate Rows Removed"
                                ],
                                "Result": [
                                    preprocessing_option,
                                    rows_removed,
                                    missing_values_fixed,
                                    duplicate_rows_removed
                                ]
                            }
                        )

                        st.dataframe(
                            report_data,
                            use_container_width=True,
                            height=get_dynamic_table_height(
                                report_data,
                                max_height=260
                            )
                        )

                with st.expander(
                    "Rows Affected by Preprocessing",
                    expanded=True
                ):
                    filling_operations = [
                        "Fill Numeric Missing Values with Mean",
                        "Fill Numeric Missing Values with Median",
                        "Fill Numeric Missing Values with Zero",
                        "Fill Text Missing Values with Unknown"
                    ]

                    if preprocessing_option in filling_operations:
                        affected_rows = df[
                            df.isnull().any(axis=1)
                        ]

                        if affected_rows.empty:
                            st.info(
                                "No rows were affected by this "
                                "preprocessing operation."
                            )

                        else:
                            before_col, after_col = st.columns(2)

                            with before_col:
                                st.write("Before")

                                st.dataframe(
                                    affected_rows,
                                    use_container_width=True,
                                    height=get_dynamic_table_height(
                                        affected_rows,
                                        max_height=320
                                    )
                                )

                            with after_col:
                                st.write("After")

                                after_rows = cleaned_df.loc[
                                    affected_rows.index
                                ]

                                st.dataframe(
                                    after_rows,
                                    use_container_width=True,
                                    height=get_dynamic_table_height(
                                        after_rows,
                                        max_height=320
                                    )
                                )

                    elif preprocessing_option == (
                        "Remove Duplicate Rows"
                    ):
                        affected_rows = df[
                            df.duplicated(keep=False)
                        ]

                        if affected_rows.empty:
                            st.info(
                                "No duplicate rows were found."
                            )

                        else:
                            st.write(
                                "Duplicate Rows Before Preprocessing"
                            )

                            st.dataframe(
                                affected_rows,
                                use_container_width=True,
                                height=get_dynamic_table_height(
                                    affected_rows,
                                    max_height=320
                                )
                            )

                    elif preprocessing_option == (
                        "Drop Rows with Missing Values"
                    ):
                        affected_rows = df[
                            df.isnull().any(axis=1)
                        ]

                        if affected_rows.empty:
                            st.info(
                                "No rows with missing values were found."
                            )

                        else:
                            st.write(
                                "Rows Removed During Preprocessing"
                            )

                            st.dataframe(
                                affected_rows,
                                use_container_width=True,
                                height=get_dynamic_table_height(
                                    affected_rows,
                                    max_height=320
                                )
                            )

                final_col, download_col = st.columns([3, 1])

                with final_col:
                    with st.container(border=True):
                        st.subheader(
                            "Final Cleaned Dataset Preview"
                        )

                        final_preview = cleaned_df.head(8)

                        st.dataframe(
                            final_preview,
                            use_container_width=True,
                            height=get_dynamic_table_height(
                                final_preview,
                                max_height=340
                            )
                        )

                with download_col:
                    with st.container(border=True):
                        st.subheader("Export")

                        csv_data = get_csv_download(
                            cleaned_df
                        )

                        st.download_button(
                            label="Download Cleaned CSV",
                            data=csv_data,
                            file_name="cleaned_dataset.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

            else:
                st.info(
                    "Choose a preprocessing operation to view "
                    "the cleaned dataset report and export option."
                )

        # ---------------- TAB 4: VISUALIZATIONS ----------------
        with tab4:
            with st.container(border=True):
                st.caption(
                    "Charts use the cleaned dataset when a "
                    "preprocessing operation is selected. "
                    "Otherwise, they use the original dataset."
                )

                render_visualizations(cleaned_df)

    except Exception as e:
        st.error(
            f"Something went wrong: {e}"
        )

else:
    st.info(
        "Please upload a CSV or Excel file to begin."
    )