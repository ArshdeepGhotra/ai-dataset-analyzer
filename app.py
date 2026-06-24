import hashlib
import re
from html import escape

import pandas as pd
import streamlit as st

from modules.database import (
    create_user,
    init_db,
    login_user,
    save_dataset_summary,
)
from modules.history_dashboard import render_history_dashboard
from modules.loader import load_dataset
from modules.ml_recommender import render_ml_recommendations
from modules.preprocessing import (
    drop_missing_rows,
    fill_numeric_missing,
    fill_text_missing,
    get_csv_download,
    remove_duplicates,
)
from modules.visualizer import render_visualizations


st.set_page_config(
    page_title="AI Dataset Analyzer",
    page_icon="📊",
    layout="wide",
)


# ---------------- APP SETTINGS ----------------
APP_VERSION = "1.0"
MAX_UPLOAD_SIZE_MB = 50
ALLOWED_FILE_EXTENSIONS = {"csv", "xlsx", "xls"}


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
    }

    section[data-testid="stFileUploader"] {
        background: rgba(30, 41, 59, 0.7);
        padding: 0.7rem;
        border-radius: 14px;
        border: 1px dashed #38bdf8;
    }

    div.stDownloadButton > button {
        background: linear-gradient(90deg, #2563eb, #7c3aed);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.65rem 1.1rem;
        font-weight: 700;
    }

    section[data-testid="stSidebar"] > div:first-child {
        background: rgba(2, 6, 23, 0.94);
        border-right: 1px solid rgba(148, 163, 184, 0.14);
    }

    .sidebar-user {
        background: rgba(30, 41, 59, 0.72);
        border: 1px solid rgba(56, 189, 248, 0.20);
        border-radius: 14px;
        padding: 0.8rem;
        margin-bottom: 0.8rem;
    }

    .sidebar-user-name {
        color: #f8fafc;
        font-weight: 700;
        margin: 0;
    }

    .sidebar-user-email {
        color: #94a3b8;
        font-size: 0.82rem;
        overflow-wrap: anywhere;
    }

    .app-footer {
        color: #64748b;
        font-size: 0.82rem;
        text-align: center;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(148, 163, 184, 0.13);
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
    unsafe_allow_html=True,
)


# ---------------- DATABASE AND SESSION STATE ----------------
init_db()

SESSION_DEFAULTS = {
    "logged_in": False,
    "user": None,
    "last_saved_file_signature": None,
    "loaded_history_df": None,
    "loaded_history_file_name": None,
    "uploader_key_version": 0,
}

for key, value in SESSION_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ---------------- HELPER FUNCTIONS ----------------
def get_dynamic_table_height(
    dataframe: pd.DataFrame,
    max_height: int = 360,
    min_height: int = 115,
) -> int:
    """Choose a compact table height for the current number of rows."""

    if dataframe.empty:
        return min_height

    return min(
        42 + (len(dataframe) * 35) + 18,
        max_height,
    )


def get_file_extension(file_name: str) -> str:
    """Return a lowercase extension without the dot."""

    if "." not in file_name:
        return ""

    return file_name.rsplit(".", 1)[-1].lower()


def format_file_size(file_size_bytes: int) -> str:
    """Return a readable file-size label."""

    if file_size_bytes < 1024:
        return f"{file_size_bytes} B"

    if file_size_bytes < 1024 * 1024:
        return f"{file_size_bytes / 1024:.1f} KB"

    return f"{file_size_bytes / (1024 * 1024):.1f} MB"


def validate_uploaded_file(uploaded_file) -> tuple[bool, str]:
    """Validate extension, empty files, and file size before loading."""

    extension = get_file_extension(uploaded_file.name)

    if extension not in ALLOWED_FILE_EXTENSIONS:
        return (
            False,
            "Unsupported file type. Upload a CSV, XLSX, or XLS file.",
        )

    file_size = len(uploaded_file.getvalue())

    if file_size == 0:
        return (
            False,
            "This file is empty. Choose a dataset with data in it.",
        )

    if file_size > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        return (
            False,
            f"File is too large. The maximum supported size is "
            f"{MAX_UPLOAD_SIZE_MB} MB.",
        )

    return True, ""


def clear_current_dataset() -> None:
    """Remove both uploaded and history-loaded dataset state."""

    st.session_state.loaded_history_df = None
    st.session_state.loaded_history_file_name = None
    st.session_state.last_saved_file_signature = None
    st.session_state.uploader_key_version += 1


def render_footer() -> None:
    """Show the app footer."""

    st.markdown(
        f"""
        <div class="app-footer">
            AI Dataset Analyzer · Version {APP_VERSION} ·
            Built with Streamlit, Pandas, and Plotly
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_dataset_state() -> None:
    """Show guidance before a dataset is selected."""

    st.markdown("### Start with a dataset")

    st.info(
        "Upload a CSV, XLSX, or XLS file above to begin. "
        f"Files up to {MAX_UPLOAD_SIZE_MB} MB are supported."
    )

    upload_col, analyze_col, clean_col = st.columns(3)

    with upload_col:
        with st.container(border=True):
            st.markdown("#### 1. Upload")
            st.write(
                "Choose a CSV or Excel dataset. Its summary is "
                "saved to your history automatically."
            )

    with analyze_col:
        with st.container(border=True):
            st.markdown("#### 2. Analyze")
            st.write(
                "Review rows, columns, missing values, duplicates, "
                "column types, and data-quality issues."
            )

    with clean_col:
        with st.container(border=True):
            st.markdown("#### 3. Clean & explore")
            st.write(
                "Apply a cleaning operation, compare results, make "
                "charts, and export a cleaned CSV."
            )

    with st.expander("Privacy and data handling", expanded=False):
        st.write(
            "Only upload datasets you are allowed to use. Avoid "
            "passwords, financial details, government IDs, or other "
            "confidential personal information."
        )

        st.write(
            "Dataset summaries and saved history use the database "
            "configured for this application."
        )


# ---------------- AUTH PAGE ----------------
def render_auth_page() -> None:
    """Display login and account creation."""

    st.markdown(
        """
        <div class="hero-box">
            <h1 class="main-title">📊 AI Dataset Analyzer</h1>
            <p class="subtitle">
                Upload datasets, check data quality, clean data,
                build interactive charts, and save your analysis history.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    login_tab, signup_tab = st.tabs(
        ["Login", "Create Account"]
    )

    with login_tab:
        st.subheader("Welcome back")

        email = st.text_input(
            "Email",
            key="login_email",
            placeholder="you@example.com",
        ).strip().lower()

        password = st.text_input(
            "Password",
            type="password",
            key="login_password",
        )

        if st.button("Login", width="stretch"):
            if not email or not password:
                st.error("Enter both your email and password.")

            else:
                user = login_user(email, password)

                if user:
                    st.session_state.logged_in = True
                    st.session_state.user = user
                    st.session_state.last_saved_file_signature = None

                    st.success("Login successful.")
                    st.rerun()

                else:
                    st.error("Invalid email or password.")

    with signup_tab:
        st.subheader("Create your account")

        name = st.text_input(
            "Full Name",
            key="signup_name",
            placeholder="Your name",
        ).strip()

        email = st.text_input(
            "Email",
            key="signup_email",
            placeholder="you@example.com",
        ).strip().lower()

        password = st.text_input(
            "Password",
            type="password",
            key="signup_password",
            help="Use at least 6 characters.",
        )

        confirm_password = st.text_input(
            "Confirm Password",
            type="password",
            key="signup_confirm_password",
        )

        if st.button(
            "Create Account",
            width="stretch",
        ):
            if not name or not email or not password or not confirm_password:
                st.error("Please fill in all fields.")

            elif not re.fullmatch(
                r"[^@\s]+@[^@\s]+\.[^@\s]+",
                email,
            ):
                st.error("Enter a valid email address.")

            elif password != confirm_password:
                st.error("Passwords do not match.")

            elif len(password) < 6:
                st.error("Password must be at least 6 characters.")

            else:
                success, message = create_user(
                    name=name,
                    email=email,
                    password=password,
                )

                if not success:
                    st.error(message)

                else:
                    user = login_user(email, password)

                    if user:
                        st.session_state.logged_in = True
                        st.session_state.user = user
                        st.session_state.last_saved_file_signature = None

                        st.success("Account created successfully.")
                        st.rerun()

                    else:
                        st.success(message)
                        st.info(
                            "Your account was created. Please log in."
                        )


# ---------------- LOGIN GATE ----------------
if not st.session_state.logged_in:
    render_auth_page()
    render_footer()
    st.stop()


# ---------------- HERO ----------------
st.markdown(
    """
    <div class="hero-box">
        <h1 class="main-title">📊 AI Dataset Analyzer</h1>
        <p class="subtitle">
            Upload a dataset, assess its quality, clean it,
            compare before-and-after results, build interactive charts,
            and export a cleaned CSV.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------- SIDEBAR ----------------
user = st.session_state.user

user_name = escape(str(user["name"]))
user_email = escape(str(user["email"]))

with st.sidebar:
    st.markdown("## Workspace")

    st.markdown(
        f"""
        <div class="sidebar-user">
            <p class="sidebar-user-name">{user_name}</p>
            <p class="sidebar-user-email">{user_email}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected_page = st.radio(
        "Navigation",
        ["Dashboard", "Dataset History"],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("#### How to use the app")
    st.caption("1. Upload a CSV or Excel file.")
    st.caption("2. Review overview and data quality checks.")
    st.caption("3. Clean, visualize, and export the data.")

    with st.expander(
        "Privacy and data handling",
        expanded=False,
    ):
        st.caption(
            "Avoid uploading confidential or sensitive personal "
            "information. Use datasets you are authorized to analyze."
        )

    st.divider()

    if st.button("Logout", width="stretch"):
        clear_current_dataset()
        st.session_state.logged_in = False
        st.session_state.user = None
        st.rerun()

    st.caption(f"Version {APP_VERSION}")


if selected_page == "Dataset History":
    st.markdown("## Dataset History")
    st.caption("Review and reload datasets saved to your account.")

    render_history_dashboard(
        st.session_state.user["id"]
    )

    render_footer()
    st.stop()


# ---------------- FILE UPLOADER ----------------
with st.container(border=True):
    title_col, note_col = st.columns([3, 2])

    with title_col:
        st.subheader("Upload a dataset")

        st.caption(
            f"Supported formats: CSV, XLSX, XLS · "
            f"Maximum size: {MAX_UPLOAD_SIZE_MB} MB."
        )

    with note_col:
        st.caption(
            "Your upload is checked before analysis. Dataset "
            "summaries are saved to your account history."
        )

    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=sorted(ALLOWED_FILE_EXTENSIONS),
        key=(
            "dataset_uploader_"
            f"{st.session_state.uploader_key_version}"
        ),
        help="Upload a CSV, XLSX, or XLS file.",
    )


history_df_loaded = st.session_state.get(
    "loaded_history_df"
)

history_file_name = st.session_state.get(
    "loaded_history_file_name"
)


if uploaded_file is not None:
    is_valid_file, validation_message = validate_uploaded_file(
        uploaded_file
    )

    if not is_valid_file:
        st.error(validation_message)
        uploaded_file = None


# ---------------- DASHBOARD ----------------
if uploaded_file is None and history_df_loaded is None:
    render_empty_dataset_state()
    render_footer()
    st.stop()


try:
    if uploaded_file is not None:
        df = load_dataset(uploaded_file)

        active_file_name = uploaded_file.name

        upload_size_label = format_file_size(
            len(uploaded_file.getvalue())
        )

        is_loaded_from_history = False

        st.session_state.loaded_history_df = None
        st.session_state.loaded_history_file_name = None

    else:
        df = history_df_loaded.copy()

        active_file_name = (
            history_file_name or "saved_dataset.csv"
        )

        upload_size_label = "Loaded from history"
        is_loaded_from_history = True

    active_file_name = str(active_file_name)

    if "." in active_file_name:
        base_dataset_name = active_file_name.rsplit(
            ".",
            1,
        )[0]
    else:
        base_dataset_name = active_file_name

    cleaned_export_file_name = (
        f"{base_dataset_name}_cleaned.csv"
    )

    status_col, action_col = st.columns([5, 1])

    with status_col:
        source_label = (
            "Saved dataset"
            if is_loaded_from_history
            else "Current upload"
        )

        st.caption(
            f"{source_label}: **{active_file_name}** · "
            f"{upload_size_label}"
        )

    with action_col:
        if st.button(
            "Remove",
            key="remove_current_dataset",
            width="stretch",
            help=(
                "Clear this dataset and return to the upload screen."
            ),
        ):
            clear_current_dataset()
            st.rerun()

    if df.empty:
        st.warning(
            "This dataset has no rows. You can inspect columns, "
            "but charts and ML recommendations may be limited."
        )

    total_missing = int(df.isnull().sum().sum())

    total_duplicates = int(df.duplicated().sum())

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

    # ---------------- AUTO SAVE DATASET SUMMARY ----------------
    if not is_loaded_from_history:
        file_bytes = uploaded_file.getvalue()

        file_signature = hashlib.sha256(
            file_bytes
        ).hexdigest()

        if st.session_state.last_saved_file_signature != file_signature:
            success, message = save_dataset_summary(
                user_id=st.session_state.user["id"],
                file_name=active_file_name,
                file_type=(
                    get_file_extension(active_file_name).upper()
                    or "UNKNOWN"
                ),
                file_size_kb=round(
                    len(file_bytes) / 1024,
                    2,
                ),
                file_signature=file_signature,
                total_rows=int(df.shape[0]),
                total_columns=int(df.shape[1]),
                missing_values=total_missing,
                duplicate_rows=total_duplicates,
                numeric_columns=numeric_cols,
                categorical_columns=text_cols,
                dataset_csv=df.to_csv(index=False),
            )

            st.session_state.last_saved_file_signature = (
                file_signature
            )

            if success:
                st.caption(
                    "✓ Dataset summary saved to your history."
                )
            else:
                st.caption(f"History status: {message}")

    else:
        st.caption(
            "You are working with a dataset loaded from your "
            "saved history."
        )

    cleaned_df = df.copy()

    # ---------------- TOP METRICS ----------------
    metrics = [
        ("Rows", int(df.shape[0])),
        ("Columns", int(df.shape[1])),
        ("Missing", total_missing),
        ("Duplicates", total_duplicates),
        ("Numeric", numeric_cols),
        ("Text", text_cols),
    ]

    for column, metric_data in zip(
        st.columns(6),
        metrics,
    ):
        label, value = metric_data

        with column:
            st.metric(label, value)

    # ---------------- MAIN TABS ----------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📄 Overview",
            "🔎 Data Quality",
            "🧹 Preprocessing",
            "📈 Visualizations",
            "🤖 ML Recommendations",
        ]
    )

    # ---------------- OVERVIEW ----------------
    with tab1:
        preview_col, summary_col = st.columns([2, 1])

        with preview_col:
            with st.container(border=True):
                st.subheader("Dataset Preview")

                preview_df = df.head(8)

                st.dataframe(
                    preview_df,
                    width="stretch",
                    height=get_dynamic_table_height(
                        preview_df,
                        max_height=340,
                    ),
                )

        with summary_col:
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
                            "Text Columns",
                        ],
                        "Value": [
                            int(df.shape[0]),
                            int(df.shape[1]),
                            total_missing,
                            total_duplicates,
                            numeric_cols,
                            text_cols,
                        ],
                    }
                )

                st.dataframe(
                    summary_table,
                    width="stretch",
                    height=get_dynamic_table_height(
                        summary_table,
                        max_height=300,
                    ),
                )

        with st.expander(
            "View Column Details",
            expanded=False,
        ):
            column_info = pd.DataFrame(
                {
                    "Column Name": df.columns,
                    "Data Type": df.dtypes.astype(str),
                    "Missing Values": df.isnull().sum().values,
                    "Unique Values": df.nunique(
                        dropna=True
                    ).values,
                }
            )

            st.dataframe(
                column_info,
                width="stretch",
                height=get_dynamic_table_height(
                    column_info,
                    max_height=380,
                ),
            )

    # ---------------- DATA QUALITY ----------------
    with tab2:
        missing_col, duplicates_col = st.columns(2)

        with missing_col:
            with st.container(border=True):
                st.subheader("Missing Values")

                if total_missing == 0:
                    st.success("No missing values found.")

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
                            ),
                        }
                    )

                    missing_info = missing_info[
                        missing_info["Missing Values"] > 0
                    ]

                    st.dataframe(
                        missing_info,
                        width="stretch",
                        height=get_dynamic_table_height(
                            missing_info,
                            max_height=260,
                        ),
                    )

        with duplicates_col:
            with st.container(border=True):
                st.subheader("Duplicate Rows")

                if total_duplicates == 0:
                    st.success("No duplicate rows found.")

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
                        width="stretch",
                        height=get_dynamic_table_height(
                            duplicate_rows,
                            max_height=300,
                        ),
                    )

        with st.expander(
            "View Full Dataset",
            expanded=False,
        ):
            st.dataframe(
                df,
                width="stretch",
                height=430,
            )

    # ---------------- PREPROCESSING ----------------
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
                    "Drop Rows with Missing Values",
                ],
                key="preprocessing_operation",
            )

        original_missing = total_missing
        original_duplicates = total_duplicates
        original_shape = df.shape

        if preprocessing_option == "Remove Duplicate Rows":
            cleaned_df = remove_duplicates(cleaned_df)

            st.success("Duplicate rows removed successfully.")

        elif preprocessing_option == (
            "Fill Numeric Missing Values with Mean"
        ):
            cleaned_df = fill_numeric_missing(
                cleaned_df,
                method="mean",
            )

            st.success(
                "Numeric missing values filled with mean."
            )

        elif preprocessing_option == (
            "Fill Numeric Missing Values with Median"
        ):
            cleaned_df = fill_numeric_missing(
                cleaned_df,
                method="median",
            )

            st.success(
                "Numeric missing values filled with median."
            )

        elif preprocessing_option == (
            "Fill Numeric Missing Values with Zero"
        ):
            cleaned_df = fill_numeric_missing(
                cleaned_df,
                method="zero",
            )

            st.success(
                "Numeric missing values filled with zero."
            )

        elif preprocessing_option == (
            "Fill Text Missing Values with Unknown"
        ):
            cleaned_df = fill_text_missing(
                cleaned_df,
                value="Unknown",
            )

            st.success(
                "Text missing values filled with 'Unknown'."
            )

        elif preprocessing_option == (
            "Drop Rows with Missing Values"
        ):
            cleaned_df = drop_missing_rows(cleaned_df)

            st.success(
                "Rows with missing values dropped successfully."
            )

        if preprocessing_option == "None":
            st.info(
                "Choose a preprocessing operation to view the "
                "cleaned dataset report and export option."
            )

        else:
            cleaned_missing = int(
                cleaned_df.isnull().sum().sum()
            )

            cleaned_duplicates = int(
                cleaned_df.duplicated().sum()
            )

            cleaned_shape = cleaned_df.shape

            comparison_col, report_col = st.columns(2)

            with comparison_col:
                with st.container(border=True):
                    st.subheader("What Changed?")

                    comparison_df = pd.DataFrame(
                        {
                            "Metric": [
                                "Rows",
                                "Columns",
                                "Total Missing Values",
                                "Duplicate Rows",
                            ],
                            "Before": [
                                int(original_shape[0]),
                                int(original_shape[1]),
                                original_missing,
                                original_duplicates,
                            ],
                            "After": [
                                int(cleaned_shape[0]),
                                int(cleaned_shape[1]),
                                cleaned_missing,
                                cleaned_duplicates,
                            ],
                        }
                    )

                    st.dataframe(
                        comparison_df,
                        width="stretch",
                        height=get_dynamic_table_height(
                            comparison_df,
                            max_height=260,
                        ),
                    )

            with report_col:
                with st.container(border=True):
                    st.subheader("Preprocessing Report")

                    report_df = pd.DataFrame(
                        {
                            "Report Item": [
                                "Operation Applied",
                                "Rows Removed",
                                "Missing Values Fixed",
                                "Duplicate Rows Removed",
                            ],
                            "Result": [
                                preprocessing_option,
                                int(
                                    original_shape[0]
                                    - cleaned_shape[0]
                                ),
                                int(
                                    original_missing
                                    - cleaned_missing
                                ),
                                int(
                                    original_duplicates
                                    - cleaned_duplicates
                                ),
                            ],
                        }
                    )

                    st.dataframe(
                        report_df,
                        width="stretch",
                        height=get_dynamic_table_height(
                            report_df,
                            max_height=260,
                        ),
                    )

            with st.expander(
                "Rows Affected by Preprocessing",
                expanded=True,
            ):
                filling_operations = {
                    "Fill Numeric Missing Values with Mean",
                    "Fill Numeric Missing Values with Median",
                    "Fill Numeric Missing Values with Zero",
                    "Fill Text Missing Values with Unknown",
                }

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
                                width="stretch",
                                height=get_dynamic_table_height(
                                    affected_rows,
                                    max_height=320,
                                ),
                            )

                        with after_col:
                            st.write("After")

                            st.dataframe(
                                cleaned_df.loc[
                                    affected_rows.index
                                ],
                                width="stretch",
                                height=get_dynamic_table_height(
                                    affected_rows,
                                    max_height=320,
                                ),
                            )

                elif preprocessing_option == (
                    "Remove Duplicate Rows"
                ):
                    affected_rows = df[
                        df.duplicated(keep=False)
                    ]

                    if affected_rows.empty:
                        st.info("No duplicate rows were found.")

                    else:
                        st.write(
                            "Duplicate Rows Before Preprocessing"
                        )

                        st.dataframe(
                            affected_rows,
                            width="stretch",
                            height=get_dynamic_table_height(
                                affected_rows,
                                max_height=320,
                            ),
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
                            width="stretch",
                            height=get_dynamic_table_height(
                                affected_rows,
                                max_height=320,
                            ),
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
                        width="stretch",
                        height=get_dynamic_table_height(
                            final_preview,
                            max_height=340,
                        ),
                    )

            with download_col:
                with st.container(border=True):
                    st.subheader("Export")

                    st.download_button(
                        label="Download Cleaned CSV",
                        data=get_csv_download(cleaned_df),
                        file_name=cleaned_export_file_name,
                        mime="text/csv",
                        width="stretch",
                    )

    # ---------------- VISUALIZATIONS ----------------
    with tab4:
        with st.container(border=True):
            st.caption(
                "Charts use the cleaned dataset when a "
                "preprocessing operation is selected. Otherwise, "
                "they use the original dataset."
            )

            render_visualizations(cleaned_df)

    # ---------------- ML RECOMMENDATIONS ----------------
    with tab5:
        st.caption(
            "ML recommendations use the cleaned dataset when a "
            "preprocessing operation is selected."
        )

        render_ml_recommendations(cleaned_df)

except Exception as error:
    st.error(
        "We could not analyze this file. Confirm that it is a valid "
        "CSV, XLSX, or XLS dataset and try again."
    )

    with st.expander(
        "Technical details",
        expanded=False,
    ):
        st.code(str(error))

render_footer()