import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


def _make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _auto_convert_numeric_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for column in df.columns:
        if df[column].dtype == "object" or str(df[column].dtype) == "string":
            cleaned_series = (
                df[column]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("$", "", regex=False)
                .str.replace("%", "", regex=False)
                .str.strip()
            )

            converted_series = pd.to_numeric(cleaned_series, errors="coerce")

            non_null_original = df[column].notna().sum()
            non_null_converted = converted_series.notna().sum()

            if non_null_original > 0:
                conversion_ratio = non_null_converted / non_null_original

                if conversion_ratio >= 0.8:
                    df[column] = converted_series

    return df


def _get_target_recommendation(df: pd.DataFrame) -> pd.DataFrame:
    recommendations = []

    for column in df.columns:
        unique_count = df[column].nunique(dropna=True)
        missing_count = df[column].isnull().sum()
        dtype = str(df[column].dtype)

        if unique_count < 2:
            continue

        if pd.api.types.is_numeric_dtype(df[column]):
            if unique_count > 10:
                task_type = "Regression"
                reason = "Numeric column with many unique values"
            else:
                task_type = "Classification"
                reason = "Numeric column with limited unique values"

            recommendations.append(
                {
                    "Column": column,
                    "Recommended Task": task_type,
                    "Unique Values": unique_count,
                    "Missing Values": missing_count,
                    "Data Type": dtype,
                    "Reason": reason,
                }
            )

        else:
            if unique_count <= 100:
                recommendations.append(
                    {
                        "Column": column,
                        "Recommended Task": "Classification",
                        "Unique Values": unique_count,
                        "Missing Values": missing_count,
                        "Data Type": dtype,
                        "Reason": "Categorical column with manageable classes",
                    }
                )

    return pd.DataFrame(recommendations)


def _prepare_feature_dataframe(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()

    for column in X.columns:
        if pd.api.types.is_datetime64_any_dtype(X[column]):
            X[column] = X[column].astype("int64") // 10**9

    return X


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category", "string", "bool"]
    ).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", _make_one_hot_encoder()),
        ]
    )

    transformers = []

    if numeric_features:
        transformers.append(("num", numeric_transformer, numeric_features))

    if categorical_features:
        transformers.append(("cat", categorical_transformer, categorical_features))

    return ColumnTransformer(transformers=transformers)


def _get_models(task_type: str):
    if task_type == "Regression":
        return {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(random_state=42),
            "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42),
        }

    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest Classifier": RandomForestClassifier(random_state=42),
        "Gradient Boosting Classifier": GradientBoostingClassifier(random_state=42),
    }


def _create_bar_chart(results_df: pd.DataFrame, task_type: str) -> None:
    if task_type == "Regression":
        y_metric = "R2 Score"
        title = "Model Comparison by R2 Score"
    else:
        y_metric = "Accuracy"
        title = "Model Comparison by Accuracy"

    fig = px.bar(
        results_df,
        x="Model",
        y=y_metric,
        text=y_metric,
        title=title,
    )

    fig.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside",
    )

    st.plotly_chart(fig, width="stretch")


def _create_scatter_chart(results_df: pd.DataFrame, task_type: str) -> None:
    if task_type == "Regression":
        x_metric = "RMSE"
        y_metric = "R2 Score"
        title = "Regression Model Comparison: RMSE vs R2 Score"
    else:
        x_metric = "Accuracy"
        y_metric = "F1 Score"
        title = "Classification Model Comparison: Accuracy vs F1 Score"

    fig = px.scatter(
        results_df,
        x=x_metric,
        y=y_metric,
        text="Model",
        title=title,
    )

    fig.update_traces(
        marker=dict(size=14),
        textposition="top center",
    )

    st.plotly_chart(fig, width="stretch")


def _create_heatmap(results_df: pd.DataFrame, task_type: str) -> None:
    if task_type == "Regression":
        metric_columns = ["R2 Score", "MAE", "MSE", "RMSE"]
    else:
        metric_columns = ["Accuracy", "Precision", "Recall", "F1 Score"]

    heatmap_data = results_df.set_index("Model")[metric_columns]

    fig = px.imshow(
        heatmap_data,
        aspect="auto",
        title="Model Performance Heatmap",
    )

    st.plotly_chart(fig, width="stretch")


def render_ml_recommendations(df: pd.DataFrame) -> None:
    st.subheader("Machine Learning Recommendation System")

    if df is None:
        st.warning("No dataset found. Please upload a dataset first.")
        return

    if df.empty:
        st.warning("The uploaded dataset is empty.")
        return

    if df.shape[1] < 2:
        st.warning("The dataset needs at least two columns for machine learning.")
        return

    df = _auto_convert_numeric_text_columns(df)

    st.info(
        "This section recommends target columns, detects regression or classification, "
        "trains basic ML models, and compares their performance."
    )

    numeric_count = len(df.select_dtypes(include=["number"]).columns)
    text_count = len(df.select_dtypes(include=["object", "string", "category"]).columns)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Numeric Columns", numeric_count)

    with col3:
        st.metric("Text Columns", text_count)

    target_recommendations = _get_target_recommendation(df)

    if target_recommendations.empty:
        st.warning(
            "No strong target column was found automatically. "
            "This usually means your dataset has mostly text columns with too many unique values."
        )

        manual_target = st.selectbox(
            "Select a target column manually",
            df.columns.tolist(),
        )

        manual_unique_count = df[manual_target].nunique(dropna=True)

        if pd.api.types.is_numeric_dtype(df[manual_target]):
            target_recommendations = pd.DataFrame(
                [
                    {
                        "Column": manual_target,
                        "Recommended Task": "Regression",
                        "Unique Values": manual_unique_count,
                        "Missing Values": df[manual_target].isnull().sum(),
                        "Data Type": str(df[manual_target].dtype),
                        "Reason": "Manual numeric target selection",
                    }
                ]
            )
        elif manual_unique_count <= 200:
            target_recommendations = pd.DataFrame(
                [
                    {
                        "Column": manual_target,
                        "Recommended Task": "Classification",
                        "Unique Values": manual_unique_count,
                        "Missing Values": df[manual_target].isnull().sum(),
                        "Data Type": str(df[manual_target].dtype),
                        "Reason": "Manual categorical target selection",
                    }
                ]
            )
        else:
            st.error(
                "This target column has too many unique text values for classification. "
                "Please select a column with fewer repeated categories."
            )
            return

    st.markdown("### Recommended Target Columns")
    st.dataframe(target_recommendations, width="stretch")

    target_column = st.selectbox(
        "Select target column",
        target_recommendations["Column"].tolist(),
    )

    selected_recommendation = target_recommendations[
        target_recommendations["Column"] == target_column
    ].iloc[0]

    default_task = selected_recommendation["Recommended Task"]

    if pd.api.types.is_numeric_dtype(df[target_column]):
        task_options = ["Regression", "Classification"]
    else:
        task_options = ["Classification"]

    task_type = st.selectbox(
        "Select ML task type",
        task_options,
        index=task_options.index(default_task) if default_task in task_options else 0,
    )

    unique_target_values = df[target_column].nunique(dropna=True)

    if task_type == "Classification" and unique_target_values > 200:
        st.error(
            "The selected target has too many classes for classification. "
            "Choose another target column."
        )
        return

    available_features = [column for column in df.columns if column != target_column]

    selected_features = st.multiselect(
        "Select feature columns",
        available_features,
        default=available_features,
    )

    if not selected_features:
        st.warning("Please select at least one feature column.")
        return

    data = df[selected_features + [target_column]].copy()
    data = data.dropna(subset=[target_column])

    if data.empty:
        st.warning("No valid rows are available after removing missing target values.")
        return

    X = data[selected_features]
    y = data[target_column]

    usable_features = [
        column for column in X.columns
        if not X[column].isnull().all()
    ]

    if not usable_features:
        st.warning("No usable feature columns found.")
        return

    X = X[usable_features]
    X = _prepare_feature_dataframe(X)

    if task_type == "Regression":
        y = pd.to_numeric(y, errors="coerce")
        valid_rows = y.notna()

        X = X.loc[valid_rows]
        y = y.loc[valid_rows]

        if y.nunique() < 2:
            st.warning("The selected target does not have enough variation for regression.")
            return

    else:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y.astype(str))

        if len(np.unique(y)) < 2:
            st.warning("The selected target needs at least two classes.")
            return

    if len(X) < 10:
        st.warning("The dataset is too small for a reliable train/test split.")
        return

    try:
        if task_type == "Classification":
            class_counts = pd.Series(y).value_counts()
            number_of_classes = len(class_counts)
            test_count = max(1, int(len(y) * 0.2))

            if class_counts.min() >= 2 and number_of_classes <= test_count:
                stratify_value = y
            else:
                stratify_value = None

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=stratify_value,
            )

        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
            )

    except ValueError as error:
        st.error(f"Could not split the dataset: {error}")
        return

    preprocessor = _build_preprocessor(X)
    models = _get_models(task_type)

    st.markdown("### Model Training Results")

    results = []

    with st.spinner("Training models..."):
        for model_name, model in models.items():
            try:
                pipeline = Pipeline(
                    steps=[
                        ("preprocessor", preprocessor),
                        ("model", model),
                    ]
                )

                pipeline.fit(X_train, y_train)
                predictions = pipeline.predict(X_test)

                if task_type == "Regression":
                    mse = mean_squared_error(y_test, predictions)
                    rmse = np.sqrt(mse)

                    results.append(
                        {
                            "Model": model_name,
                            "R2 Score": r2_score(y_test, predictions),
                            "MAE": mean_absolute_error(y_test, predictions),
                            "MSE": mse,
                            "RMSE": rmse,
                        }
                    )

                else:
                    results.append(
                        {
                            "Model": model_name,
                            "Accuracy": accuracy_score(y_test, predictions),
                            "Precision": precision_score(
                                y_test,
                                predictions,
                                average="weighted",
                                zero_division=0,
                            ),
                            "Recall": recall_score(
                                y_test,
                                predictions,
                                average="weighted",
                                zero_division=0,
                            ),
                            "F1 Score": f1_score(
                                y_test,
                                predictions,
                                average="weighted",
                                zero_division=0,
                            ),
                        }
                    )

            except Exception as error:
                st.warning(f"{model_name} could not be trained: {error}")

    if not results:
        st.error("No models could be trained on this dataset.")
        return

    results_df = pd.DataFrame(results)

    numeric_result_columns = results_df.select_dtypes(include=["number"]).columns

    for column in numeric_result_columns:
        results_df[column] = results_df[column].round(4)

    st.dataframe(results_df, width="stretch")

    st.markdown("### Best Model Recommendation")

    if task_type == "Regression":
        best_model_row = results_df.sort_values(
            by="R2 Score",
            ascending=False,
        ).iloc[0]

        st.success(
            f"Best model: {best_model_row['Model']} "
            f"with R2 Score = {best_model_row['R2 Score']}"
        )

    else:
        best_model_row = results_df.sort_values(
            by="Accuracy",
            ascending=False,
        ).iloc[0]

        st.success(
            f"Best model: {best_model_row['Model']} "
            f"with Accuracy = {best_model_row['Accuracy']}"
        )

    st.markdown("### Model Comparison Charts")

    chart_choice = st.selectbox(
        "Choose comparison chart",
        [
            "Bar Chart",
            "Scatter Plot",
            "Heatmap",
            "All Charts",
        ],
    )

    if chart_choice == "Bar Chart":
        _create_bar_chart(results_df, task_type)

    elif chart_choice == "Scatter Plot":
        _create_scatter_chart(results_df, task_type)

    elif chart_choice == "Heatmap":
        _create_heatmap(results_df, task_type)

    elif chart_choice == "All Charts":
        _create_bar_chart(results_df, task_type)
        _create_scatter_chart(results_df, task_type)
        _create_heatmap(results_df, task_type)