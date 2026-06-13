import pandas as pd
import plotly.express as px
import streamlit as st


def prepare_chart_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare a copy of the dataset for visualizations.

    This converts numeric-looking text columns into real numbers
    and date-looking text columns into datetime columns.
    """

    chart_df = df.copy()

    for column in chart_df.columns:
        if chart_df[column].dtype == "object":
            original_values = chart_df[column].notna().sum()

            if original_values == 0:
                continue

            cleaned_series = (
                chart_df[column]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("$", "", regex=False)
                .str.replace("%", "", regex=False)
                .str.strip()
            )

            numeric_series = pd.to_numeric(
                cleaned_series,
                errors="coerce"
            )

            numeric_values = numeric_series.notna().sum()

            if numeric_values / original_values >= 0.70:
                chart_df[column] = numeric_series
                continue

            datetime_series = pd.to_datetime(
                chart_df[column],
                errors="coerce"
            )

            datetime_values = datetime_series.notna().sum()

            if datetime_values / original_values >= 0.70:
                chart_df[column] = datetime_series

    return chart_df


def render_visualizations(df: pd.DataFrame) -> None:
    """Display interactive charts for the uploaded dataset."""

    st.subheader("Interactive Data Visualizations")

    if df is None or df.empty:
        st.warning("The dataset is empty.")
        return

    chart_df = prepare_chart_dataframe(df)

    numeric_columns = chart_df.select_dtypes(
        include="number"
    ).columns.tolist()

    datetime_columns = chart_df.select_dtypes(
        include=["datetime64[ns]", "datetime64[ns, UTC]"]
    ).columns.tolist()

    categorical_columns = [
        column
        for column in chart_df.columns
        if column not in numeric_columns
        and column not in datetime_columns
    ]

    metric1, metric2, metric3 = st.columns(3)

    with metric1:
        st.metric("Numeric Columns", len(numeric_columns))

    with metric2:
        st.metric("Categorical Columns", len(categorical_columns))

    with metric3:
        st.metric("Date Columns", len(datetime_columns))

    with st.expander("Detected Column Types"):
        st.write("Numeric columns:", numeric_columns)
        st.write("Categorical columns:", categorical_columns)
        st.write("Date columns:", datetime_columns)

    chart_options = []

    if numeric_columns:
        chart_options.extend(
            [
                "Histogram",
                "Box Plot",
                "Line Chart"
            ]
        )

    if categorical_columns:
        chart_options.extend(
            [
                "Bar Chart",
                "Pie Chart"
            ]
        )

    if len(numeric_columns) >= 2:
        chart_options.extend(
            [
                "Scatter Plot",
                "Correlation Heatmap"
            ]
        )

    if not chart_options:
        st.warning(
            "No suitable columns were found for visualization."
        )
        return

    chart_type = st.selectbox(
        "Choose a chart",
        chart_options,
        key="chart_type_selector"
    )

    if chart_type == "Histogram":
        display_histogram(chart_df, numeric_columns)

    elif chart_type == "Bar Chart":
        display_bar_chart(chart_df, categorical_columns)

    elif chart_type == "Pie Chart":
        display_pie_chart(chart_df, categorical_columns)

    elif chart_type == "Box Plot":
        display_box_plot(
            chart_df,
            numeric_columns,
            categorical_columns
        )

    elif chart_type == "Line Chart":
        display_line_chart(
            chart_df,
            numeric_columns,
            datetime_columns
        )

    elif chart_type == "Scatter Plot":
        display_scatter_plot(
            chart_df,
            numeric_columns,
            categorical_columns
        )

    elif chart_type == "Correlation Heatmap":
        display_correlation_heatmap(
            chart_df,
            numeric_columns
        )


def display_histogram(
    chart_df: pd.DataFrame,
    numeric_columns: list
) -> None:
    selected_column = st.selectbox(
        "Select a numeric column",
        numeric_columns,
        key="histogram_column"
    )

    bins = st.slider(
        "Number of bins",
        min_value=5,
        max_value=100,
        value=30,
        key="histogram_bins"
    )

    chart_data = chart_df[
        [selected_column]
    ].dropna()

    if chart_data.empty:
        st.warning("The selected column contains no usable values.")
        return

    figure = px.histogram(
        chart_data,
        x=selected_column,
        nbins=bins,
        title=f"Distribution of {selected_column}"
    )

    figure.update_layout(
        xaxis_title=selected_column,
        yaxis_title="Frequency"
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
        key="histogram_figure"
    )


def display_bar_chart(
    chart_df: pd.DataFrame,
    categorical_columns: list
) -> None:
    selected_column = st.selectbox(
        "Select a categorical column",
        categorical_columns,
        key="bar_column"
    )

    category_series = (
        chart_df[selected_column]
        .astype("string")
        .fillna("Missing")
    )

    max_categories = min(
        30,
        int(category_series.nunique())
    )

    top_categories = st.slider(
        "Number of categories to show",
        min_value=1,
        max_value=max_categories,
        value=min(10, max_categories),
        key="bar_top_categories"
    )

    category_counts = (
        category_series
        .value_counts()
        .head(top_categories)
        .rename_axis(selected_column)
        .reset_index(name="Count")
    )

    figure = px.bar(
        category_counts,
        x=selected_column,
        y="Count",
        title=f"Category Counts for {selected_column}"
    )

    figure.update_layout(
        xaxis_title=selected_column,
        yaxis_title="Count"
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
        key="bar_figure"
    )


def display_pie_chart(
    chart_df: pd.DataFrame,
    categorical_columns: list
) -> None:
    selected_column = st.selectbox(
        "Select a categorical column",
        categorical_columns,
        key="pie_column"
    )

    category_series = (
        chart_df[selected_column]
        .astype("string")
        .fillna("Missing")
    )

    max_categories = min(
        15,
        int(category_series.nunique())
    )

    top_categories = st.slider(
        "Number of slices to show",
        min_value=1,
        max_value=max_categories,
        value=min(8, max_categories),
        key="pie_top_categories"
    )

    pie_data = (
        category_series
        .value_counts()
        .head(top_categories)
        .rename_axis(selected_column)
        .reset_index(name="Count")
    )

    figure = px.pie(
        pie_data,
        names=selected_column,
        values="Count",
        title=f"Category Share for {selected_column}"
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
        key="pie_figure"
    )


def display_box_plot(
    chart_df: pd.DataFrame,
    numeric_columns: list,
    categorical_columns: list
) -> None:
    selected_numeric_column = st.selectbox(
        "Select a numeric column",
        numeric_columns,
        key="box_numeric_column"
    )

    group_options = ["No Grouping"] + categorical_columns

    selected_group = st.selectbox(
        "Group by category",
        group_options,
        key="box_group_column"
    )

    if selected_group == "No Grouping":
        chart_data = chart_df[
            [selected_numeric_column]
        ].dropna()

        figure = px.box(
            chart_data,
            y=selected_numeric_column,
            points="outliers",
            title=f"Box Plot of {selected_numeric_column}"
        )

    else:
        top_groups = (
            chart_df[selected_group]
            .astype("string")
            .value_counts()
            .head(15)
            .index
        )

        chart_data = chart_df[
            chart_df[selected_group]
            .astype("string")
            .isin(top_groups)
        ][
            [selected_group, selected_numeric_column]
        ].dropna()

        figure = px.box(
            chart_data,
            x=selected_group,
            y=selected_numeric_column,
            points="outliers",
            title=(
                f"{selected_numeric_column} "
                f"Grouped by {selected_group}"
            )
        )

    if chart_data.empty:
        st.warning("The selected columns contain no usable values.")
        return

    st.plotly_chart(
        figure,
        use_container_width=True,
        key="box_figure"
    )


def display_line_chart(
    chart_df: pd.DataFrame,
    numeric_columns: list,
    datetime_columns: list
) -> None:
    x_axis_options = (
        ["Row Number"]
        + datetime_columns
        + numeric_columns
    )

    column1, column2 = st.columns(2)

    with column1:
        x_column = st.selectbox(
            "Select the X-axis",
            x_axis_options,
            key="line_x_column"
        )

    with column2:
        y_column = st.selectbox(
            "Select the Y-axis",
            numeric_columns,
            key="line_y_column"
        )

    if x_column == y_column:
        st.warning(
            "Choose different columns for the X-axis and Y-axis."
        )
        return

    if x_column == "Row Number":
        chart_data = (
            chart_df[[y_column]]
            .dropna()
            .reset_index(drop=True)
        )

        chart_data["Row Number"] = range(
            1,
            len(chart_data) + 1
        )

    else:
        chart_data = chart_df[
            [x_column, y_column]
        ].dropna()

        chart_data = chart_data.sort_values(
            by=x_column
        )

    if chart_data.empty:
        st.warning("The selected columns contain no usable values.")
        return

    figure = px.line(
        chart_data,
        x=x_column,
        y=y_column,
        markers=True,
        title=f"{y_column} Across {x_column}"
    )

    figure.update_layout(
        xaxis_title=x_column,
        yaxis_title=y_column
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
        key="line_figure"
    )


def display_scatter_plot(
    chart_df: pd.DataFrame,
    numeric_columns: list,
    categorical_columns: list
) -> None:
    column1, column2 = st.columns(2)

    with column1:
        x_column = st.selectbox(
            "Select the X-axis",
            numeric_columns,
            key="scatter_x_column"
        )

    with column2:
        y_column = st.selectbox(
            "Select the Y-axis",
            numeric_columns,
            index=1,
            key="scatter_y_column"
        )

    if x_column == y_column:
        st.warning(
            "Choose different columns for the X-axis and Y-axis."
        )
        return

    color_options = ["No Color Grouping"] + categorical_columns

    color_column = st.selectbox(
        "Color by category",
        color_options,
        key="scatter_color_column"
    )

    selected_columns = [x_column, y_column]

    if color_column != "No Color Grouping":
        selected_columns.append(color_column)

    chart_data = chart_df[
        selected_columns
    ].dropna(
        subset=[x_column, y_column]
    )

    if chart_data.empty:
        st.warning("The selected columns contain no usable values.")
        return

    selected_color = None

    if color_column != "No Color Grouping":
        selected_color = color_column

    figure = px.scatter(
        chart_data,
        x=x_column,
        y=y_column,
        color=selected_color,
        opacity=0.75,
        title=f"{y_column} Compared with {x_column}"
    )

    figure.update_layout(
        xaxis_title=x_column,
        yaxis_title=y_column
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
        key="scatter_figure"
    )


def display_correlation_heatmap(
    chart_df: pd.DataFrame,
    numeric_columns: list
) -> None:
    selected_columns = st.multiselect(
        "Select at least two numeric columns",
        numeric_columns,
        default=numeric_columns[:10],
        key="heatmap_columns"
    )

    if len(selected_columns) < 2:
        st.warning("Select at least two numeric columns.")
        return

    correlation_matrix = (
        chart_df[selected_columns]
        .corr(numeric_only=True)
        .dropna(how="all")
        .dropna(axis=1, how="all")
    )

    if correlation_matrix.shape[0] < 2:
        st.warning(
            "Correlation could not be calculated for these columns."
        )
        return

    figure = px.imshow(
        correlation_matrix,
        text_auto=".2f",
        aspect="auto",
        title="Correlation Heatmap",
        labels={
            "color": "Correlation"
        }
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
        key="heatmap_figure"
    )