import pandas as pd


def convert_possible_numeric_columns(df):
    """
    Converts text columns that look numeric into actual numeric columns.
    Example:
    ₹999 -> 999
    64% -> 64
    1,234 -> 1234
    """
    cleaned_df = df.copy()

    for column in cleaned_df.columns:
        if cleaned_df[column].dtype == "object":
            temp_series = (
                cleaned_df[column]
                .astype(str)
                .str.replace("₹", "", regex=False)
                .str.replace(",", "", regex=False)
                .str.replace("%", "", regex=False)
                .str.strip()
            )

            numeric_series = pd.to_numeric(temp_series, errors="coerce")

            original_non_null = cleaned_df[column].notnull().sum()
            converted_non_null = numeric_series.notnull().sum()

            if original_non_null > 0:
                conversion_ratio = converted_non_null / original_non_null

                if conversion_ratio >= 0.8:
                    cleaned_df[column] = numeric_series

    return cleaned_df


def remove_duplicates(df):
    """
    Removes duplicate rows from the dataset.
    """
    return df.drop_duplicates()


def fill_numeric_missing(df, method="mean"):
    """
    Converts numeric-looking text columns into numbers,
    then fills missing values in numeric columns.
    Example:
    24,269 -> 24269
    ₹999 -> 999
    80% -> 80
    """
    cleaned_df = df.copy()

    for column in cleaned_df.columns:
        if cleaned_df[column].dtype == "object":
            temp_series = (
                cleaned_df[column]
                .astype(str)
                .str.replace("₹", "", regex=False)
                .str.replace(",", "", regex=False)
                .str.replace("%", "", regex=False)
                .str.strip()
            )

            converted_series = pd.to_numeric(temp_series, errors="coerce")

            non_missing_original = cleaned_df[column].notna().sum()
            non_missing_converted = converted_series.notna().sum()

            if non_missing_original > 0:
                conversion_ratio = non_missing_converted / non_missing_original

                if conversion_ratio >= 0.8:
                    cleaned_df[column] = converted_series

    numeric_columns = cleaned_df.select_dtypes(include=["number"]).columns

    for column in numeric_columns:
        if cleaned_df[column].isnull().sum() > 0:
            if method == "mean":
                fill_value = cleaned_df[column].mean()

            elif method == "median":
                fill_value = cleaned_df[column].median()

            elif method == "zero":
                fill_value = 0

            cleaned_df[column] = cleaned_df[column].fillna(fill_value)

    return cleaned_df


def fill_text_missing(df, value="Unknown"):
    """
    Fills missing values in text columns.
    """
    cleaned_df = df.copy()

    text_columns = cleaned_df.select_dtypes(include=["object"]).columns

    for column in text_columns:
        cleaned_df[column] = cleaned_df[column].fillna(value)

    return cleaned_df


def drop_missing_rows(df):
    """
    Drops all rows that contain missing values.
    """
    return df.dropna()


def get_csv_download(df):
    """
    Converts dataframe to CSV for download.
    """
    return df.to_csv(index=False).encode("utf-8")