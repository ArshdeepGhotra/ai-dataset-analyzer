import sqlite3
import hashlib
from pathlib import Path
from datetime import datetime

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DB_FOLDER = BASE_DIR / "database"
DB_PATH = DB_FOLDER / "dataset_history.db"


def get_connection():
    """Create database folder and return SQLite connection."""
    DB_FOLDER.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def hash_password(password: str) -> str:
    """Hash password for basic MVP authentication."""
    return hashlib.sha256(password.encode()).hexdigest()


def get_table_columns(cursor, table_name: str) -> list:
    """Return column names for a database table."""
    cursor.execute(f"PRAGMA table_info({table_name})")
    return [column[1] for column in cursor.fetchall()]


def add_column_if_missing(
    cursor,
    existing_columns: list,
    column_name: str,
    column_type: str
) -> None:
    """Add a column to a table if it does not already exist."""
    if column_name not in existing_columns:
        cursor.execute(
            f"""
            ALTER TABLE dataset_history
            ADD COLUMN {column_name} {column_type}
            """
        )
        existing_columns.append(column_name)


def init_db() -> None:
    """
    Create required database tables.

    Also updates older database tables if new columns are missing.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS dataset_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            file_name TEXT,
            file_type TEXT,
            file_size_kb REAL,
            file_signature TEXT,
            total_rows INTEGER,
            total_columns INTEGER,
            missing_values INTEGER,
            duplicate_rows INTEGER,
            numeric_columns INTEGER,
            categorical_columns INTEGER,
            dataset_csv TEXT,
            saved_at TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """
    )

    existing_columns = get_table_columns(cursor, "dataset_history")

    add_column_if_missing(cursor, existing_columns, "user_id", "INTEGER")
    add_column_if_missing(cursor, existing_columns, "file_name", "TEXT")
    add_column_if_missing(cursor, existing_columns, "file_type", "TEXT")
    add_column_if_missing(cursor, existing_columns, "file_size_kb", "REAL")
    add_column_if_missing(cursor, existing_columns, "file_signature", "TEXT")
    add_column_if_missing(cursor, existing_columns, "total_rows", "INTEGER")
    add_column_if_missing(cursor, existing_columns, "total_columns", "INTEGER")
    add_column_if_missing(cursor, existing_columns, "missing_values", "INTEGER")
    add_column_if_missing(cursor, existing_columns, "duplicate_rows", "INTEGER")
    add_column_if_missing(cursor, existing_columns, "numeric_columns", "INTEGER")
    add_column_if_missing(cursor, existing_columns, "categorical_columns", "INTEGER")
    add_column_if_missing(cursor, existing_columns, "dataset_csv", "TEXT")
    add_column_if_missing(cursor, existing_columns, "saved_at", "TEXT")

    conn.commit()
    conn.close()


def add_dataset_csv_column_if_missing():
    """Add dataset_csv column to dataset_history if it does not already exist."""
    conn = get_connection()
    cursor = conn.cursor()

    existing_columns = get_table_columns(cursor, "dataset_history")

    if "dataset_csv" not in existing_columns:
        cursor.execute(
            """
            ALTER TABLE dataset_history
            ADD COLUMN dataset_csv TEXT
            """
        )

    conn.commit()
    conn.close()


def create_user(name: str, email: str, password: str):
    """Create a new user account."""
    init_db()

    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
            INSERT INTO users (
                name,
                email,
                password,
                created_at
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                name.strip(),
                email.lower().strip(),
                hash_password(password),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )

        conn.commit()
        return True, "Account created successfully."

    except sqlite3.IntegrityError:
        return False, "An account with this email already exists."

    finally:
        conn.close()


def login_user(email: str, password: str):
    """Login user using email and password."""
    init_db()

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT id, name, email
        FROM users
        WHERE email = ? AND password = ?
        """,
        (
            email.lower().strip(),
            hash_password(password),
        ),
    )

    user = cursor.fetchone()
    conn.close()

    if user:
        return {
            "id": user[0],
            "name": user[1],
            "email": user[2],
        }

    return None


def dataset_already_saved(
    user_id: int,
    file_signature: str
) -> bool:
    """Check if the same user already saved this exact file."""
    init_db()

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT id
        FROM dataset_history
        WHERE user_id = ? AND file_signature = ?
        LIMIT 1
        """,
        (
            user_id,
            file_signature,
        ),
    )

    result = cursor.fetchone()
    conn.close()

    return result is not None


def save_dataset_summary(
    user_id,
    file_name,
    file_type,
    file_size_kb,
    file_signature,
    total_rows,
    total_columns,
    missing_values,
    duplicate_rows,
    numeric_columns,
    categorical_columns,
    dataset_csv=None,
):
    """Save dataset summary and CSV data for a specific logged-in user."""
    init_db()
    add_dataset_csv_column_if_missing()

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT id, dataset_csv
        FROM dataset_history
        WHERE user_id = ? AND file_signature = ?
        LIMIT 1
        """,
        (
            user_id,
            file_signature,
        ),
    )

    existing_record = cursor.fetchone()

    if existing_record:
        record_id = existing_record[0]
        existing_dataset_csv = existing_record[1]

        if dataset_csv and not existing_dataset_csv:
            cursor.execute(
                """
                UPDATE dataset_history
                SET
                    dataset_csv = ?,
                    file_name = ?,
                    file_type = ?,
                    file_size_kb = ?,
                    total_rows = ?,
                    total_columns = ?,
                    missing_values = ?,
                    duplicate_rows = ?,
                    numeric_columns = ?,
                    categorical_columns = ?
                WHERE id = ? AND user_id = ?
                """,
                (
                    dataset_csv,
                    file_name,
                    file_type,
                    file_size_kb,
                    total_rows,
                    total_columns,
                    missing_values,
                    duplicate_rows,
                    numeric_columns,
                    categorical_columns,
                    record_id,
                    user_id,
                ),
            )

            conn.commit()
            conn.close()

            return True, "Existing history record updated with dataset data."

        conn.close()
        return False, "This dataset is already saved in your history."

    existing_columns = get_table_columns(cursor, "dataset_history")

    insert_data = {
        "user_id": user_id,
        "file_name": file_name,
        "file_type": file_type,
        "file_size_kb": file_size_kb,
        "file_signature": file_signature,
        "total_rows": total_rows,
        "total_columns": total_columns,
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "dataset_csv": dataset_csv,
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Support older table columns if they exist from previous Day 8 code.
    if "filename" in existing_columns:
        insert_data["filename"] = file_name

    if "rows_count" in existing_columns:
        insert_data["rows_count"] = total_rows

    if "columns_count" in existing_columns:
        insert_data["columns_count"] = total_columns

    if "text_columns" in existing_columns:
        insert_data["text_columns"] = categorical_columns

    valid_columns = [
        column for column in insert_data.keys()
        if column in existing_columns
    ]

    placeholders = ", ".join(["?"] * len(valid_columns))
    column_names = ", ".join(valid_columns)

    values = [
        insert_data[column]
        for column in valid_columns
    ]

    cursor.execute(
        f"""
        INSERT INTO dataset_history (
            {column_names}
        )
        VALUES (
            {placeholders}
        )
        """,
        values,
    )

    conn.commit()
    conn.close()

    return True, "Dataset summary saved automatically."


def get_dataset_history(user_id: int) -> pd.DataFrame:
    """Return dataset history for the logged-in user only."""
    init_db()

    conn = get_connection()

    history_df = pd.read_sql_query(
        """
        SELECT *
        FROM dataset_history
        WHERE user_id = ?
        ORDER BY id DESC
        """,
        conn,
        params=(user_id,),
    )

    conn.close()

    if history_df.empty:
        return history_df

    # Support old column names from previous history table versions.
    if "file_name" not in history_df.columns and "filename" in history_df.columns:
        history_df["file_name"] = history_df["filename"]

    if "total_rows" not in history_df.columns and "rows_count" in history_df.columns:
        history_df["total_rows"] = history_df["rows_count"]

    if "total_columns" not in history_df.columns and "columns_count" in history_df.columns:
        history_df["total_columns"] = history_df["columns_count"]

    if (
        "categorical_columns" not in history_df.columns
        and "text_columns" in history_df.columns
    ):
        history_df["categorical_columns"] = history_df["text_columns"]

    display_columns = [
        "id",
        "file_name",
        "file_type",
        "file_size_kb",
        "total_rows",
        "total_columns",
        "missing_values",
        "duplicate_rows",
        "numeric_columns",
        "categorical_columns",
        "saved_at",
    ]

    available_columns = [
        column for column in display_columns
        if column in history_df.columns
    ]

    return history_df[available_columns]


def get_saved_dataset_by_id(user_id: int, record_id: int):
    """Get saved dataset CSV content by history record ID."""
    init_db()
    add_dataset_csv_column_if_missing()

    conn = get_connection()

    result_df = pd.read_sql_query(
        """
        SELECT *
        FROM dataset_history
        WHERE user_id = ? AND id = ?
        LIMIT 1
        """,
        conn,
        params=(user_id, record_id),
    )

    conn.close()

    if result_df.empty:
        return None

    record = result_df.iloc[0].to_dict()

    if not record.get("file_name") and record.get("filename"):
        record["file_name"] = record["filename"]

    return {
        "id": record.get("id"),
        "file_name": record.get("file_name"),
        "dataset_csv": record.get("dataset_csv"),
    }


def clear_dataset_history(user_id: int) -> None:
    """Clear dataset history for the logged-in user only."""
    init_db()

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        DELETE FROM dataset_history
        WHERE user_id = ?
        """,
        (user_id,),
    )

    conn.commit()
    conn.close()


def delete_dataset_history_record(
    user_id: int,
    record_id: int
) -> None:
    """Delete one selected dataset history record."""
    init_db()

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        DELETE FROM dataset_history
        WHERE user_id = ? AND id = ?
        """,
        (
            user_id,
            record_id,
        ),
    )

    conn.commit()
    conn.close()