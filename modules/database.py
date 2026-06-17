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
    DB_FOLDER.mkdir(exist_ok=True)
    return sqlite3.connect(DB_PATH)


def hash_password(password: str) -> str:
    """Hash password for basic MVP authentication."""
    return hashlib.sha256(password.encode()).hexdigest()


def init_db() -> None:
    """
    Create required database tables.

    Also updates old Day 6 database tables if new columns are missing.
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
            saved_at TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """
    )

    cursor.execute("PRAGMA table_info(dataset_history)")
    existing_columns = [column[1] for column in cursor.fetchall()]

    if "user_id" not in existing_columns:
        cursor.execute(
            """
            ALTER TABLE dataset_history
            ADD COLUMN user_id INTEGER
            """
        )

    if "file_size_kb" not in existing_columns:
        cursor.execute(
            """
            ALTER TABLE dataset_history
            ADD COLUMN file_size_kb REAL
            """
        )

    if "file_signature" not in existing_columns:
        cursor.execute(
            """
            ALTER TABLE dataset_history
            ADD COLUMN file_signature TEXT
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
    user_id: int,
    file_name: str,
    file_type: str,
    file_size_kb: float,
    file_signature: str,
    total_rows: int,
    total_columns: int,
    missing_values: int,
    duplicate_rows: int,
    numeric_columns: int,
    categorical_columns: int,
):
    """Save dataset summary for a specific logged-in user."""
    init_db()

    if dataset_already_saved(
        user_id=user_id,
        file_signature=file_signature
    ):
        return False, "This dataset is already saved in your history."

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO dataset_history (
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
            saved_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
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
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ),
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
        SELECT
            id,
            file_name,
            file_type,
            file_size_kb,
            total_rows,
            total_columns,
            missing_values,
            duplicate_rows,
            numeric_columns,
            categorical_columns,
            saved_at
        FROM dataset_history
        WHERE user_id = ?
        ORDER BY id DESC
        """,
        conn,
        params=(user_id,),
    )

    conn.close()
    return history_df


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