import io

import pandas as pd
import streamlit as st

from modules.database import (
    get_dataset_history,
    clear_dataset_history,
    delete_dataset_history_record,
    get_saved_dataset_by_id,
)


def render_history_dashboard(user_id: int) -> None:
    """Display saved dataset upload history for the logged-in user."""

    st.subheader("Saved Dataset History")

    history_df = get_dataset_history(user_id)

    if history_df.empty:
        st.info("No dataset history saved yet.")
        return

    stat1, stat2, stat3, stat4 = st.columns(4)

    with stat1:
        st.metric(
            "Total Uploads",
            int(len(history_df))
        )

    with stat2:
        st.metric(
            "Rows Analyzed",
            int(history_df["total_rows"].sum())
        )

    with stat3:
        st.metric(
            "Missing Values Found",
            int(history_df["missing_values"].sum())
        )

    with stat4:
        st.metric(
            "Duplicate Rows Found",
            int(history_df["duplicate_rows"].sum())
        )

    st.divider()

    display_df = history_df.copy()

    if "dataset_csv" in display_df.columns:
        display_df = display_df.drop(columns=["dataset_csv"])

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    csv_data = display_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download History as CSV",
        data=csv_data,
        file_name="dataset_history.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.divider()
    st.subheader("Work on a Saved Dataset Again")

    record_options = {
        f"{row['id']} | {row['file_name']} | {row['saved_at']}": row["id"]
        for _, row in history_df.iterrows()
    }

    selected_record_label = st.selectbox(
        "Select a saved dataset record:",
        list(record_options.keys())
    )

    selected_record_id = int(record_options[selected_record_label])

    if st.button("Load Selected Dataset"):
        saved_record = get_saved_dataset_by_id(
            user_id=user_id,
            record_id=selected_record_id
        )

        if not saved_record or not saved_record.get("dataset_csv"):
            st.error(
                "This old record does not have the actual dataset saved. "
                "Upload this dataset one more time so the app can save "
                "the full dataset file for future use."
            )
            return

        loaded_df = pd.read_csv(
            io.StringIO(saved_record["dataset_csv"])
        )

        st.session_state["loaded_history_df"] = loaded_df
        st.session_state["loaded_history_file_name"] = saved_record["file_name"]

        st.success(
            f"Loaded {saved_record['file_name']} successfully. "
            "Go to Dashboard to continue working on it."
        )

    st.divider()
    st.subheader("Manage History")

    delete_col, clear_col = st.columns(2)

    with delete_col:
        if st.button("Delete Selected Record"):
            delete_dataset_history_record(
                user_id=user_id,
                record_id=selected_record_id
            )

            st.success("Selected history record deleted.")
            st.rerun()

    with clear_col:
        st.write("Clear all saved history for your account.")

        if st.button("Clear My Dataset History"):
            clear_dataset_history(user_id)

            st.success("Your dataset history was cleared.")
            st.rerun()