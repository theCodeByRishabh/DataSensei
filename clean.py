import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
from sklearn.impute import KNNImputer
import numpy as np
import os
from io import BytesIO


st.set_page_config(
    page_title="DataSensei - Data Cleaning",
    page_icon="😎",
    initial_sidebar_state="collapsed"
)

with open('header.html', 'r') as file:
        header_html_content = file.read()
st.markdown(header_html_content, unsafe_allow_html=True)


nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)



def step_trim_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data = data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    return data


def step_text_clean_stem(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()

    def clean_and_stem(value):
        if isinstance(value, str):
            tokens = word_tokenize(value)
            tokens = [t.lower() for t in tokens]
            tokens = [t for t in tokens if t not in string.punctuation]
            tokens = [t for t in tokens if t not in stop_words]
            tokens = [stemmer.stem(t) for t in tokens]
            return " ".join(tokens)
        return value

    data = data.applymap(clean_and_stem)
    return data


def step_handle_missing_mode_mean(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    object_cols = data.select_dtypes(include=["object"]).columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns

    for col in object_cols:
        if data[col].isna().any():
            mode = data[col].mode()
            if not mode.empty:
                data[col].fillna(mode.iloc[0], inplace=True)

    for col in numeric_cols:
        if data[col].isna().any():
            mean_val = data[col].mean()
            if pd.notna(mean_val):
                data[col].fillna(mean_val, inplace=True)

    return data


def step_knn_impute_numeric(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    numeric_cols = data.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) > 0 and data[numeric_cols].isna().any().any():
        imputer = KNNImputer()
        imputed_array = imputer.fit_transform(data[numeric_cols])
        imputed_df = pd.DataFrame(imputed_array, columns=numeric_cols, index=data.index)
        data[numeric_cols] = imputed_df

    return data


def step_drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data = data.drop_duplicates()
    return data


def step_infer_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data = data.infer_objects()
    return data


def step_remove_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        Q1 = data[numeric_cols].quantile(0.25)
        Q3 = data[numeric_cols].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outlier_mask = ((data[numeric_cols] < lower_bound) | (data[numeric_cols] > upper_bound)).any(axis=1)
        data = data[~outlier_mask]

    return data


CLEANING_STEPS = {
    "Remove whitespace (text columns)": step_trim_whitespace,
    "Text cleaning (tokenize, punctuation/stopwords removal, stemming)": step_text_clean_stem,
    "Handle missing values (mode for text, mean for numeric)": step_handle_missing_mode_mean,
    "KNN imputation for numeric missing values": step_knn_impute_numeric,
    "Remove duplicate rows": step_drop_duplicates,
    "Infer / fix data types": step_infer_dtypes,
    "Remove outliers (IQR, numeric columns)": step_remove_outliers_iqr,
}


st.title("DataSensei - Smart Data Cleaning🧹  ")

uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()


    if file_extension == ".csv":
        try:
            uploaded_file.seek(0)
            data = pd.read_csv(uploaded_file, encoding="utf-8")
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            data = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
    elif file_extension in [".xlsx", ".xls"]:
        uploaded_file.seek(0)
        data = pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file type.")
        st.stop()


    if "uploaded_filename" not in st.session_state or st.session_state["uploaded_filename"] != uploaded_file.name:
        st.session_state["uploaded_filename"] = uploaded_file.name
        st.session_state["original_data"] = data.copy()
        st.session_state["cleaned_data"] = data.copy()


    st.subheader("Dataset Overview")

    st.markdown("**Original Data (head)**")
    st.dataframe(st.session_state["original_data"].head(10))
    st.markdown("**Original Column Types**")
    st.write(st.session_state["original_data"].dtypes)

    st.markdown("---")

    st.subheader("Choose a Cleaning Step")

    step_names = ["Select a step"] + list(CLEANING_STEPS.keys())
    selected_step = st.selectbox("Select a cleaning operation to apply:", step_names)

    if selected_step != "Select a step":
        if st.button("Apply selected cleaning step"):
            func = CLEANING_STEPS[selected_step]
            st.session_state["cleaned_data"] = func(st.session_state["cleaned_data"])
            st.success(f"Applied: {selected_step}")

            st.markdown("**Updated Cleaned Data (head)**")
            st.dataframe(st.session_state["cleaned_data"].head(10))

    st.markdown("---")


    st.subheader("Download Cleaned Dataset")

    cleaned_data = st.session_state["cleaned_data"]
    cleaned_data_filename = "cleaned_" + uploaded_file.name

    if file_extension == ".csv":
        cleaned_bytes = cleaned_data.to_csv(index=False).encode("utf-8")
        mime_type = "text/csv"
    else:
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            cleaned_data.to_excel(writer, index=False, sheet_name="CleanedData")
        buffer.seek(0)
        cleaned_bytes = buffer.getvalue()
        mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

    st.download_button(
        label="📥 Download Cleaned Data",
        data=cleaned_bytes,
        file_name=cleaned_data_filename,
        mime=mime_type,
        key="download_button",
    )

else:
    st.info("Please upload a CSV or Excel file to begin cleaning.")
