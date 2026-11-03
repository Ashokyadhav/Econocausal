# --------------------------------------------------
# DATASET UPLOAD - CSV AND EXCEL
# --------------------------------------------------

st.sidebar.subheader("Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Choose a dataset",
    type=["csv", "xlsx"]
)

df = None

if uploaded_file is not None:
    try:
        file_name = uploaded_file.name.lower()

        if file_name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)

        elif file_name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file, engine="openpyxl")

        if df is not None:
            st.sidebar.success("Dataset loaded successfully!")
            st.sidebar.write(f"File: {uploaded_file.name}")
            st.sidebar.write(f"Rows: {df.shape[0]:,}")
            st.sidebar.write(f"Columns: {df.shape[1]}")

    except Exception as e:
        st.sidebar.error(f"Dataset loading failed: {e}")
