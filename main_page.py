import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

#Initialize state
if "merge_bool" not in st.session_state:
    st.session_state.merge_bool = False
if "stack_bool" not in st.session_state:
    st.session_state.stack_bool = False
if "step2_bool" not in st.session_state:
    st.session_state.step2_bool = False
if "dataset" not in st.session_state:
    st.session_state.dataset = pd.DataFrame()
if "preview_bool" not in st.session_state:
    st.session_state.preview_bool = False
if "dupes_bool" not in st.session_state:
    st.session_state.dupes_bool = False


st.title("Data Cleaner App")

st.sidebar.title("Navigation")
st.sidebar.write("Step 1: Upload and combine datasets")
st.sidebar.write("Step 2: Clean data")

# STEP 1 

st.header("Step 1: Upload and combine datasets")
st.write("Upload, preview and optionally merge or stack datasets")

uploaded_files = st.file_uploader(label="Upload CSV:", 
                                  accept_multiple_files=True, 
                                  type="csv") 

if uploaded_files:
    st.session_state.datasets = [pd.read_csv(file) for file in uploaded_files]
    datasets = st.session_state.datasets

    def merge():
        if not datasets:
            st.session_state.merge_bool = False
            return
        st.session_state.merge_bool = True
        st.session_state.stack_bool = False

    def stack():
        if not datasets:
            st.session_state.stack_bool = False
            return 
        st.session_state.stack_bool = True
        st.session_state.merge_bool = False
    
    # Preview datasets 
    if len(datasets) > 1:
        
        for i, df in enumerate(datasets, 1):
            st.subheader(f"Dataset {i} Preview")
            st.dataframe(df.head())
        
        st.button(label="Merge these datasets?", on_click=merge)
        st.button(label="Stack these datasets?", on_click=stack)
    
    elif len(datasets) == 1:
        dataset = datasets[0]
        st.subheader(f"Dataset Preview")
        st.dataframe(dataset.head())
        st.session_state.dataset = dataset
    
    # Finding common columns
    common_cols = set(datasets[0].columns)
        
    for df in datasets[1:]:
        common_cols &= set(df.columns)

    def merge_all(datasets):
        merged = datasets[0]
        for df in datasets[1:]:
            merged = pd.merge(merged, df, how="inner")
        return merged

    if st.session_state.merge_bool:
        st.session_state.dataset = None
        
        if not common_cols:
            st.error("Merging cannot be done without common columns.")
            st.stop()
        
        merged_df = merge_all(datasets)
        st.subheader("Merged Dataset")
        st.dataframe(merged_df.head())
        st.session_state.dataset = merged_df

    if st.session_state.stack_bool:
        st.session_state.dataset = None

        combine_type = st.radio(
            "How do you want to stack these datasets?",
            ["Row-wise", "Column-wise"]
        )
        
        if combine_type.startswith("Row"):
            if not common_cols:
                st.error("Row-wise stacking cannot be done without common columns.")
                st.stop()
            stacked_df = pd.concat(datasets, axis=0, ignore_index=True) 
        else:
            columns = [col for df in datasets for col in df.columns]
            stacked_df = pd.concat(datasets, axis=1)
            stacked_df.columns = columns
          
        st.subheader("Stacked_Dataset")
        st.dataframe(stacked_df.head())
        st.session_state.dataset = stacked_df

#STEP 2

st.divider()
st.header("Step 2: Clean data")
st.write("Once you have combined your datasets, move to cleaning.")

if not st.session_state.dataset.empty:
    st.session_state.update(step2_bool=True)

if st.session_state.step2_bool:
    dataset = st.session_state.dataset.copy()
    
    # Clean formatting
    dataset.columns = [col.title() for col in dataset.columns]
    dataset = dataset.applymap(lambda x: x.strip().title() if isinstance(x, str) else x)
    
    # Remove duplicates
    st.write("1. Do you want to remove duplicates?")
    st.button("Remove duplicates", on_click=lambda: st.session_state.update(dupes_bool=True))
    
    if st.session_state.dupes_bool:
            before_dupes = len(dataset)
            dataset = dataset.drop_duplicates()
            after_dupes = len(dataset)
            st.success(f"{before_dupes - after_dupes} duplicate values successfully removed!")
     
    st.divider()

    # Missing valuees
    st.write("2. Pick an option for how to deal with missing values:")
    temp1 = dataset
    missing_strat = st.selectbox(label="",
                                 options = ["Do nothing",
                                            "Drop rows with missing values",
                                            "Impute mean/median/mode",
                                             "Fill with custom value"])
    
    if missing_strat == "Drop rows with missing values":
        temp1 = dataset
        temp1 = temp1.dropna(inplace=True)
        st.success("Dropped rows with missing values")
    
    elif missing_strat == "Impute mean/median/mode":
        temp1 = dataset
        method = st.radio("Select method:", ["Mean", "Median", "Mode"])
        
        for col in dataset.select_dtypes(include=[np.number]).columns:
            temp2 = temp1
            
            if method == "Mean":
                temp2 = temp1
                temp2 = temp2[col].fillna(temp2[col].mean(), inplace=True)
            
            elif method == "Median":
                temp2 = temp1
                temp2 = temp2[col].fillna(temp2[col].median(), inplace=True)
            
            elif method == "Mode":
                temp2 = temp1
                temp2 = temp2[col].fillna(temp2[col].mode(), inplace=True)
        
        st.success(f"Filled missing values with {method.lower()} values.")
        temp1 = temp2
    
    elif missing_strat == "Fill with custom value":
        value = st.text_input("Enter value:")
        temp1 = dataset
        
        if value:
            temp1 = temp1.fillna(value, inplace=True)
            st.success(f"Filled missing values with {value}") 
        temp1 = dataset

    st.divider()
    
    # Drop columns
    st.write("3. Do you want to drop any columns?")
    temp3 = temp1
    cols_to_drop = st.pills("Drop columns", temp3.columns, selection_mode="multi")
    
    if cols_to_drop:
        temp3 = temp3.drop(columns=cols_to_drop, inplace=True)
        st.success(f"Dropped columns: {cols_to_drop}")
    temp3 = temp1

    temp1 = temp3
    st.divider()
    

    # Filter columns
    st.write("4. Select column to filter")
    temp4 = temp1
    filter_col = st.pills("", list(temp4.columns), selection_mode="single")
    selected_vals = []
    
    if filter_col:
        col = filter_col[0]
        selected_vals = st.multiselect("Select values to keep", temp4[filter_col].dropna().unique())
    temp4 = temp1
    
    if selected_vals:
        temp4 = temp4[temp4[filter_col].isin(selected_vals)]
        st.success(f"Filtered {filter_col} for selected values.")
    temp4 = temp1

    temp1 = temp4
    st.divider()
    
    st.button(label="Confirm changes", on_click= lambda: st.session_state.update(preview_bool=True))
    
    if st.session_state.preview_bool:
        dataset = temp1
        st.subheader("Preview of cleaned data:")
        st.dataframe(dataset.head())

    st.session_state.dataset = dataset
    

