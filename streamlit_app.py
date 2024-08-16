# Data Manipulation and Analysis
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit for Web App
import streamlit as st

# Machine Learning Models
from sklearn.linear_model import (
    LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
)
from sklearn.tree import (
    DecisionTreeClassifier, DecisionTreeRegressor
)
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.svm import (
    SVC, SVR
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
import lightgbm as lgb

# Metrics
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, roc_curve, precision_score, recall_score,
    confusion_matrix, classification_report, mean_squared_error, r2_score
)

# Preprocessing and Data Manipulation
from sklearn.preprocessing import (
    LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
)
from sklearn.model_selection import (
    train_test_split, cross_val_score
)

# Statistical Tests
from scipy import stats
from scipy.stats import (
    ttest_1samp, ttest_ind, f_oneway
)

# Clustering Models
from sklearn.cluster import (
    KMeans, AgglomerativeClustering, DBSCAN
)

# Dimensionality Reduction
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# Deep Learning Models
#from keras.models import Sequential
#from tensorflow.keras.models import Model
#from tensorflow.keras.layers import Input, Dense
#from keras.utils import to_categorical
# Utility Libraries
import inspect
import io
import contextlib
import time

# Serialization
import pickle
import joblib

import os
import plotly.express as px
def main():
    initialize_session_state()
    apply_custom_css()
    st.sidebar.title("EasyML")
    st.sidebar.markdown("<p style='font-size: 12px; color: grey;'><i>Success is just a step away!</i></p>", unsafe_allow_html=True)
    #Upload and DataFrame Selection
    file_upload()
    
    # Main Tabs
     # Main Tabs
    tabs = st.tabs(["Exploration", "Hypothesis","Preprocessing", "Machine Learning","Bonus"])
    
    selected_df = st.sidebar.selectbox("Select DataFrame", list(st.session_state.dataframes.keys()))
    
    if not selected_df:
        st.warning("Please upload a file or select an existing DataFrame or refresh.")
        return
    
    df = st.session_state.dataframes[selected_df]
    
       
    with tabs[0]:
        data_exploration(df)
    with tabs[1]:
        hypothesis_testing(df)
    with tabs[2]:
        data_cleaning_and_preprocessing(df)
    with tabs[3]:
        machine_learning(df)
    with tabs[4]:
        bonus(df)


def initialize_session_state():
    
    if 'dataframes' not in st.session_state:
        st.session_state['dataframes'] = {}

    if "code_snippets" not in st.session_state:
        st.session_state.code_snippets = []

    # Initialize upload state
    if 'file_uploaded' not in st.session_state:
        st.session_state.file_uploaded = False
    if 'model' not in st.session_state:
        st.session_state['model'] = {}
    if 'X_train' not in st.session_state:
        st.session_state['X_train'] = None
        st.session_state['X_test'] = None
        st.session_state['y_train'] = None
        st.session_state['y_test'] = None
def file_upload():
    if not st.session_state.file_uploaded:
        uploaded_file = st.sidebar.file_uploader("Upload a file", type=["csv"])

        if uploaded_file:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            
            st.session_state.dataframes[uploaded_file.name] = df

            # Set file_uploaded to True
            st.session_state.file_uploaded = True

            
    else:
        st.sidebar.checkbox("File uploaded", value=True, disabled=True)
def apply_custom_css():
    st.markdown("""
    <style>
    /* Style for tab headers */
    .stTabs [role="tab"] {
        width: 200px;
        padding: 8px;
        border: 2px solid #e6e6e6;
        margin-right: 4px;
        text-align: center;
        font-weight: bold;
        font-size: 10px;
    }
    /* Hover effect for tabs */
    .stTabs [role="tab"]:hover {
        background-color: #f1f1f1;
    }
    /* Style for selected tab */
    .stTabs [role="tab"][aria-selected="true"] {
        border-bottom: 2px solid #ff4b4b;
    }
    </style>
    """, unsafe_allow_html=True)
def save_dataframe(df):
    new_df_name = st.sidebar.text_input("Name Modified dataframe")
    st.write(new_df_name)
    
    if st.sidebar.checkbox("Save") and new_df_name:
        st.session_state.dataframes[new_df_name] = df.copy()
        st.success(f"DataFrame saved as {new_df_name}")
@st.cache_data  # Cache based on data inputs
def label_encode_columns(df, columns_to_encode, ordered_unique_values):
    encoded_info = {}
    for column in columns_to_encode:
        # Create a dictionary for the custom ordering
        custom_order_dict = {val: idx for idx, val in enumerate(ordered_unique_values[column])}
        df[column] = df[column].map(custom_order_dict)
        encoded_info[column] = custom_order_dict
    return df, encoded_info


def data_exploration(df):
   
    sub_menu = st.selectbox("Select an Option", [
        "Descriptive Statistics","Show Missing(Null) Values","Show Column Datatypes","Show Unique Values","Show Data Imbalances","Visualization Tools"
    ])
    
    if sub_menu == "Descriptive Statistics":
        display_descriptive_statistics(df)
    elif sub_menu == "Show Missing(Null) Values":
        show_missing_values(df)
    elif sub_menu == "Show Column Datatypes":
        column_datatypes(df)
    elif sub_menu == "Show Unique Values":
        find_unique_values(df)
    elif sub_menu == "Show Data Imbalances":
        show_data_imbalance(df)
    elif sub_menu == "Visualization Tools":
        display_graphs(df)

def display_descriptive_statistics(df):
    if st.checkbox("Show Descriptive statistics"):
        st.write("Descriptive Statistics:")
        st.write(df.describe())

def show_missing_values(df):
    st.write("Null values:")
    st.write(df.isnull().sum().to_frame('Null Values')) 

def column_datatypes(df):
    st.write("Column Data Types:")
    st.write(df.dtypes.to_frame('Data Type'))

def find_unique_values(df):
    min_unique_count = st.number_input(label='Enter minimum unique count:', min_value=1, value=2, key="min_unique_count")
    columns_with_few_unique = {}
    for column in df.columns:
        unique_values = df[column].nunique()
        if unique_values <= min_unique_count:
            columns_with_few_unique[column] = df[column].unique()

    if columns_with_few_unique:
        st.write(f"Columns with fewer than or equal to {min_unique_count} unique values:")
        for column, unique_values in columns_with_few_unique.items():
            st.write(f" '{column}' : {len(unique_values)} unique value(s): {unique_values}")
    else:
        st.write(f"No columns found with fewer than or equal to {min_unique_count} unique values.")
    return df 

def show_data_imbalance(df):
    """
    Checks and visualizes class imbalance for specified columns in a dataset.
    Allows user to select columns, set an imbalance percentage threshold, specify the number of bins for histogram,
    and choose whether to view plots side by side.
    
    Parameters:
    df (pd.DataFrame): The dataset containing the columns to analyze.

    Returns:
    None
    """
    # Select columns from the DataFrame
    columns = df.columns.tolist()
    selected_columns = st.multiselect('Select Columns to Analyze', columns, default=[columns[0]])

    # Set threshold value
    percentage_threshold = st.slider('Select Threshold Percentage', min_value=0, max_value=100, value=10)
    
    # Set number of bins for histogram
    num_bins = st.slider('Select Number of Bins for Histogram', min_value=5, max_value=50, value=10)

    # Option to view plots side by side
    side_by_side = st.checkbox('View Plots Side by Side')

    # Validate selected columns
    if not selected_columns:
        st.error("Please select at least one column.")
        return

    # Prepare to display plots
    if side_by_side:
        # Display plots side by side
        cols = st.columns(len(selected_columns))
        for idx, target_column in enumerate(selected_columns):
            with cols[idx]:
                plot_column(df, target_column, percentage_threshold, num_bins)
    else:
        # Display plots one by one
        for target_column in selected_columns:
            plot_column(df, target_column, percentage_threshold, num_bins)

def plot_column(df, target_column, percentage_threshold, num_bins):
    """
    Helper function to plot class imbalance for a single column.
    
    Parameters:
    df (pd.DataFrame): The dataset containing the column to analyze.
    target_column (str): The column to analyze.
    percentage_threshold (int): The percentage threshold for class imbalance.
    num_bins (int): Number of bins for histogram.
    
    Returns:
    None
    """
    if target_column not in df.columns:
        st.error(f"Column '{target_column}' not found in dataset.")
        return

    # Calculate class distribution
    class_counts = df[target_column].value_counts()
    total_count = len(df)
    class_percentages = (class_counts / total_count) * 100

    # Filter by percentage threshold
    imbalance_filter = class_percentages[class_percentages < percentage_threshold]
    if imbalance_filter.empty:
        st.write(f"No classes below {percentage_threshold}% for column '{target_column}'.")
        return

    # Plot based on datatype
    st.write(f"### Visualization for '{target_column}'")

    if pd.api.types.is_numeric_dtype(df[target_column]):
        # Plot histogram for numeric columns
        fig = px.histogram(df, x=target_column, nbins=num_bins,
                           labels={'x': target_column, 'y': 'Frequency'},
                           title=f'Histogram of {target_column}')
        fig.update_layout(bargap=0.1)  # Adjust gap between bars
    else:
        # Plot bar chart for categorical columns
        fig = px.bar(class_counts, x=class_counts.index, y=class_counts.values,
                     labels={'x': target_column, 'y': 'Number of Samples'},
                     title=f'Class Distribution in {target_column}')
        # Add percentage labels to the bars
        fig.update_traces(text=class_percentages.map(lambda x: f"{x:.2f}%"), textposition='outside')

    st.plotly_chart(fig)

    # Display class distribution table with percentages
    st.write(f"#### Class Counts and Percentages for '{target_column}':")
    class_distribution = pd.DataFrame({
        'Class': class_counts.index,
        'Count': class_counts.values,
        'Percentage': class_percentages.map(lambda x: f"{x:.2f}%")
    })
    st.write(class_distribution)

def display_graphs(df):
        
    if df is not None and st.checkbox("Show graphs"):
       
        graph_type = st.selectbox("Choose graph type", ["Histogram", "Boxplot", "Violin Plot", "Correlation Heatmap", "Scatter Plot with Best Fit Line"])
        columns = st.multiselect("Choose columns for graph", df.columns)
        
        if graph_type == "Histogram" and columns:
            for column in columns:
                st.write(f"Histogram for {column}")
                plt.figure(figsize=(10, 4))
                sns.histplot(df[column].dropna(), kde=True)
                st.pyplot(plt)
        
        elif graph_type == "Boxplot" and columns:
            st.write("Boxplot")
            plt.figure(figsize=(10, 4))
            sns.boxplot(data=df[columns].dropna())
            st.pyplot(plt)

        elif graph_type == "Violin Plot" and columns:
            st.write("Violin Plot")
            plt.figure(figsize=(10, 4))
            sns.violinplot(data=df[columns].dropna())
            st.pyplot(plt)
        
        elif graph_type == "Correlation Heatmap" and columns:
            st.write("Correlation Heatmap")
            plt.figure(figsize=(10, 4))
            
            # Compute the correlation matrix
            corr = df[columns].corr()
            
            # Create a mask for the upper triangle
            mask = np.triu(np.ones_like(corr, dtype=bool))
            
            # Draw the heatmap with the mask
            sns.heatmap(corr, mask=mask, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
            st.pyplot(plt)
            st.write("Explanation: The values in the heatmap range from -1 to 1. A value close to 1 implies a strong positive correlation, close to -1 implies a strong negative correlation, and around 0 implies no correlation.")
        
        elif graph_type == "Scatter Plot with Best Fit Line" and len(columns) == 2:
            hue_column = st.selectbox("Select column for hue", [None] + list(df.columns))
            palette = st.selectbox("Select palette", ["deep", "muted", "bright", "pastel", "dark", "colorblind"])

            st.write(f"Scatter Plot for {columns[0]} vs {columns[1]} with Best Fit Line")
            plt.figure(figsize=(10, 4))
            
            if hue_column and hue_column != "None":
                sns.scatterplot(x=df[columns[0]], y=df[columns[1]], hue=df[hue_column], palette=palette)
            else:
                sns.scatterplot(x=df[columns[0]], y=df[columns[1]])

            sns.regplot(x=df[columns[0]], y=df[columns[1]], scatter=False, color='red')
            st.pyplot(plt) 


# Function for hypothesis testing
def hypothesis_testing(df):
    
    if df is not None and st.checkbox("Perform hypothesis testing"):
        st.subheader("Hypothesis Testing")

        # Get user-defined p-value threshold
        alpha = st.number_input("Significance level (alpha)", min_value=0.0, max_value=1.0, value=0.05)

        test_type = st.selectbox("Choose test type", ["One Sample T-test", "Two Sample T-test", "ANOVA", "Chi-square", "Correlation Coefficient"])
        
        if test_type == "One Sample T-test":
            column = st.selectbox("Choose column", df.columns)
            popmean = st.number_input("Population mean", value=0.0)
            if st.button("Perform One Sample T-test"):
                t_stat, p_val = ttest_1samp(df[column].dropna(), popmean)
                st.write(f"T-statistic: {t_stat}, P-value: {p_val}")
                if p_val <= alpha:
                    st.write(f"Conclusion: Since the p-value is less than or equal to {alpha}, there is strong evidence against the null hypothesis. Therefore, we reject the null hypothesis.")
                else:
                    st.write(f"Conclusion: Since the p-value is greater than {alpha}, we do not have sufficient evidence to reject the null hypothesis. Therefore, we accept the null hypothesis.")

        elif test_type == "Two Sample T-test":
            column1 = st.selectbox("Choose first column", df.columns)
            column2 = st.selectbox("Choose second column", df.columns)
            if st.button("Perform Two Sample T-test"):
                t_stat, p_val = ttest_ind(df[column1].dropna(), df[column2].dropna())
                st.write(f"T-statistic: {t_stat}, P-value: {p_val}")
                if p_val <= alpha:
                    st.write(f"Conclusion: Since the p-value is less than or equal to {alpha}, there is strong evidence against the null hypothesis. Therefore, we reject the null hypothesis.")
                else:
                    st.write(f"Conclusion: Since the p-value is greater than {alpha}, we do not have sufficient evidence to reject the null hypothesis. Therefore, we accept the null hypothesis.")

        elif test_type == "ANOVA":
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()
            
            if numeric_columns:
                columns = st.multiselect("Choose columns for ANOVA", df.columns)
                if columns:
                    accepted_columns = [col for col in columns if col in numeric_columns]
                    rejected_columns = [col for col in columns if col in non_numeric_columns]
                    
                    if accepted_columns:
                        if st.button("Perform ANOVA"):
                            try:
                                f_stat, p_val = f_oneway(*[df[col].dropna() for col in accepted_columns])
                                
                                st.write("Columns considered for ANOVA:", accepted_columns)
                                if rejected_columns:
                                    st.warning(f"These columns were not suitable for ANOVA due to non-numeric data types: {rejected_columns}")
                                st.write(f"F-statistic: {f_stat}, P-value: {p_val}")
                                if p_val <= alpha:
                                    st.write(f"Conclusion: Since the p-value is less than or equal to {alpha}, there is strong evidence against the null hypothesis. Therefore, we reject the null hypothesis.")
                                else:
                                    st.write(f"Conclusion: Since the p-value is greater than {alpha}, we do not have sufficient evidence to reject the null hypothesis. Therefore, we accept the null hypothesis.")
                            except Exception as e:
                                st.error(f"Error performing ANOVA: {e}")
                    else:
                        st.write("Please select at least two numeric columns to perform ANOVA.")
                else:
                    st.write("Please select columns to perform ANOVA.")
            else:
                st.write("No numeric columns available for ANOVA.")

        elif test_type == "Correlation Coefficient":
            column1 = st.selectbox("Choose first column", df.columns)
            column2 = st.selectbox("Choose second column", df.columns)
            if st.button("Calculate Correlation Coefficient"):
                corr_coef = np.corrcoef(df[column1].dropna(), df[column2].dropna())[0, 1]
                st.write(f"Correlation Coefficient: {corr_coef}")
                st.write("Explanation: A value close to 1 implies a strong positive correlation, close to -1 implies a strong negative correlation, and around 0 implies no correlation.")

        elif test_type == "Chi-square":
            categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
            col1 = st.selectbox("Select first categorical column", categorical_columns)
            col2 = st.selectbox("Select second categorical column", categorical_columns)
            if st.button("Perform Chi-square test"):
                contingency_table = pd.crosstab(df[col1], df[col2])
                chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
                st.write("Chi-square test results:")
                st.write(pd.DataFrame({"Chi2-statistic": [chi2_stat], "p-value": [p_value], "degrees of freedom": [dof]}))
                st.write("Explanation: A low p-value (< {alpha}) indicates that we can reject the null hypothesis, suggesting a significant association between the two categorical variables.")


def data_cleaning_and_preprocessing(df):
    
    sub_menu = st.selectbox("Select an Option", [
        "Handle Missing Values",
        "Encode Categorical Variables",
        "Scale and Normalize Data","Drop Columns","Drop Rows","Drop Duplicates","Shift Values","Create new Columns",
        "Feature Engineering"
    ])
    
    if sub_menu == "Handle Missing Values":
        handle_missing_values(df)
    elif sub_menu == "Encode Categorical Variables":
        encode_categorical_variables(df)
    elif sub_menu == "Scale and Normalize Data":
        scale_normalize_data(df)
    elif sub_menu == "Drop Columns":
        drop_columns(df)
    elif sub_menu == "Drop Rows":
        drop_null_rows(df)
    elif sub_menu == "Drop Duplicates":
        drop_null_rows(df)
    elif sub_menu == "Shift Values":
        shift_values(df)
    elif sub_menu == "Create new Columns":
        create_new_column(df)
    elif sub_menu == "Feature Engineering":
        feature_engineering(df)
# Subcategory Functions
def handle_missing_values(df):
    
    # Section for filling missing values with the mean
    if st.checkbox("Fill with Mean"):
        columns_to_fill_mean = st.multiselect("Select columns to fill with Mean", df.columns)
        if columns_to_fill_mean:
            for column in columns_to_fill_mean:
                df[column] = df[column].fillna(df[column].mean())
            st.write("DataFrame after filling missing values with Mean" )
            st.write(df)

    # Section for filling missing values with the median
    if st.checkbox("Fill with Median"):
        columns_to_fill_median = st.multiselect("Select columns to fill with Median", df.columns)
        if columns_to_fill_median:
            for column in columns_to_fill_median:
                df[column] = df[column].fillna(df[column].median())
            st.write("DataFrame after filling missing values with Median")
            st.write(df)

    # Section for filling missing values with a specified value
    if st.checkbox("Fill with Specified Value"):
        columns_to_fill_value = st.multiselect("Select columns to fill with Specified Value", df.columns)
        specified_value = st.number_input("Enter specified value to fill missing values", value=0.0)
        if columns_to_fill_value:
            for column in columns_to_fill_value:
                df[column] = df[column].fillna(specified_value)
            st.write("DataFrame after filling missing values with Specified Value")
            st.write(df)
    # Call the save function
    save_dataframe(df)

def encode_categorical_variables(df):
    
    if st.checkbox("Label Encoding"):
        st.subheader("Label Encoding")
        columns = df.select_dtypes(include=['object', 'category']).columns
        selected_columns = st.multiselect("Select columns for label encoding", columns)

        if selected_columns:
            unique_values_dict = {}
            for column in selected_columns:
                unique_values_dict[column] = df[column].dropna().unique()
                unique_values_dict[column] = sorted(unique_values_dict[column])

            st.write("Unique values for selected columns:")
            for column in selected_columns:
                st.write(f"Column '{column}':")
                st.write(unique_values_dict[column])

            # Allow the user to order the unique values for each selected column
            ordered_unique_values = {}
            for column in selected_columns:
                ordered_values = st.multiselect(f"Order the unique values for '{column}' encoding", unique_values_dict[column], unique_values_dict[column])
                ordered_unique_values[column] = ordered_values

            if all(len(ordered_unique_values[column]) == len(unique_values_dict[column]) for column in selected_columns):
                # Perform label encoding for selected columns and cache the result
                df, encoding_info = label_encode_columns(df.copy(), selected_columns, ordered_unique_values)
                st.write("Dataframe after Custom Ordered Label Encoding")
                st.dataframe(df)
                st.write("Custom Label Encoding Information:")
                st.write(encoding_info)
            else:
                st.warning("Please order all unique values for selected columns.")
               

    if st.checkbox("One-hot Encoding"):
        st.subheader("One-hot Encoding")
        columns = st.multiselect("Select columns for one-hot encoding", df.select_dtypes(include=['object', 'category']).columns)
        if columns:
            encoder = OneHotEncoder(drop='first')
            encoded_data = encoder.fit_transform(df[columns])
            encoded_df = pd.DataFrame(encoded_data.toarray(), columns=encoder.get_feature_names_out(columns))
            df = pd.concat([df, encoded_df], axis=1)
            df = df.drop(columns=columns)
            st.write("Dataframe after One-hot Encoding", df)
        # Call the save function
    save_dataframe(df)    

def scale_normalize_data(df):
    st.sidebar.subheader("Scaling/Normalizing going on...")
    if st.checkbox("Standardization"):
        columns_to_standardize = st.multiselect("Select columns for standardization", df.select_dtypes(include=['float64', 'int64']).columns)
        if columns_to_standardize:
            scaler = StandardScaler()
            df[columns_to_standardize] = scaler.fit_transform(df[columns_to_standardize])
            st.write("DataFrame after Standardization:")
            st.write(df)
    if st.checkbox("Normalization"):
        columns_to_normalize = st.multiselect("Select columns for normalization", df.select_dtypes(include=['float64', 'int64']).columns)
        if columns_to_normalize:
            scaler = MinMaxScaler()
            df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
            st.write("DataFrame after Normalization:")
            st.write(df)
    
    save_dataframe(df)  

def drop_columns(df):
    drop_columns = st.multiselect("Select columns to drop", df.columns, key="drop_columns")
    if drop_columns:
        df = df.drop(columns=drop_columns)
        st.write("Dataframe after dropping columns")
        st.write(df)
    # Call the save function
    save_dataframe(df)

def drop_null_rows(df):
    drop_null_rows = st.checkbox("Drop rows with null values", key="drop_null_rows")
    if drop_null_rows:
        df = df.dropna()
        st.write("Dataframe after dropping rows with null values")
        st.write(df)
    save_dataframe(df)

def drop_duplicates(df):
    drop_duplicates = st.checkbox("Check for Duplicates and drop", key="drop_duplicates")
    if drop_duplicates:
        duplicates = df[df.duplicated()]
        if duplicates.shape[0] == 0:
            st.write("No duplicates found.")
        else:
            st.write("Duplicates found. Dropping duplicates...")
            st.write("Dataframe after dropping duplicates")
            df = df.drop_duplicates()
            st.write(df)
    save_dataframe(df)

def shift_values(df):
    st.subheader("Shift Values")
    shift_columns = st.multiselect("Select columns to shift", df.columns, key="shift_columns")
    shift_period = st.number_input("Enter number of periods to shift", min_value=1, value=1, key="shift_period")
    shift_direction = st.radio("Select shift direction", ["Forward", "Backward"], key="shift_direction")
    if shift_direction == "Forward":
        shift_period = -shift_period
    if st.button("Apply Shift", key="apply_shift"):
        try:
            df[shift_columns] = df[shift_columns].shift(periods=shift_period)
            st.write("Dataframe after shifting values", df)
        except Exception as e:
            st.error(f"Error during shifting: {e}")
    save_dataframe(df)

def create_new_column(df):
    new_col_name = st.text_input("New column name", key="new_col_name")
    new_col_expr = st.text_input("Expression for new column (e.g., col1 + col2)", key="new_col_expr")
    if new_col_name and new_col_expr:
        try:
            df[new_col_name] = df.eval(new_col_expr)
            st.write("Dataframe with new column")
            st.write(df)
        except Exception as e:
            st.error(f"Error creating new column: {e}")
    save_dataframe(df)

def feature_engineering(df):
    
    # Convert a column to datetime
    if st.checkbox("Convert Column to Datetime"):
        st.subheader("Convert Column to Datetime")
        column_to_convert = st.selectbox("Select column to convert to datetime", df.columns)
        if column_to_convert:
            try:
                df[column_to_convert] = pd.to_datetime(df[column_to_convert])
                st.write(f"Column '{column_to_convert}' successfully converted to datetime.")
                st.write(df.head())
            except Exception as e:
                st.error(f"Error converting column to datetime: {e}")
    
    # Extracting date features
    if st.checkbox("Extract Date Features"):
        st.subheader("Extract Date Features")
        date_column = st.selectbox("Select date column", df.columns)
        if date_column and pd.api.types.is_datetime64_any_dtype(df[date_column]):
            df['Year'] = df[date_column].dt.year
            df['Month'] = df[date_column].dt.month
            df['Day'] = df[date_column].dt.day
            df['DayOfWeek'] = df[date_column].dt.dayofweek  # Monday=0, Sunday=6
            df['Season'] = (df[date_column].dt.month % 12 // 3 + 1).map({1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'})
            st.write("DataFrame with extracted date features")
            st.write(df)
        else:
            st.error(f"The selected column '{date_column}' is not in datetime format.")

    if st.checkbox("Create New Features"):
        st.subheader("New Features")
        new_col_name = st.text_input("New column name")
        new_col_expr = st.text_input("Expression for new column (e.g., col1 + col2)")
        if new_col_name and new_col_expr:
            try:
                df[new_col_name] = df.eval(new_col_expr)
                st.write("Dataframe with new column")
                st.write(df)
            except Exception as e:
                st.error(f"Error creating new column: {e}")
    
    # Group by and aggregate options
    if st.checkbox("Group By and Aggregate"):
        st.subheader("Group By and Aggregate")
        group_by_column = st.selectbox("Select column to group by", df.columns)
        aggregate_column = st.selectbox("Select column to aggregate", df.columns)
        aggregation_func = st.selectbox("Select aggregation function", ["Mean", "Sum", "Count", "Max", "Min"])
        
        if group_by_column and aggregate_column and aggregation_func:
            agg_func_map = {
                "Mean": 'mean',
                "Sum": 'sum',
                "Count": 'count',
                "Max": 'max',
                "Min": 'min'
            }
            aggregation = agg_func_map[aggregation_func]
            grouped_df = df.groupby(group_by_column).agg({aggregate_column: aggregation}).reset_index()
            st.write("Grouped and Aggregated DataFrame")
            st.write(grouped_df)
    
    # Mapping values
    if st.checkbox("Map Values"):
        st.subheader("Map Values")
        column_to_map = st.selectbox("Select column to map", df.columns)
        if column_to_map:
            unique_values = df[column_to_map].unique()
            st.write(f"Unique values in '{column_to_map}':", unique_values)
            value_mapping = {}
            for value in unique_values:
                new_value = st.text_input(f"Map '{value}' to", key=value)
                if new_value:
                    value_mapping[value] = new_value
            if value_mapping:
                df[column_to_map] = df[column_to_map].map(value_mapping)
                st.write("DataFrame after mapping values")
                st.write(df)
    save_dataframe(df)   


def machine_learning(df):
      
    if df is not None and st.checkbox("Perform machine learning"):
        st.subheader("Machine Learning")
        
        problem_type = st.selectbox("Choose problem type", ["Regression", "Classification"])
        
        target_column = st.selectbox("Select target column", df.columns)

        show_feature_selection = st.checkbox("Show Feature Selection Options")
    

        
        all_feature_columns = [col for col in df.columns if col != target_column]
        # Display feature columns selection
        if show_feature_selection:
            features = st.multiselect("Select Feature Columns", all_feature_columns, default=all_feature_columns)
        
        
        
            test_size = st.slider("Select test size", 0.1, 0.5, 0.2)
            random_state = st.slider("Select random state", 0, 100, 42)
            
            X = df[features]
            y = df[target_column]
            
            hyperparameters = {}
            
            tune_regression = st.checkbox("Enable Hyperparameter Tuning for Regression Models", value=True)
            tune_classification = st.checkbox("Enable Hyperparameter Tuning for Classification Models", value=True)
            tune_nn = st.checkbox("Enable Hyperparameter Tuning for Neural Networks", value=True)

            if problem_type == "Regression":
                regression_models = {
                    "Linear Regression": LinearRegression(),
                    "Ridge Regression": Ridge(),
                    "Lasso Regression": Lasso(),
                    "ElasticNet": ElasticNet(),
                    "Decision Tree Regressor": DecisionTreeRegressor(),
                    "Random Forest Regressor": RandomForestRegressor(),
                    "Gradient Boosting Regressor": GradientBoostingRegressor(),
                    "SVR": SVR(),
                    "LightGBM Regressor": lgb.LGBMRegressor()
                }
                selected_models = st.multiselect("Select regression models", list(regression_models.keys()), default=list(regression_models.keys()))
                
                if "Linear Regression" in selected_models and tune_regression:
                    fit_intercept = st.checkbox("Linear Regression - Fit Intercept", True)
                    hyperparameters["Linear Regression"] = {"fit_intercept": fit_intercept}
                    
                if "Ridge Regression" in selected_models and tune_regression:
                    alpha = st.slider("Ridge Regression - Alpha", 0.01, 10.0, 1.0)
                    hyperparameters["Ridge Regression"] = {"alpha": alpha}
                    
                if "Lasso Regression" in selected_models and tune_regression:
                    alpha = st.slider("Lasso Regression - Alpha", 0.01, 10.0, 1.0)
                    hyperparameters["Lasso Regression"] = {"alpha": alpha}
                    
                if "ElasticNet" in selected_models and tune_regression:
                    alpha = st.slider("ElasticNet - Alpha", 0.01, 10.0, 1.0)
                    l1_ratio = st.slider("ElasticNet - L1 Ratio", 0.0, 1.0, 0.5)
                    hyperparameters["ElasticNet"] = {"alpha": alpha, "l1_ratio": l1_ratio}
                    
            elif problem_type == "Classification":
                classification_models = {
                    "Logistic Regression": LogisticRegression(),
                    "Decision Tree": DecisionTreeClassifier(),
                    "Random Forest": RandomForestClassifier(),
                    "Gradient Boosting": GradientBoostingClassifier(),
                    "SVM": SVC(),
                    "LightGBM": lgb.LGBMClassifier(),
                    "K-Nearest Neighbors": KNeighborsClassifier(),
                    "Naive Bayes": GaussianNB()
                }
                selected_models = st.multiselect("Select classification models", list(classification_models.keys()), default=list(classification_models.keys()))
                
                if "Logistic Regression" in selected_models and tune_classification:
                    solver = st.selectbox("Logistic Regression - Solver", ["liblinear", "saga", "newton-cg", "lbfgs"])
                    penalty_options = ["l1", "l2", "elasticnet", None] if solver in ["saga", "lbfgs"] else ["l1", "l2"]
                    penalty = st.selectbox("Logistic Regression - Penalty", penalty_options)
                    C = st.slider("Logistic Regression - C", 0.001, 10.0, 1.0)
                    max_iter = st.slider("Logistic Regression - Max Iter", 50, 500, 100)
                    hyperparameters["Logistic Regression"] = {"C": C, "solver": solver, "penalty": penalty, "max_iter": max_iter}
                    
                if "Decision Tree" in selected_models and tune_classification:
                    max_depth = st.slider("Decision Tree - Max Depth", 1, 20, 10)
                    min_samples_split = st.slider("Decision Tree - Min Samples Split", 2, 20, 2)
                    hyperparameters["Decision Tree"] = {"max_depth": max_depth, "min_samples_split": min_samples_split}
                    
            
                                    
            if st.button("Train models"):
                try:
                    if problem_type in ["Regression", "Classification"]:
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
                        
                        results = []
                        for name in selected_models:
                            if problem_type == "Regression":
                                model = regression_models[name]
                                if name in hyperparameters:
                                    model.set_params(**hyperparameters[name])
                                model.fit(X_train, y_train)
                                predictions = model.predict(X_test)
                                mse = mean_squared_error(y_test, predictions)
                                r2 = r2_score(y_test, predictions)
                                results.append({"Model": name, "Mean Squared Error": mse, "R2 Score": r2})
                            
                            elif problem_type == "Classification":
                                model = classification_models[name]
                                if name in hyperparameters:
                                    model.set_params(**hyperparameters[name])
                                model.fit(X_train, y_train)
                                predictions = model.predict(X_test)
                                accuracy = accuracy_score(y_test, predictions)
                                precision = precision_score(y_test, predictions, average='weighted')
                                recall = recall_score(y_test, predictions, average='weighted')
                                f1 = f1_score(y_test, predictions, average='weighted')
                                results.append({"Model": name, "Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1 Score": f1})
                        
                        st.write(f"{problem_type} model performance:")
                        st.write(pd.DataFrame(results))
                    
                    elif problem_type == "Unsupervised Learning":
                        for name in selected_models:
                            model = unsupervised_models[name]
                            if name == "PCA" or name == "t-SNE":
                                transformed_data = model.fit_transform(X)
                                st.write(f"{name} transformed data:")
                                st.write(transformed_data)
                                if name == "PCA":
                                    st.write("Explained variance ratio:")
                                    st.write(model.explained_variance_ratio_)
                            
                
                except Exception as e:
                    st.error(f"An error occurred during model training: {e}")

def bonus(df):
    sub_menu = st.selectbox("Select an Option", [
        
       
        "Show Code",
        "Delete Dataframe"
    ])
    
    if sub_menu == "Execute Code":
        execute_code(df)
    elif sub_menu == "Show Code":
        show_source_code()
    elif sub_menu == "Delete Dataframe":
        delete_dataframe()
# Function to enter custom code and execute it
def execute_code(df):
    # Input area for code to be executed
    code = st.text_area("Type your Python code here", height=200)
    
       
    if df is None:
        st.warning("No DataFrame selected or found in session state.")
        return

    exec_locals = {"df": df.copy()}  # Initialize exec_locals with a copy of the selected DataFrame

    # Button to run the code
    if st.button("Run Code"):
        output = io.StringIO()
        exec_globals = globals().copy()
        exec_globals.update({"st": st, "pd": pd, "io": io})

        try:
            # Redirect stdout to capture the output of the executed code
            with contextlib.redirect_stdout(output):
                exec(code, exec_globals, exec_locals)

            st.success("Code executed successfully!")

            # Display the output captured
            result = output.getvalue()
            if result.strip():
                st.write("Output from executed code:")
                result=pd.dataFrame(result)
                st.write(result)

            # Check if 'df' in exec_locals was updated or created
            if "df" in exec_locals and isinstance(exec_locals["df"], pd.DataFrame):
                st.session_state.modified_df = result
                
                st.write("Modified DataFrame:")
                st.write(result)
                st.write(f"Modified DataFrame Shape: {result.shape}")
            else:
                st.warning("No valid DataFrame found in the executed code.")

        except Exception as e:
            st.error(f"Error: {e}")

    # Option to save the modified DataFrame with a new name
    if "result" in st.session_state:
        new_name = st.text_input("Enter new DataFrame name", "")
        if st.button("Save DataFrame") and new_name:
            if new_name in st.session_state.dataframes:
                st.warning(f"DataFrame name '{new_name}' already exists. Please choose a different name.")
            else:
                st.session_state.dataframes[new_name] = st.session_state.modified_df.copy()
                st.success(f"DataFrame saved as '{new_name}' successfully!")
                #st.experimental_rerun()  # Force re-run to refresh UI components

    # Option to save code
    if st.text_input("Enter filename (excluding .py extension)", ""):
        if st.button("Save Code") and code:
            st.session_state.code_snippets.append(code)
            st.success("Code successfully saved!")
    
    # Display saved code snippets
    if st.session_state.code_snippets:
        st.write("Saved Code Snippets:")
        for idx, snippet in enumerate(st.session_state.code_snippets):
            st.write(f"### Code Snippet {idx + 1}")
            st.code(snippet, language='python')

    # Display all available DataFrames
    st.write("DataFrames available in session state:")
    for name, df in st.session_state.dataframes.items():
        st.write(f"{name}: {df}")
        

# Function to show code of different functions
def show_source_code():
    # Define your password here
    PASSWORD = "Fun"  # Change this to your desired password
    
    # Input field for password
    entered_password = st.text_input("Enter password to view source code", type="password")
    
    if entered_password == PASSWORD:
        # Show source code if password is correct
        function_list = [
            "initialize_session_state",
            "file_upload",
            "apply_custom_css",
            "save_dataframe",
            "label_encode_columns",
            "data_exploration",
            "display_descriptive_statistics",
            "show_missing_values",
            "column_datatypes",
            "find_unique_values",
            
            "display_graphs"
            "hypothesis_testing",
            "data_cleaning_and_preprocessing",
            "handle_missing_values",
            "encode_categorical_variables",
            "scale_normalize_data",
            "drop_columns",
            "drop_null_rows",
            "drop_duplicates",
            "shift_values",
            "create_new_column",
            "machine_learning",
            
            "delete_dataframe",
            "feature_engineering",
            "bonus"
        ]
        selected_function = st.selectbox("Select a function to view its source code", function_list)
        
        if selected_function:
            function_source_code = inspect.getsource(globals()[selected_function])
            st.code(function_source_code, language='python')
    else:
        # Show a message if the password is incorrect or not entered
        if entered_password:
            st.warning("Incorrect password. Please try again.")

# Function to delete unnecessary dataframes created
def delete_dataframe():
    # Display available DataFrames
    if "dataframes" not in st.session_state or not st.session_state.dataframes:
        st.write("No DataFrames available to delete.")
        return

    # Dropdown to select which DataFrame to delete
    df_names = list(st.session_state.dataframes.keys())
    selected_df_name = st.selectbox("Select DataFrame to delete", df_names)

    if st.button("Delete DataFrame"):
        if selected_df_name in st.session_state.dataframes:
            del st.session_state.dataframes[selected_df_name]
            st.success(f"DataFrame '{selected_df_name}' deleted successfully!")
            
        else:
            st.warning("Selected DataFrame not found.")





if __name__ == "__main__":
    main()
