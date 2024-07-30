import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing and Data Manipulation
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

# Statistical Tests
from scipy import stats
from scipy.stats import ttest_1samp, ttest_ind, f_oneway

# Metrics
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, silhouette_score
)

# Regression Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Clustering Models
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

# Dimensionality Reduction
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Deep Learning Models
from keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from keras.utils import to_categorical

# Utility Libraries
import inspect
import io
import contextlib

# Additional Libraries
import lightgbm as lgb


# Function to initialize session state
def initialize_session_state():
    
   
    if "dataframes" not in st.session_state:
        st.session_state.dataframes = {}
    if "code_snippets" not in st.session_state:
        st.session_state.code_snippets = []
    
# Function to load data from different sources
def load_data(source):
    if isinstance(source, str):
        if source.endswith('.csv'):
            return pd.read_csv(source)
        elif source.endswith('.xlsx') or source.endswith('.xls'):
            return pd.read_excel(source)
        elif source.endswith('.json'):
            return pd.read_json(source)
        else:
            raise ValueError("Unsupported file type")
    else:
        return pd.read_csv(source)

# Function to display basic information about selected DataFrame
def display_basic_info():
    st.sidebar.header("Data Information")
    
    selected_df = st.sidebar.selectbox("Select DataFrame", st.session_state.dataframes.keys())
    df = st.session_state.dataframes.get(selected_df)
    
    if df is not None:
        if st.sidebar.checkbox("Show shape and size"):
            st.write("Shape of the dataset:", df.shape)
            
            st.write("First five rows:")
            st.write(df.head())
            st.write("Last five rows:")
            st.write(df.tail())
            st.write("Null values:")
            st.write(df.isnull().sum().to_frame('Null Values'))
            
        if st.sidebar.checkbox("Show column data types"):
            st.write("Column Data Types:")
            st.write(df.dtypes.to_frame('Data Type'))
        
        # Sidebar for showing descriptive statistics
        if st.sidebar.checkbox("Show descriptive statistics"):
            # Select all button
            if st.sidebar.button('Select all columns'):
                selected_columns = df.columns.tolist()
            else:
                selected_columns = st.sidebar.multiselect("Select specific columns", df.columns)

            if selected_columns:
                numeric_cols = df[selected_columns].select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    st.write(df[numeric_cols].describe())
                else:
                    st.write("No numeric columns selected for descriptive statistics.")
                
                non_numeric_cols = list(set(selected_columns) - set(numeric_cols))
                if non_numeric_cols:
                    st.write("Non-numeric columns selected (excluded from descriptive statistics):", non_numeric_cols)
            else:
                st.write("Please select columns to view descriptive statistics.")
        
        # Sidebar for correlation and independent columns
        if st.sidebar.checkbox("Show correlation values and independent columns"):
            target_column = st.sidebar.selectbox("Select target column", df.columns)
            
            if target_column:
                if df[target_column].dtype not in [np.float64, np.float32, np.int64, np.int32]:
                    st.write(f"The selected target column '{target_column}' is non-numeric. Please select another column.")
                else:
                    st.write(f"Correlation with {target_column}:")
                    try:
                        correlation = df.corr()[target_column].sort_values(ascending=False)
                        
                        # Filter correlation values based on user input
                        filter_option = st.sidebar.radio("Filter correlation values", ("All", "Range"))
                        
                        if filter_option == "Range":
                            min_val = st.sidebar.slider("Minimum correlation value", -1.0, 1.0, 0.0)
                            max_val = st.sidebar.slider("Maximum correlation value", -1.0, 1.0, 1.0)
                            correlation = correlation[(correlation >= min_val) & (correlation <= max_val)]
                        
                        st.write(correlation.to_frame('Correlation'))
                        
                        # Plotting individual correlations with the target column as a bar plot
                        plt.figure(figsize=(10, 6))
                        sns.barplot(x=correlation.index, y=correlation.values)
                        plt.title(f"Correlation with {target_column}")
                        plt.xticks(rotation=90)
                        for i, v in enumerate(correlation.values):
                            plt.text(i, v, f"{v:.2f}", ha='center', va='bottom')
                        st.pyplot(plt)
                        
                    except Exception as e:
                        st.write(f"Error calculating correlation: {e}")
                
                possible_independent_cols = df.columns.difference([target_column]).tolist()
                st.write("Possible independent columns:")
                st.write(possible_independent_cols)

@st.cache_data  # Cache based on data inputs
def label_encode_columns(df, columns_to_encode, ordered_unique_values):
    encoded_info = {}
    for column in columns_to_encode:
        # Create a dictionary for the custom ordering
        custom_order_dict = {val: idx for idx, val in enumerate(ordered_unique_values[column])}
        df[column] = df[column].map(custom_order_dict)
        encoded_info[column] = custom_order_dict
    return df, encoded_info
    
# Function to modify the dataframe using drop column, rows, missing values, label encoder, one hot encoder, shift
def dataframe_modification():
    st.sidebar.header("Dataframe Modification")
    
    selected_df = st.selectbox("Select DataFrame", st.session_state.dataframes.keys())
    df = st.session_state.dataframes[selected_df].copy()
    
    if st.sidebar.checkbox("Data Preprocessing"):
        st.subheader("Preprocessing options")
        
        selected_columns = st.multiselect("Select columns to display", df.columns)
        if selected_columns:
            st.write("Selected Columns Dataframe")
            st.write(df[selected_columns])
        
        drop_duplicates = st.checkbox("Check for Duplicates and drop")
        if drop_duplicates:
            duplicates = df[df.duplicated()]

            if duplicates.shape[0] == 0:
                st.write("No duplicates found.")
                
            else:
                st.write("Duplicates found. Dropping duplicates...")
                st.write("Dataframe after dropping duplicates")
                df = df.drop_duplicates()
                st.write(df)
          



        find_uniques = st.checkbox("Find columns with unique values less than or equal to input number")
        if find_uniques:
            min_unique_count = st.number_input(label='Enter minimum unique count:', min_value=1, value=2)
            columns_with_few_unique = {}

            # Iterate over each column in the DataFrame
            for column in df.columns:
                unique_values = df[column].nunique()  # Count unique values in the column
                if unique_values <= min_unique_count:
                    columns_with_few_unique[column] = df[column].unique() 
            
            if columns_with_few_unique:
                st.write(f"Columns with fewer than or equal to {min_unique_count} unique values:")
                for column, unique_values in columns_with_few_unique.items():
                    st.write(f" '{column}' : {len(unique_values)} unique value(s): {unique_values}")
            else:
                st.write(f"No columns found with fewer than or equal to {min_unique_count} unique values.")   

        drop_columns = st.multiselect("Select columns to drop", df.columns)
        if drop_columns:
            df = df.drop(columns=drop_columns)
            st.write("Dataframe after dropping columns")
            st.write(df)

        drop_null_rows = st.checkbox("Drop rows with null values")
        if drop_null_rows:
            df = df.dropna()
            st.write("Dataframe after dropping rows with null values")
            st.write(df)
        
        new_col_name = st.text_input("New column name")
        new_col_expr = st.text_input("Expression for new column (e.g., col1 + col2)")
        if new_col_name and new_col_expr:
            try:
                df[new_col_name] = df.eval(new_col_expr)
                st.write("Dataframe with new column")
                st.write(df)
            except Exception as e:
                st.error(f"Error creating new column: {e}")

        if st.sidebar.checkbox("Fill Missing Values"):
            st.subheader("Fill Missing Values")
            fill_method = st.selectbox("Select fill method", ["Mean", "Median", "Specified Value"])
            columns_to_fill = st.multiselect("Select columns to fill", df.columns)
            if fill_method == "Mean" and columns_to_fill:
                for column in columns_to_fill:
                    df[column] = df[column].fillna(df[column].mean())
                st.write("Dataframe after filling missing values with mean", df)
            elif fill_method == "Median" and columns_to_fill:
                for column in columns_to_fill:
                    df[column] = df[column].fillna(df[column].median())
                st.write("Dataframe after filling missing values with median", df)
            elif fill_method == "Specified Value" and columns_to_fill:
                specified_value = st.number_input("Enter specified value to fill missing values", value=0.0)
                for column in columns_to_fill:
                    df[column] = df[column].fillna(specified_value)
                st.write("Dataframe after filling missing values with specified value", df)

        if st.sidebar.checkbox("Label Encoding"):
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

        if st.sidebar.checkbox("One-hot Encoding"):
            st.subheader("One-hot Encoding")
            columns = st.multiselect("Select columns for one-hot encoding", df.select_dtypes(include=['object', 'category']).columns)
            if columns:
                encoder = OneHotEncoder(drop='first')
                encoded_data = encoder.fit_transform(df[columns])
                encoded_df = pd.DataFrame(encoded_data.toarray(), columns=encoder.get_feature_names_out(columns))
                df = pd.concat([df, encoded_df], axis=1)
                df = df.drop(columns=columns)
                st.write("Dataframe after One-hot Encoding", df)

        if st.sidebar.checkbox("Standardization"):
            st.subheader("Standardization")
            columns_to_standardize = st.multiselect("Select columns for standardization", df.select_dtypes(include=['float64', 'int64']).columns)
            if columns_to_standardize:
                scaler = StandardScaler()
                df[columns_to_standardize] = scaler.fit_transform(df[columns_to_standardize])
                st.write("Dataframe after Standardization", df)

        if st.sidebar.checkbox("Normalization"):
            st.subheader("Normalization")
            columns_to_normalize = st.multiselect("Select columns for normalization", df.select_dtypes(include=['float64', 'int64']).columns)
            if columns_to_normalize:
                scaler = MinMaxScaler()
                df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
                st.write("Dataframe after Normalization", df)
        
        new_df_name = st.text_input("Enter name for the new DataFrame", "Modified DataFrame")
        if st.sidebar.button("Save manipulated DataFrame") and new_df_name:
            st.session_state.dataframes[new_df_name] = df.copy()
            st.success(f"DataFrame saved as {new_df_name}")

        if st.sidebar.button("Save DataFrame to file"):
            save_path = st.text_input("Enter the path to save the DataFrame", "manipulated_dataframe.csv")
            df.to_csv(save_path, index=False)
            st.success(f"DataFrame saved to {save_path}")

# Function to do feature engineering like mapping, aggregation, converting column to date and further converting date to separate columns
def feature_engineering():
    st.sidebar.header("Feature Engineering")
    
    # Select DataFrame
    selected_df = st.selectbox("Select DataFrame", st.session_state.dataframes.keys())
    df = st.session_state.dataframes[selected_df].copy()
    
    # Group by and aggregate options
    if st.sidebar.checkbox("Group By and Aggregate"):
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
    if st.sidebar.checkbox("Map Values"):
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
    
    # Extracting date features
    if st.sidebar.checkbox("Extract Date Features"):
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
    # Convert a column to datetime
    if st.sidebar.checkbox("Convert Column to Datetime"):
        st.subheader("Convert Column to Datetime")
        column_to_convert = st.selectbox("Select column to convert to datetime", df.columns)
        if column_to_convert:
            try:
                df[column_to_convert] = pd.to_datetime(df[column_to_convert])
                st.write(f"Column '{column_to_convert}' successfully converted to datetime.")
                st.write(df.head())
            except Exception as e:
                st.error(f"Error converting column to datetime: {e}")

    # Save the modified DataFrame
    new_df_name = st.text_input("Enter name for the new DataFrame", "Feature_Engineered_DataFrame")
    if st.sidebar.button("Save manipulated DataFrame") and new_df_name:
        st.session_state.dataframes[new_df_name] = df.copy()
        st.success(f"DataFrame saved as {new_df_name}")

    if st.sidebar.button("Save DataFrame to file"):
        save_path = st.text_input("Enter the path to save the DataFrame", "feature_engineered_dataframe.csv")
        df.to_csv(save_path, index=False)
        st.success(f"DataFrame saved to {save_path}")

# Function to display graphs
def display_graphs():
    st.sidebar.header("Graphs")
    
    selected_df = st.sidebar.selectbox("Select DataFrame for Graphs", st.session_state.dataframes.keys())
    df = st.session_state.dataframes.get(selected_df)
    
    if df is not None and st.sidebar.checkbox("Show graphs"):
        st.subheader("Graphs")
        graph_type = st.sidebar.selectbox("Choose graph type", ["Histogram", "Boxplot", "Violin Plot", "Correlation Heatmap", "Scatter Plot with Best Fit Line"])
        columns = st.sidebar.multiselect("Choose columns for graph", df.columns)
        
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
            hue_column = st.sidebar.selectbox("Select column for hue", [None] + list(df.columns))
            palette = st.sidebar.selectbox("Select palette", ["deep", "muted", "bright", "pastel", "dark", "colorblind"])

            st.write(f"Scatter Plot for {columns[0]} vs {columns[1]} with Best Fit Line")
            plt.figure(figsize=(10, 4))
            
            if hue_column and hue_column != "None":
                sns.scatterplot(x=df[columns[0]], y=df[columns[1]], hue=df[hue_column], palette=palette)
            else:
                sns.scatterplot(x=df[columns[0]], y=df[columns[1]])

            sns.regplot(x=df[columns[0]], y=df[columns[1]], scatter=False, color='red')
            st.pyplot(plt)

# Function for hypothesis testing
def hypothesis_testing():
    st.sidebar.header("Hypothesis Testing")
    selected_df = st.sidebar.selectbox("Select DataFrame for Hypothesis Testing", st.session_state.dataframes.keys())
    df = st.session_state.dataframes.get(selected_df)

    if df is not None and st.sidebar.checkbox("Perform hypothesis testing"):
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
            "load_data",
            "display_basic_info",
            "dataframe_modification",
            "display_graphs",
            "hypothesis_testing",
            "machine_learning",
            "execute_code",
            "delete_dataframe",
            "feature_engineering",
            "label_encode_columns"
        ]
        selected_function = st.selectbox("Select a function to view its source code", function_list)
        
        if selected_function:
            function_source_code = inspect.getsource(globals()[selected_function])
            st.code(function_source_code, language='python')
    else:
        # Show a message if the password is incorrect or not entered
        if entered_password:
            st.warning("Incorrect password. Please try again.")

# Function to enter custom code and execute it
def execute_code():
    
    # Dropdown to select which DataFrame to use
    selected_df_name = st.selectbox("Select DataFrame for Code Execution", st.session_state.dataframes.keys())
    original_df = st.session_state.dataframes.get(selected_df_name)
    
    # Input area for code to be executed
    code = st.text_area("Type your Python code here", height=400)
    
    
    
    if original_df is None:
        st.warning("No DataFrame selected or found in session state.")
        return

    exec_locals = {"df": original_df.copy()}  # Initialize exec_locals with a copy of the selected DataFrame

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
                st.write(result)

            # Check if 'df' in exec_locals was updated or created
            if "df" in exec_locals and isinstance(exec_locals["df"], pd.DataFrame):
                st.session_state.modified_df = exec_locals["df"]
                st.write("Modified DataFrame:")
                st.write(st.session_state.modified_df)
                st.write(f"Modified DataFrame Shape: {st.session_state.modified_df.shape}")
            else:
                st.warning("No valid DataFrame found in the executed code.")

        except Exception as e:
            st.error(f"Error: {e}")

    # Option to save the modified DataFrame with a new name
    if "modified_df" in st.session_state:
        new_name = st.text_input("Enter new DataFrame name", "")
        if st.button("Save DataFrame") and new_name:
            if new_name in st.session_state.dataframes:
                st.warning(f"DataFrame name '{new_name}' already exists. Please choose a different name.")
            else:
                st.session_state.dataframes[new_name] = st.session_state.modified_df.copy()
                st.success(f"DataFrame saved as '{new_name}' successfully!")
                st.experimental_rerun()  # Force re-run to refresh UI components

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
        st.write(f"{name}: {df.shape}")

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
            st.experimental_rerun()  # Refresh the app to reflect changes
        else:
            st.warning("Selected DataFrame not found.")

# Function for machine learning
def machine_learning():
    st.sidebar.header("Machine Learning")
    
    selected_df = st.sidebar.selectbox("Select DataFrame for Machine Learning", st.session_state.dataframes.keys())
    df = st.session_state.dataframes.get(selected_df)
    
    if df is not None and st.sidebar.checkbox("Perform machine learning"):
        st.subheader("Machine Learning")
        
        problem_type = st.selectbox("Choose problem type", ["Regression", "Classification", "Clustering", "Unsupervised Learning"])
        
        target_column = st.selectbox("Select target column", df.columns)
        
        all_feature_columns = [col for col in df.columns if col != target_column]
        
        # Display feature columns selection
        feature_columns = st.multiselect("Select feature columns", all_feature_columns, default=all_feature_columns)
        
        test_size = st.slider("Select test size", 0.1, 0.5, 0.2)
        random_state = st.slider("Select random state", 0, 100, 42)
        
        X = df[feature_columns]
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
                
        elif problem_type == "Unsupervised Learning":
            unsupervised_models = {
                "PCA": PCA(),
                "t-SNE": TSNE(),
                "Autoencoder": Sequential([
                    Dense(64, activation='relu', input_shape=(X.shape[1],)),
                    Dense(32, activation='relu'),
                    Dense(64, activation='relu'),
                    Dense(X.shape[1], activation='sigmoid')
                ])
            }
            selected_models = st.multiselect("Select unsupervised learning models", list(unsupervised_models.keys()), default=list(unsupervised_models.keys()))
            
            if "Autoencoder" in selected_models and tune_nn:
                epochs = st.slider("Autoencoder - Epochs", 1, 100, 50)
                batch_size = st.slider("Autoencoder - Batch Size", 16, 256, 32)
                hyperparameters["Autoencoder"] = {"epochs": epochs, "batch_size": batch_size}
                
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
                        elif name == "Autoencoder":
                            model.compile(optimizer='adam', loss='mean_squared_error')
                            model.fit(X, X, epochs=hyperparameters["Autoencoder"]["epochs"], 
                                      batch_size=hyperparameters["Autoencoder"]["batch_size"], 
                                      shuffle=True, 
                                      validation_split=0.2)
                            encoded_data = model.predict(X)
                            st.write("Autoencoder encoded data:")
                            st.write(encoded_data)
            
            except Exception as e:
                st.error(f"An error occurred during model training: {e}")

# Main starts here
def main():
    initialize_session_state()
    st.title("Machine Learning Tool")

    st.sidebar.title("Navigation")
    options = ["Upload Data", "Data Information", "Dataframe Modification","Feature Engineering", "Graphs", "Hypothesis Testing", "Machine Learning", "View Source Code", "Execute Code","Delete DataFrame"]
    choice = st.sidebar.radio("Choose an option", options)

    if choice == "Upload Data":
        st.sidebar.header("Upload your data")
        uploaded_file = st.sidebar.file_uploader("Upload a CSV, Excel, or JSON file", type=["csv", "xlsx", "xls", "json"])
        
        if uploaded_file:
            df = load_data(uploaded_file)
            st.session_state.dataframes["Original"] = df.copy()
            st.success("File uploaded successfully!")
            st.write("DataFrame Preview")
            st.write(df.head())

    elif choice == "Data Information":
        display_basic_info()

    elif choice == "Dataframe Modification":
        if "Original" in st.session_state.dataframes:
            dataframe_modification()
        else:
            st.warning("Please upload a dataset first.")

    elif choice == "Feature Engineering":
        feature_engineering()

    elif choice == "Graphs":
        display_graphs()

    elif choice == "Hypothesis Testing":
        hypothesis_testing()

    elif choice == "Machine Learning":
        machine_learning()

    elif choice == "View Source Code":
        show_source_code()

    elif choice == "Execute Code":
        execute_code()
    elif choice == "Delete DataFrame":
        delete_dataframe()

if __name__ == "__main__":
    main()
