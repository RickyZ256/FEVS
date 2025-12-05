# Import the necessary libraries
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap

# --- App Configuration ---
st.set_page_config(layout="wide")
st.title("FEVS Dashboard: 3-Year Analysis")

# --- 1. Data Loading ---
# Restoring st.cache_data now that we've found the issue
@st.cache_data
def load_data(file_path):
    """Loads all data from the specified Excel file and fixes Population cols."""
    try:
        responses_df = pd.read_excel(file_path, sheet_name="fevs_sample_data_3FYs_Set3")
        questions_df = pd.read_excel(file_path, sheet_name="Index-Qns-Map", skiprows=8)
        population_df = pd.read_excel(file_path, sheet_name="Population")
        
        # Get the current column names
        current_cols = population_df.columns.tolist()
        
        # We know from debugging that the columns are bad. We will just force them.
        # We will create a new list of names.
        new_cols_list = [
            'Data Set',  # Position 0
            2023,        # Position 1
            2024,        # Position 2
            2025,        # Position 3
            'Random.Seed'# Position 4
        ]
        
        # Create a dictionary to map old names to new names
        # e.g., { 'Data Set ': 'Data Set', 'nan': 2023, 'nan.1': 2024, ... }
        # This is the robust way to rename, handling duplicates.
        rename_map = {current_cols[i]: new_cols_list[i] for i in range(len(new_cols_list))}
        
        population_df = population_df.rename(columns=rename_map)
        
        # Also clean the 'Data Set' column name just in case
        if 'Data Set' in population_df.columns:
             population_df = population_df.rename(columns={'Data Set': 'Data Set'})
        # --- END OF FIX ---

        return responses_df, questions_df, population_df
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found. Please update the path.")
        return None, None, None
    except ValueError as e:
        st.error(f"Error reading a sheet: {e}. Please check your sheet names.")
        return None, None, None

# IMPORTANT: Make sure this path is correct
excel_filename = "fevs_sample_data_3FYs_DataSet_3.xlsx"
responses_df, questions_df, population_df = load_data(excel_filename)

# --- 2. Data Processing ---
@st.cache_data # Restoring cache
def process_data(responses_df, questions_df):
    """Processes the raw data and creates the final dataframe for analysis."""
    if responses_df is None or questions_df is None:
        return None
    
    melted_responses = responses_df.melt(id_vars=['FY', 'Response.ID'], var_name='Question', value_name='Response')
    
    def categorize_response(response):
        if response >= 4: return 'Positive'
        elif response == 3: return 'Neutral'
        else: return 'Negative'
        
    melted_responses['Perception'] = melted_responses['Response'].apply(categorize_response)
    
    perception_counts = melted_responses.groupby(['FY', 'Question', 'Perception']).size().unstack(fill_value=0)
    perception_percentages = perception_counts.div(perception_counts.sum(axis=1), axis=0) * 100
    
    merged_df = pd.merge(perception_percentages.reset_index(), questions_df, left_on='Question', right_on='Item.ID')
    
    if 'Index' in merged_df.columns:
        merged_df.rename(columns={'Index': 'Performance Dimension'}, inplace=True)
    
    return merged_df

# Run the data processing
full_data = process_data(responses_df, questions_df)

# --- Main App Logic ---
if full_data is not None:
    
    # --- Create Tabs ---
    tab1, tab2, tab3 = st.tabs(["Executive Summary", "Interactive Analysis", "Dataset Summary & References"])

    # --- TAB 1: EXECUTIVE SUMMARY ---
    with tab1:
        st.header("Executive Summary (Bottom Line Up Front)")
        
        st.markdown("""
        **Bottom Line Up Front (BLUF):**
        The organization possesses a highly resilient and confident workforce but faces a critical risk regarding employee satisfaction and retention. While employees strongly believe in the quality of their own work (**Performance Confidence**), there is a sharp and concerning decline in **Global Satisfaction**, particularly regarding pay.

        * **Key Strength (Performance Confidence):** The organization's greatest asset is its people. The top-rated area is **Performance Confidence**, with employees nearly unanimously agreeing that they produce high-quality work and contribute positively to the agency's mission.
        * **Critical Weakness (Pay & Leadership Action):** The lowest-rated area is **Global Satisfaction**. Specifically, **Pay Satisfaction** is the lowest-scoring item and shows a negative trend, worsening significantly from 2023 to 2025. Additionally, employees are skeptical that leadership will act on these survey results.
        * **Strategic Recommendation:** To prevent the low satisfaction from eroding the currently high performance, leadership must address the compensation concerns immediately. Furthermore, increasing transparency about how this survey data is used is essential to rebuilding trust.
        """)
        
        st.subheader("Performance Dimension Overview (3-Year Average)")
        
        dimension_avg = full_data.groupby('Performance Dimension')['Positive'].mean().reset_index()
        dimension_avg = dimension_avg.sort_values(by='Positive', ascending=False)
        
        fig_dim, ax_dim = plt.subplots(figsize=(10, 8))
        sns.barplot(ax=ax_dim, data=dimension_avg, x='Positive', y='Performance Dimension', palette='viridis')
        ax_dim.set_title('Average Positive Perception by Performance Dimension', fontsize=16)
        ax_dim.set_xlabel('Average Positive Perception (%)', fontsize=12)
        ax_dim.set_ylabel('')
        ax_dim.set_xlim(0, 100)
        st.pyplot(fig_dim)


    # --- TAB 2: INTERACTIVE ANALYSIS (No changes) ---
    with tab2:
        st.header("Overall Strengths and Areas for Improvement (3-Year Average)")

        three_year_avg = full_data.groupby('Item.Text')['Positive'].mean().reset_index()
        three_year_avg_sorted = three_year_avg.sort_values(by='Positive', ascending=False)
        strengths_3yr = three_year_avg_sorted.head(5)
        areas_to_improve_3yr = three_year_avg_sorted.tail(5).sort_values(by='Positive', ascending=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Top 5 Strengths")
            strengths_3yr['wrapped_labels'] = strengths_3yr['Item.Text'].apply(lambda x: textwrap.fill(x, width=50))
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            sns.barplot(ax=ax1, x='Positive', y='wrapped_labels', data=strengths_3yr, palette='Greens_r')
            st.pyplot(fig1)

        with col2:
            st.subheader("Top 5 Areas for Improvement")
            areas_to_improve_3yr['wrapped_labels'] = areas_to_improve_3yr['Item.Text'].apply(lambda x: textwrap.fill(x, width=50))
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            sns.barplot(ax=ax2, x='Positive', y='wrapped_labels', data=areas_to_improve_3yr, palette='Reds_r')
            st.pyplot(fig2)

        st.header("Detailed Question Analysis")
        st.markdown("---")

        dimension_list = ['All'] + sorted(full_data['Performance Dimension'].dropna().unique())
        selected_dimension = st.selectbox("Step 1: Filter by Performance Dimension", dimension_list)

        if selected_dimension == 'All':
            question_list_df = full_data
        else:
            question_list_df = full_data[full_data['Performance Dimension'] == selected_dimension]
        
        question_list = sorted(question_list_df['Item.Text'].unique())
        selected_question = st.selectbox("Step 2: Select a Question to Analyze", question_list)
        
        if selected_question:
            question_details = full_data[full_data['Item.Text'] == selected_question]
            chart_data = question_details.melt(id_vars=['FY'], value_vars=['Positive', 'Neutral', 'Negative'],
                                               var_name='Perception', value_name='Percentage')
            st.subheader(f"Comparison for: '{selected_question}'")
            fig, ax = plt.subplots(figsize=(10, 6))
            custom_palette = {'Positive': 'green', 'Neutral': 'yellow', 'Negative': 'orange'}
            sns.barplot(ax=ax, data=chart_data, x='FY', y='Percentage', hue='Perception', palette=custom_palette)
            ax.set_title('Perception Percentage by Year', fontsize=16)
            ax.set_xlabel('Fiscal Year', fontsize=12)
            ax.set_ylabel('Percentage (%)', fontsize=12)
            ax.set_ylim(0, 100)
            ax.legend(title='Perception', loc='upper right')
            st.pyplot(fig)
            st.subheader("Data Table")
            st.dataframe(question_details.set_index('FY')[['Positive', 'Neutral', 'Negative']])


    # --- TAB 3: DATASET SUMMARY & REFERENCES ---
    with tab3:
        st.header("Dataset Descriptive Summary")
        
        st.markdown("""
        This analysis is based on the **FEVS Sample Data (DataSet 3)**, provided in a single Excel workbook. The dataset includes:
        * **Survey Responses:** Raw 5-point Likert scale responses for 2023 and 2024.
        * **Question Mapping:** A dictionary (`Index-Qns-Map`) that maps question IDs (e.g., "Q1") to their full text and categorizes them by Performance Dimension.
        * **Population Data:** The total number of respondents for each year.
        """)
        
        if population_df is not None:
            st.subheader("Respondent Population")
            
            # Filter for our dataset (using the cleaned column name 'Data Set')
            pop_data = population_df[population_df['Data Set'] == 3] 

            if not pop_data.empty:
                # --- THIS IS THE FIX ---
                # Change to 3 columns instead of 2
                col1, col2, col3 = st.columns(3) 
                
                # Use INTEGER keys (2023) to access the columns
                if 2023 in pop_data.columns:
                    col1.metric("2023 Respondents", pop_data[2023].values[0])
                else:
                    col1.error("Column 2023 (integer) not found.")
                
                # Use INTEGER key (2024)
                if 2024 in pop_data.columns:
                    col2.metric("2024 Respondents", pop_data[2024].values[0])
                else:
                    col2.error("Column 2024 (integer) not found.")
                
                # NEW: Add the metric for 2025
                if 2025 in pop_data.columns:
                    col3.metric("2025 Respondents", pop_data[2025].values[0])
                else:
                    col3.error("Column 2025 (integer) not found.")
                # --- END OF FIX ---
            else:
                st.warning("Could not find data for 'Data Set 3' in the Population sheet.")

        st.header("References")
        st.markdown("""
        * **Data Source:** OPM (Office of Personnel Management) FEVS Public Data File.
        * **Python Libraries:**
            * **Streamlit:** For creating the interactive web dashboard.
            * **Pandas:** For data loading, manipulation, and analysis.
            * **Seaborn & Matplotlib:** For data visualization.
        """)

else:
    st.error("There was an error loading the data. Please check the file path and sheet names.")
