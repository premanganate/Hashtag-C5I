import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64
from datetime import datetime
import os

# Set page configuration
st.set_page_config(
    page_title="Survey Response Quality Validator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4B5563;
        margin-bottom: 0.5rem;
    }
    .card {
        background-color: #F9FAFB;
        border-radius: 0.5rem;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: white;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        text-align: center;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #6B7280;
        font-size: 0.8rem;
    }
    .stButton button {
        background-color: #1E3A8A;
        color: white;
        font-weight: bold;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.markdown('<div class="main-header">Survey Response Quality Validator</div>', unsafe_allow_html=True)
st.markdown('Analyze open-ended survey responses to detect low-quality or fraudulent submissions')

# Sidebar for configuration
with st.sidebar:
    st.markdown('<div class="sub-header">Configuration</div>', unsafe_allow_html=True)
    
    # Analysis thresholds section
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Quality Thresholds")
    
    quality_threshold = st.slider(
        "Overall Quality Score Threshold", 
        0, 100, 60, 
        help="Responses with quality scores below this threshold will be flagged"
    )
    
    min_response_length = st.slider(
        "Minimum Response Length", 
        0, 50, 10, 
        help="Minimum number of characters for a valid response"
    )
    
    min_word_diversity = st.slider(
        "Minimum Word Diversity", 
        0.0, 1.0, 0.5, 
        help="Minimum ratio of unique words to total words (higher = more diverse vocabulary)"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Analysis options section
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Analysis Options")
    
    enable_length_check = st.checkbox("Check Response Length", value=True)
    enable_diversity_check = st.checkbox("Check Vocabulary Diversity", value=True)
    enable_relevance_check = st.checkbox("Check Response Relevance", value=True)
    enable_consistency_check = st.checkbox("Check Cross-Question Consistency", value=True)
    enable_bot_detection = st.checkbox("Enable Bot/AI Detection", value=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Help section
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Help")
    st.markdown("""
    **How to use this tool:**
    1. Upload your survey data Excel file
    2. Select the appropriate data sheet
    3. Adjust quality thresholds if needed
    4. Click "Analyze Response Quality"
    5. Review the analysis results
    6. Download the report for your records
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# Main content area
main_col1, main_col2 = st.columns([2, 1])

with main_col1:
    # File upload section
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Upload Survey Data")
    
    uploaded_file = st.file_uploader(
        "Upload survey data file (Excel format)", 
        type=["xlsx", "xls"],
        help="Upload Excel file containing survey responses"
    )
    
    if uploaded_file:
        try:
            # Data preview section
            st.success("File uploaded successfully")
            df_dict = pd.read_excel(uploaded_file, sheet_name=None)
            sheet_names = list(df_dict.keys())
            
            selected_sheet = st.selectbox(
                "Select Data Sheet", 
                options=sheet_names,
                help="Choose the sheet containing survey response data"
            )
            
            df = df_dict[selected_sheet]
            
            # Show data preview
            st.subheader("Data Preview")
            st.dataframe(df.head(5), use_container_width=True)
            
            # Show basic stats
            row_count = len(df)
            col_count = len(df.columns)
            
            stats_col1, stats_col2, stats_col3 = st.columns(3)
            with stats_col1:
                st.metric("Total Responses", f"{row_count:,}")
            with stats_col2:
                st.metric("Total Variables", f"{col_count:,}")
            with stats_col3:
                st.metric("Open-Ended Questions", "3")
            
            # Analysis button
            if st.button("Analyze Response Quality", type="primary", use_container_width=True):
                with st.spinner("Analyzing responses..."):
                    # Placeholder for analysis process
                    st.info("Analysis would run here with real implementation")
                    
                    # Placeholder results
                    st.subheader("Analysis Results")
                    
                    # Metrics
                    result_col1, result_col2, result_col3 = st.columns(3)
                    with result_col1:
                        st.metric("Average Quality Score", "78.5")
                    with result_col2:
                        st.metric("Flagged Responses", f"{int(row_count * 0.15):,} ({15}%)")
                    with result_col3:
                        st.metric("Processing Time", "8.2 sec")
                    
                    # Placeholder for quality distribution chart
                    fig = px.histogram(
                        x=[65, 70, 75, 80, 85, 90, 95, 100] * 10, 
                        nbins=20,
                        labels={"x": "Quality Score", "y": "Number of Responses"},
                        title="Distribution of Quality Scores"
                    )
                    fig.add_vline(x=quality_threshold, line_dash="dash", line_color="red",
                                annotation_text=f"Threshold ({quality_threshold})")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Sample flagged responses
                    st.subheader("Sample Flagged Responses")
                    flagged_df = pd.DataFrame({
                        "Respondent ID": ["R-2338", "R-567", "R-139"],
                        "Q16A (Likes)": ["Don't drink it", "I don't like anything", "I don't like it."],
                        "Q16B (Dislikes)": ["Don't drink it", "Nothing", "It looks bad."],
                        "Quality Score": [45, 30, 25],
                        "Issues": ["Inconsistent", "Generic", "Too short"]
                    })
                    st.dataframe(flagged_df, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)

with main_col2:
    # Quality insights section
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Quality Insights")
    
    if uploaded_file:
        # Common issues chart placeholder
        issues_data = pd.DataFrame({
            "Issue": ["Too Short", "Generic Response", "Inconsistent", "Off-topic", "Bot-like"],
            "Count": [45, 37, 25, 18, 10]
        })
        
        fig = px.bar(
            issues_data, 
            y="Issue", 
            x="Count", 
            orientation='h',
            title="Common Quality Issues",
            color="Count",
            color_continuous_scale="Reds"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Demographics of flagged responses
        st.subheader("Flagged Response Demographics")
        
        # Age distribution of flagged responses
        age_data = pd.DataFrame({
            "Age Group": ["21-30", "31-40", "41-50", "51-60", "61-65"],
            "Percent Flagged": [22, 15, 10, 12, 18]
        })
        
        fig = px.bar(
            age_data,
            x="Age Group",
            y="Percent Flagged",
            title="Flagging Rate by Age Group",
            color="Percent Flagged",
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("Upload a file to see quality insights")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Export options section
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Export Results")
    
    if uploaded_file:
        col1, col2 = st.columns(2)
        with col1:
            st.button("Download PDF Report", disabled=not uploaded_file)
        with col2:
            st.button("Export Flagged Data", disabled=not uploaded_file)
    else:
        st.info("Upload and analyze a file to enable exports")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">Survey Response Quality Validator | Created for C5I Hackathon 2025</div>', unsafe_allow_html=True)
