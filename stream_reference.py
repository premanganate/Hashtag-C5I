import streamlit as st
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from PIL import Image
import pickle
import re
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from catboost import CatBoostClassifier
from tqdm import tqdm
import time
import nbimporter
from fpdf import FPDF

import re
import random
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


import shap
import spacy
import xgboost as xgb
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSequenceClassification, AutoTokenizer

st.set_page_config(
    page_title="Survey Response Quality Analyzer",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .flagged {
        background-color: #ffcccc;
        padding: 10px;
        border-radius: 5px;
    }
    .unflagged {
        background-color: #ccffcc;
        padding: 10px;
        border-radius: 5px;
    }
    .header-style {
        font-size: 30px;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
    }
    .subheader-style {
        font-size: 20px;
        font-weight: bold;
        color: #424242;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<p class='header-style'>Survey Response Quality Analyzer</p>", unsafe_allow_html=True)
st.markdown("""
This app analyzes survey responses to detect low-quality submissions based on:
- AI-Generated Content
- Copy-Paste Detection
- Relevance to Context
- Gibberish Detection

Upload your Excel file to begin analysis.
""")

if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'flagged_responses' not in st.session_state:
    st.session_state.flagged_responses = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'models_initialized' not in st.session_state:
    st.session_state.models_initialized = False

@st.cache_resource
def load_models():
    bert_model = SentenceTransformer("paraphrase-MiniLM-L3-v2") 
    ai_tokenizer = AutoTokenizer.from_pretrained("roberta-base-openai-detector")
    ai_model = AutoModelForSequenceClassification.from_pretrained("roberta-base-openai-detector")
    
    try:
        with open('catboost_model.pkl', 'rb') as f:
            cat_model = pickle.load(f)
    except:
        st.warning("Model file not found. Please upload a model file.")
        cat_model = None
    
    try:
        proto_model = ProtoNet(64)
        proto_model.load_state_dict(torch.load("protonet.pth"))
        proto_model.eval()
    except:
        st.warning("Model file not found. Please upload a model file.")
        proto_model = None
    
    return bert_model, ai_tokenizer, ai_model, cat_model, proto_model

def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text) 
    text = text.lower().strip() 
    return text

def calculate_entropy(text):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return 0  # Return low entropy for empty text
    vectorizer = CountVectorizer()
    try:
        X = vectorizer.fit_transform([text])
    except ValueError: 
        return 0  
    probs = np.array(X.toarray()[0], dtype=np.float32)
    if np.sum(probs) == 0:
        return 0 
    probs = probs / np.sum(probs)
    entropy = -np.sum(probs * np.log2(probs))
    return entropy

def check_relevance_batch(responses, expected_context,bert_model):
    context_embedding = bert_model.encode(expected_context, convert_to_tensor=True)
    
    # Encode all responses in a batch
    response_embeddings = bert_model.encode(responses, convert_to_tensor=True)
    
    # Compute cosine similarities in one operation
    similarities = util.pytorch_cos_sim(response_embeddings, context_embedding).cpu().numpy()

    return similarities #> 0.25  # Returns a NumPy array of similarity scores

def detect_ai_generated(text,ai_tokenizer,ai_model):
    if not isinstance(text, str) or text.strip() == "":
        return False

    inputs = ai_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    with torch.no_grad():
        outputs = ai_model(**inputs)

    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    ai_probability = probabilities[0][1].item()  # Assuming 1 is the AI-generated class

    return ai_probability #> 0.5  # Adjust this threshold as needed

def process_df(sheet_name,training_path):
    df = pd.read_excel(training_path,sheet_name=sheet_name)
    d = {}
    cols = df.columns
    
    for i in range(len(cols)):
        if cols[i].startswith("Unnamed"): 
            d[cols[i]] = df.iloc[0, i]  
    df = df.rename(columns = d)
    df.drop([0],axis=0,inplace=True)
    d = {}
    for i in range(len(cols)):
        if cols[i].startswith("Q"): 
            d[cols[i]] = cols[i][0:cols[i].index(' ')]
    df = df.rename(columns=d)
    df = df.fillna("")
    df['time_taken'] = (df['End Date'] - df['Start Date']).dt.total_seconds()
    df.insert(3, 'time_taken', df.pop('time_taken'))
    df['OE_Quality_Flag'] = 0.0
    return df


def detect_copy_paste(df,bert_model):
    similarity_results = []
    
    text_columns = ['Q16A.','Q16B.']
    
    for idx, row in df.iterrows():
        try:
            responses = row[text_columns].astype(str).tolist()
            
            embeddings = bert_model.encode(responses, convert_to_tensor=True)
            
            cosine_scores = util.pytorch_cos_sim(embeddings, embeddings)
            
            n = len(responses)
            upper_triangle_values = [cosine_scores[i][j].item() 
                                   for i in range(n) for j in range(i+1, n)]
            
        except Exception as e:
            print(f"Error processing responder {idx}: {str(e)}")
            upper_triangle_values = [0.0] * (len(text_columns) * (len(text_columns)-1)) // 2
            
        similarity_results.append(upper_triangle_values)
    
    question_pairs = [f"sim_{text_columns[i]}_{text_columns[j]}" 
                     for i in range(len(text_columns)) for j in range(i+1, len(text_columns))]
    
    return pd.DataFrame(similarity_results, columns=question_pairs, index=df.index)


def binary_encode_dataframe(df, categorical_cols):
    binary_encoded_dfs = []
    bin_cols = {
        'Q2.': ['Male', 'Female', 'Transgender', 'Non-binary / Non-conforming', 'Other:', 'Prefer not to disclose'], 
        'Q3.': ['Big city', 'Small city', 'Suburban / just outside of a city', 'Small Town', 'Rural / distant or remote'], 
        'Q4.': ['Less than $20,000','$20,000 to $29,999','$30,000 to $39,999','$40,000 to $49,999','$50,000 to $59,999','$60,000 to $69,999','$70,000 to $79,999','$80,000 to $89,999','$90,000 to $99,999','$100,000 to $149,999','$150,000 or more'], 
        'Q9.': ['Very Irrelevant','Somewhat Irrelevant','Neutral','Somewhat Relevant','Completely Relevant To Me'], 
        'Q10.': ['Extremely Unappealing','Somewhat Unappealing','Neither Appealing nor Unappealing','Somewhat Appealing','Extremely Appealing'], 
        'Q11.': ['Not Different at All','Slightly Different','Somewhat Different','Very Different','Extremely Different'], 
        'Q12.': ['Not at all Believable','Not Very Believable','Somewhat Believable','Very Believable'], 
        'Q13.': ['Very Inexpensive','Somewhat Inexpensive','About Average','Somewhat Expensive','Very Expensive'], 
        'Q14.': ['Definitely Would Not Buy','Probably Would Not Buy','May or May Not Buy','Probably Would Buy','Definitely Would Buy'], 
        'Q15.': ['Never','Less Often Than Every 6 Months','Once Every 6 Months','Once Every 2-3 Months','Once a Month','Once Every 2 to 3 Weeks','Once a Week','Two or Three Days/Nights a Week','Daily/Nightly'], 
        'Q17.': ['Be used in addition to products you are currently using','Sometimes replace products you are currently drinking','Totally replace products you are currently drinking']
    }
    for col in categorical_cols:
        print(col)
        unique_vals = bin_cols[col]
        mapping = {val: idx for idx, val in enumerate(unique_vals)}  # Label Encoding
        max_bits = len(bin(len(mapping) - 1)[2:])  # Find required bits

        binary_encoded = df[col].map(mapping).apply(lambda x: list(map(int, bin(int(x))[2:].zfill(max_bits))))
        binary_df = pd.DataFrame(binary_encoded.tolist(), columns=[f"{col}bin{i}" for i in range(max_bits)], index=df.index)
        binary_encoded_dfs.append(binary_df)

    # Concatenate the binary-encoded features with the original numerical features
    df_encoded = pd.concat([df.drop(columns=categorical_cols)] + binary_encoded_dfs, axis=1)
    
    return df_encoded

def binary_encoded_dataframe(df, categorical_cols):
    binary_encoded_dfs = []
    
    for col in categorical_cols:
        unique_vals = df[col].unique()
        mapping = {val: idx for idx, val in enumerate(unique_vals)}  # Label Encoding
        max_bits = len(bin(len(mapping) - 1)[2:])  # Find required bits

        binary_encoded = df[col].map(mapping).apply(lambda x: list(map(int, bin(x)[2:].zfill(max_bits))))
        binary_df = pd.DataFrame(binary_encoded.tolist(), columns=[f"{col}bin{i}" for i in range(max_bits)], index=df.index)
        binary_encoded_dfs.append(binary_df)

    # Concatenate the binary-encoded features with the original numerical features
    df_encoded = pd.concat([df.drop(columns=categorical_cols)] + binary_encoded_dfs, axis=1)
    
    return df_encoded

def generate_features(df, column_names, column_groups, feature_types=["mean", "var", "interaction",'ratio']):
    new_features = pd.DataFrame(index=df.index)  # Keep index aligned with original DF
    
    group_name = '_'.join(column_names)
    if "mean" in feature_types:
        new_features[f"{group_name}_Mean"] = df[column_groups].mean(axis=1)
    
    if "var" in feature_types:
        new_features[f"{group_name}_Var"] = df[column_groups].var(axis=1)
    
    if "interaction" in feature_types:
        col1 = column_groups[0]
        col2 = column_groups[1]
        new_features[f"{col1}_x_{col2}"] = df[col1] * df[col2]
    if "ratio" in feature_types:
        col1 = column_groups[0]
        col2 = column_groups[1]
        new_features[f"{col1}_/_{col2}"] = df[col1] / df[col2]

    return new_features

def apply_feature_engg(df, column_names, feature_types):
    cmp = {'Q1.':'Age','Q2.':'Gender','Q3.':'Urban/Rural','Q4.':'Income','Q9.':'Relevance','Q10.':'Appeal','Q11.':'Differentiation','Q12.':'Believability','Q13.':'Price','Q14.':'Purchase_Intent','Q15.':'Drinking_Frequency'}
    cmp = {v: k for k, v in cmp.items()}
    print(column_names)
    res_df = generate_features(df, column_names, [cmp[i] for i in column_names], feature_types)
    return pd.concat([df,res_df], axis=1)
    

class ProtoNet(nn.Module):
    def __init__(self, input_dim):
        super(ProtoNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(64, 32)  # Smaller embedding
        )
        # Register prototypes as a buffer
        self.register_buffer("prototypes", torch.zeros(2,32))  

    def forward(self, x):
        return self.fc(x)

    def compute_and_store_prototypes(self, support_X, support_y):
        """
        Computes class prototypes and stores them in the model as a buffer.
        """
        with torch.no_grad():
            embeddings = self(support_X)  # Get embeddings

            unique_classes = torch.unique(support_y)
            prototypes = torch.stack([
                embeddings[support_y.squeeze(-1) == c].mean(dim=0) for c in unique_classes
            ])

        self.prototypes = prototypes  # Store prototypes inside the buffer

    def get_prototypes(self):
        """
        Returns stored prototypes.
        """
        if self.prototypes.numel() == 0:
            raise ValueError("Prototypes have not been computed. Call compute_and_store_prototypes() first.")
        return self.prototypes
    
def compute_prototypes(support_X, support_y, model):
    with torch.no_grad():
        embeddings = model(support_X)  # Get embeddings
    unique_classes = torch.unique(support_y)
    prototypes = torch.stack([embeddings[support_y.squeeze() == c].mean(0) for c in unique_classes])
    return prototypes

def compute_loss(prototypes, query_X, query_y, model):
    query_embeddings = model(query_X)  # Get query embeddings
    distances = torch.cdist(query_embeddings, prototypes)  # Euclidean distance
    log_probs = F.log_softmax(-distances, dim=1)  # Convert distances to log probs
    if query_y == None:
        return None, log_probs
    loss = F.nll_loss(log_probs, query_y.squeeze().long())  # Negative Log-Likelihood Loss
    return loss, log_probs



def preprocess_data(df, bert_model, ai_tokenizer, ai_model, options,path=None):
    # Process the dataframe
    df = process_df(1,path)
    
    df = apply_feature_engg(df, ['Relevance','Appeal','Differentiation','Believability'],['mean','var'])
    df = apply_feature_engg(df, ['Price','Purchase_Intent'],['mean','interaction'])
    df = apply_feature_engg(df, ['Drinking_Frequency','Purchase_Intent'],['mean','var'])
    df = apply_feature_engg(df, ['Income','Price'],['mean','var'])

    df['Q16A.'] = df['Q16A.'].apply(lambda x: clean_text(x))
    df['Q16B.'] = df['Q16B.'].apply(lambda x: clean_text(x))

    # Process based on selected options
    with st.spinner("Processing data..."):
        progress_bar = st.progress(0)
        total_steps = sum(options.values())
        current_step = 0

        st.info("Detecting gibberish content...")
        df['gibberish_16A'] = df['Q16A.'].apply(lambda x: calculate_entropy(x))
        df['gibberish_16B'] = df['Q16B.'].apply(lambda x: calculate_entropy(x))
        current_step += 1
        progress_bar.progress(current_step / total_steps)

        st.info("Detecting AI-generated content...")
        df['ai_generated_16A'] = df['Q16A.'].apply(lambda x: detect_ai_generated(x,ai_tokenizer,ai_model))
        df['ai_generated_16B'] = df['Q16B.'].apply(lambda x: detect_ai_generated(x,ai_tokenizer,ai_model))
        df['ai_generated_16A'] = df['ai_generated_16A'].astype('float')
        df['ai_generated_16B'] = df['ai_generated_16B'].astype('float')
        current_step += 1
        progress_bar.progress(current_step / total_steps)

        st.info("Checking response relevance...")
        q16a_question = "What is the most important thing you LIKE about the shown concept? This can include anything you would want kept for sure or aspects that might drive you to buy or try it."
        q16b_question = "What is the most important thing you DISLIKE about the shown concept? This can include general concerns, annoyances, or any aspects of the product that need fixing for this to be more appealing to you."
        df["is_relevant_16A"] = check_relevance_batch(df["Q16A."].tolist(), q16a_question,bert_model)
        df["is_relevant_16B"] = check_relevance_batch(df["Q16B."].tolist(), q16b_question,bert_model)
        current_step += 1
        progress_bar.progress(current_step / total_steps)

        st.info("Detecting copy-paste content...")
        resdf = detect_copy_paste(df[['Q16A.', 'Q16B.']],bert_model)
        df = pd.concat([df,resdf], axis = 1)    
        current_step += 1
        progress_bar.progress(current_step / total_steps)
    return df

def df_to_tensors(df):
    print(df.columns)
    if 'OE_Quality_Flag' not in df.columns:
        X = torch.tensor(df.values, dtype=torch.float32)
        y = torch.tensor(pd.DataFrame({'a':123}).values, dtype=torch.long)
        return X
    X = torch.tensor(df.drop(columns=['OE_Quality_Flag']).values, dtype=torch.float32)
    y = torch.tensor(df['OE_Quality_Flag'].values, dtype=torch.long)
    return X, y

def predict_protonet_ordered(model, prototypes, query_X):
    model.eval()  # Set model to evaluation mode

    # Compute class prototypes from support set
    prototypes = prototypes

    # Compute class probabilities for the query set
    _, log_probs = compute_loss(prototypes, query_X, None, model)

    # Get predictions (keeping order)
    preds = log_probs.argmax(dim=1).cpu().numpy()

    return preds

def predict_quality(df,cat_model,proto_model,path):

    cols_to_convert = []
    for i in df.columns[:29]:
        if df[i].dtype == "object":
            cols_to_convert.append(i)

    cols_to_drop = []
    for i in df.columns[30:36]:
        if df[i].dtype == "object":
            cols_to_drop.append(i)

    df = df.fillna('')
    dft = df.drop(cols_to_drop+['Start Date', 'End Date', 'Unique ID','time_taken'], axis=1, inplace=False)

    dftt = process_df(0,path)
    cols_to_binarize=[]
    cols_to_binarize.extend(['Q2.','Q3.','Q4.'])
    for i in range(9,16):
        col = 'Q'+str(i)+'.'
        cols_to_binarize.append(col)   
    cols_to_binarize.append('Q17.')
    for i in cols_to_binarize:
        dft[i] = dftt[i]
    print(cols_to_binarize)
    dft = binary_encode_dataframe(dft,cols_to_binarize)
    #test_predictions = cat_model.predict(dft)
    #dft['OE_Quality_Flag'] = test_predictions
    #dft['OE_Quality_Flag'] = dft.pop('OE_Quality_Flag')
    #dft.insert(34,column='Q2.bin2',value=0)
    scaler = StandardScaler()
    dfts = scaler.fit_transform(dft) 
    dft = pd.DataFrame(dfts, columns=dft.columns, index=dft.index)
    prototypes = proto_model.get_prototypes()
    query_X, _ = df_to_tensors(dft)
    # Get ordered predictions for test data
    test_predictions = predict_protonet_ordered(proto_model, prototypes, query_X) 
    dft['OE_Quality_Flag'] = test_predictions
    dft['OE_Quality_Flag'] = dft.pop('OE_Quality_Flag')
    return dft

def filter_responses(df, thresholds):
    # Apply thresholds to filter responses
    mask = (
        (df['ai_generated_score'] >= thresholds['ai_threshold']) |
        (df['copy_paste_score'] >= thresholds['copy_paste_threshold']) |
        (df['relevance_score'] <= thresholds['relevance_threshold']) |
        (df['gibberish_score'] >= thresholds['gibberish_threshold']) |
        (df['predicted_flag'] >= thresholds['model_threshold'])
    )
    
    flagged_df = df[mask].copy()
    return flagged_df

def generate_report(df, flagged_df, thresholds, options):
    # Create a PDF report
    pdf = FPDF()
    pdf.add_page()
    
    # Add title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Survey Response Quality Analysis Report", ln=True, align="C")
    pdf.ln(5)
    
    # Add date and time
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(5)
    
    # Add summary
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Summary:", ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 10, f"Total responses analyzed: {len(df)}", ln=True)
    pdf.cell(0, 10, f"Flagged responses: {len(flagged_df)} ({len(flagged_df)/len(df)*100:.2f}%)", ln=True)
    pdf.ln(5)
    
    # Add threshold information
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Applied Thresholds:", ln=True)
    pdf.set_font("Arial", "", 10)
    
    if options['ai_detection']:
        pdf.cell(0, 10, f"AI-Generated Content: {thresholds['ai_threshold']}", ln=True)
    if options['copy_paste']:
        pdf.cell(0, 10, f"Copy-Paste Detection: {thresholds['copy_paste_threshold']}", ln=True)
    if options['relevance']:
        pdf.cell(0, 10, f"Relevance Score: {thresholds['relevance_threshold']}", ln=True)
    if options['gibberish']:
        pdf.cell(0, 10, f"Gibberish Detection: {thresholds['gibberish_threshold']}", ln=True)
    pdf.cell(0, 10, f"Model Prediction Threshold: {thresholds['model_threshold']}", ln=True)
    pdf.ln(5)
    
    # Add flagged responses summary
    if not flagged_df.empty:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Top Flagged Responses:", ln=True)
        pdf.set_font("Arial", "", 10)
        
        # Display top 10 flagged responses or less if fewer exist
        for idx, row in flagged_df.head(10).iterrows():
            pdf.cell(0, 10, f"Response ID: {row['Unique ID']}", ln=True)
            pdf.multi_cell(0, 10, f"Q16A: {row['Q16A.'][:100]}...")
            pdf.multi_cell(0, 10, f"Q16B: {row['Q16B.'][:100]}...")
            pdf.cell(0, 10, f"AI Score: {row['ai_generated_score']:.2f}, Copy-Paste: {row['copy_paste_score']:.2f}, Relevance: {row['relevance_score']:.2f}, Gibberish: {row['gibberish_score']}", ln=True)
            pdf.ln(5)
    
    # Generate PDF as bytes
    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    return pdf_bytes

def get_download_link(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:text/csv;base64,{b64}" download="{filename}.csv">Download CSV</a>'
    return href

def main():
    # Sidebar for options
    st.sidebar.markdown("<p class='subheader-style'>Analysis Options</p>", unsafe_allow_html=True)
    
    options = {
        'ai_detection': st.sidebar.checkbox("AI-Generated Content Detection", value=True),
        'copy_paste': st.sidebar.checkbox("Copy-Paste Detection", value=True),
        'relevance': st.sidebar.checkbox("Relevance Check", value=True),
        'gibberish': st.sidebar.checkbox("Gibberish Detection", value=True)
    }
    
    # Load models if not already loaded
    if not st.session_state.models_initialized:
        with st.spinner("Initializing models..."):
            st.session_state.bert_model, st.session_state.ai_tokenizer, st.session_state.ai_model, st.session_state.catboost_model,st.session_state.proto_model = load_models()
            st.session_state.models_initialized = True
    
    # File uploader
    uploaded_file = st.file_uploader("Upload Excel file with survey responses", type=["xlsx", "xls"])
    
    if uploaded_file is not None:
        # Read the uploaded file
        try:
            df = pd.read_excel(uploaded_file)
            st.success(f"File uploaded successfully! Found {len(df)} responses.")
            
            # Process the data
            with st.spinner("Processing data..."):
                processed_df = preprocess_data(
                    df, 
                    st.session_state.bert_model, 
                    st.session_state.ai_tokenizer, 
                    st.session_state.ai_model, 
                    options,
                    path=uploaded_file
                )
                
                # Run model prediction
                processed_df = predict_quality(processed_df, st.session_state.catboost_model,st.session_state.proto_model,uploaded_file)
                
                # Store processed data in session state
                st.session_state.processed_data = processed_df
                
                st.success("Data processing complete!")
            
            # Add button to download the complete processed dataframe
            # Then in your app
            if processed_df is not None:
                st.markdown(get_download_link(processed_df, "processed_data"), unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    # If data has been processed, show filtering options
    if st.session_state.processed_data is not None:
        st.markdown("<p class='subheader-style'>Filter Flagged Responses</p>", unsafe_allow_html=True)
        
        # Sliders for thresholds
        col1, col2 = st.columns(2)
        
        with col1:
            ai_threshold = st.slider(
                "AI-Generated Content Threshold", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.7, 
                step=0.05,
                disabled=not options['ai_detection']
            )
            
            copy_paste_threshold = st.slider(
                "Copy-Paste Threshold", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.8, 
                step=0.05,
                disabled=not options['copy_paste']
            )
        
        with col2:
            relevance_threshold = st.slider(
                "Relevance Threshold (lower = less relevant)", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.3, 
                step=0.05,
                disabled=not options['relevance']
            )
            
            gibberish_threshold = st.slider(
                "Gibberish Threshold", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.5, 
                step=0.05,
                disabled=not options['gibberish']
            )
        
        model_threshold = st.slider(
            "Model Prediction Threshold", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.5, 
            step=0.05
        )
        
        # Apply filters
        thresholds = {
            'ai_threshold': ai_threshold,
            'copy_paste_threshold': copy_paste_threshold,
            'relevance_threshold': relevance_threshold,
            'gibberish_threshold': gibberish_threshold,
            'model_threshold': model_threshold
        }
        
        if st.button("Apply Filters"):
            with st.spinner("Filtering responses..."):
                flagged_df = filter_responses(st.session_state.processed_data, thresholds)
                st.session_state.flagged_responses = flagged_df
                
                if flagged_df.empty:
                    st.warning("No responses were flagged with the current thresholds.")
                else:
                    st.success(f"Found {len(flagged_df)} flagged responses out of {len(st.session_state.processed_data)}.")
        
        # Display flagged responses
        if st.session_state.flagged_responses is not None and not st.session_state.flagged_responses.empty:
            st.markdown("<p class='subheader-style'>Flagged Responses</p>", unsafe_allow_html=True)
            
            # Display flagged responses in an expandable section
            with st.expander("View Flagged Responses", expanded=True):
                st.dataframe(st.session_state.flagged_responses[['Unique ID', 'Q16A.', 'Q16B.', 'ai_generated_score', 'copy_paste_score', 'relevance_score', 'gibberish_score', 'predicted_flag']])
            
            # Generate and download report
            if st.button("Generate Report"):
                with st.spinner("Generating report..."):
                    pdf_bytes = generate_report(
                        st.session_state.processed_data, 
                        st.session_state.flagged_responses, 
                        thresholds, 
                        options
                    )
                    
                    st.markdown(
                        get_download_link(pdf_bytes, "survey_quality_report.pdf"), 
                        unsafe_allow_html=True
                    )
            
            # Download flagged responses as Excel
            if st.button("Download Flagged Responses as Excel"):
                with st.spinner("Preparing Excel file..."):
                    excel_buffer = BytesIO()
                    st.session_state.flagged_responses.to_excel(excel_buffer, index=False)
                    excel_bytes = excel_buffer.getvalue()
                    
                    st.markdown(
                        get_download_link(excel_bytes, "flagged_responses.xlsx"), 
                        unsafe_allow_html=True
                    )

if __name__ == "__main__":
    main()
