# -*- coding: utf-8 -*-
# Resume Classification Streamlit App

import re
import pandas as pd
import streamlit as st
import spacy
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import PyPDF2
from io import StringIO
import pickle as pk


# Title and Description
st.title('Resume Classification')
st.markdown('<style>h1{color: Purple;}</style>', unsafe_allow_html=True)
st.subheader('Welcome to the Resume Classification App')

# Load Model and Vectorizer
model = pk.load(open('modelDT.pkl', 'rb'))
vectorizer = pk.load(open('vector.pkl', 'rb'))



# Helper function: Extract text from resumes
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(docx_file):
    from docx import Document
    doc = Document(docx_file)
    return "\n".join([para.text for para in doc.paragraphs])

# Extract skills
def extract_skills(resume_text):
    nlp_text = nlp(resume_text)
    tokens = [token.text.lower() for token in nlp_text if not token.is_stop]
    skillset = []

    # Assuming Cleaned_Resumes.csv contains a list of skills
    try:
        skills_df = pd.read_csv("Cleaned_Resumes.csv")
        skills = list(skills_df.columns.values)
        for token in tokens:
            if token in skills:
                skillset.append(token)
    except FileNotFoundError:
        st.error("Skillset file not found. Please upload the 'Cleaned_Resumes.csv'.")
    return list(set(skillset))

# Preprocess text
def preprocess_text(text):
    text = re.sub('<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub('[0-9]+', '', text)  # Remove numbers
    text = text.lower()  # Convert to lowercase
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words("english")]
    return " ".join(words)

# Main logic
# Initialize the lists outside of the loop to avoid the NameError
filename = []
predicted = []
skills = []

# File upload logic
upload_file = st.file_uploader('Upload Your Resumes', type=['docx', 'pdf'], accept_multiple_files=True)

for doc_file in upload_file:
    if doc_file is not None:
        filename.append(doc_file.name)

        # Simulate predicted profile and skills extraction for testing
        # Replace this with actual logic for predictions and skill extraction
        predicted.append("Example Profile")  # Dummy value for prediction
        skills.append(["Example Skill 1", "Example Skill 2"])  # Dummy value for skills

# Now check if the lengths of the lists are the same
if len(filename) == len(predicted) == len(skills):
    # Create DataFrame only if lengths are the same
    file_type = pd.DataFrame({
        'Uploaded File': filename,
        'Predicted Profile': predicted,
        'Skills': skills
    })
    st.table(file_type.style.format())
else:
    st.error("Error: Mismatched lengths in data arrays. Please check your input data.")



            # Preprocess text and predict
            processed_text = preprocess_text(text)
            vectorized_text = vectorizer.transform([processed_text])
            prediction = model.predict(vectorized_text)[0]
            file_data["Predicted Profile"].append(prediction)

            # Extract skills
            skills = extract_skills(text)
            file_data["Skills"].append(", ".join(skills))

        except Exception as e:
            st.error(f"Error processing file {file.name}: {str(e)}")

    # Display results in a table
    results_df = pd.DataFrame(file_data)
    st.table(results_df)

# Add filtering options
select_options = results_df["Predicted Profile"].unique().tolist() if uploaded_files else []
st.subheader("Filter Results by Profile")
if select_options:
    option = st.selectbox("Fields", select_options)
    filtered_df = results_df[results_df["Predicted Profile"] == option]
    st.table(filtered_df)
else:
    st.info("No data available for filtering.")
