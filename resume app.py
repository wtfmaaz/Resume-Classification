# IMPORT LIBRARIES
import re
import pandas as pd
import docx2txt
import streamlit as st
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import pdfplumber
from textblob import Word

# Preprocessing function
def preprocess(sentence):
    sentence = str(sentence).lower()
    sentence = sentence.replace('{html}', "")
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url = re.sub(r'http\S+', '', cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)
    filtered_words = [w for w in tokens if len(w) > 2 and w not in stopwords.words('english')]
    lemma_words = [Word(w).lemmatize() for w in filtered_words]
    return " ".join(lemma_words)

# Streamlit app title and header
st.title('RESUME CLASSIFICATION')
st.markdown('<style>h1{color: Purple;}</style>', unsafe_allow_html=True)
st.subheader('Welcome to Resume Classification App')

# Extract skills from resume text
def extract_skills(resume_text):
    try:
        # Load skills from the CSV file
        data = pd.read_csv(r"Cleaned_Resumes.csv")
        if data.empty:
            st.error("The CSV file is empty or not loaded properly.")
            return []

        skills = list(data.columns.values)  # Assuming skills are stored as column names
        skillset = set()

        # Tokenizing the resume text
        tokens = resume_text.split()

        # Check for one-gram skills (single-word skills)
        for token in tokens:
            if token.lower() in [skill.lower() for skill in skills]:
                skillset.add(token.lower())

        # Return the list of unique skills (capitalized for readability)
        return [skill.capitalize() for skill in skillset]

    except Exception as e:
        st.error(f"Error while extracting skills: {e}")
        return []

# Function to extract text from files
def getText(filename):
    try:
        if filename.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return docx2txt.process(filename)
        else:
            with pdfplumber.open(filename) as pdf:
                content = ''
                for page in pdf.pages:
                    content += page.extract_text()
                return content
    except Exception as e:
        st.error(f"Error processing file: {filename.name}. {e}")
        return ""

# Display function
def display(doc_file):
    if doc_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return docx2txt.process(doc_file)
    else:
        with pdfplumber.open(doc_file) as pdf:
            return pdf.pages[0].extract_text()

# Initialize data storage
file_type = pd.DataFrame([], columns=['Uploaded File', 'Predicted Profile', 'Skills'])
filename = []
predicted = []
skills = []

# Load the model and vectorizer
import pickle as pk
model = pk.load(open(r'modelDT.pkl', 'rb'))
Vectorizer = pk.load(open(r'vector.pkl', 'rb'))

# File uploader
upload_file = st.file_uploader('Upload Your Resumes', type=['docx', 'pdf'], accept_multiple_files=True)

for doc_file in upload_file:
    if doc_file is not None:
        filename.append(doc_file.name)

        # Preprocess the resume text for prediction
        resume_text = preprocess(display(doc_file))
        prediction = model.predict(Vectorizer.transform([resume_text]))[0]
        predicted.append(prediction)

        # Extract skills
        extracted_text = getText(doc_file)
        skill_list = extract_skills(extracted_text)
        skills.append(skill_list)

        # Debug: Print extracted skills for each file
        st.write(f"Skills extracted from {doc_file.name}: {skill_list}")

# Display results
if len(predicted) > 0:
    file_type['Uploaded File'] = filename
    file_type['Skills'] = [", ".join(skill) if skill else "No skills identified" for skill in skills]
    file_type['Predicted Profile'] = predicted
    st.table(file_type.style.format())

# Dropdown for filtering profiles
select = ['PeopleSoft', 'SQL Developer', 'React JS Developer', 'Workday']
st.subheader('Select as per Requirement')
option = st.selectbox('Fields', select)

if option == 'PeopleSoft':
    st.table(file_type[file_type['Predicted Profile'] == 'PeopleSoft'])
elif option == 'SQL Developer':
    st.table(file_type[file_type['Predicted Profile'] == 'SQL Developer'])
elif option == 'React JS Developer':
    st.table(file_type[file_type['Predicted Profile'] == 'React JS Developer'])
elif option == 'Workday':
    st.table(file_type[file_type['Predicted Profile'] == 'Workday'])
