

"""#### ***7. Model Deployment***

##### ***7.1 Pickle File***
"""

import pickle
filename = 'modelDT.pkl'
pickle.dump(model_DT,open(filename,'wb'))

import pickle
filename = 'vector.pkl'
pickle.dump(tfidf_vector,open(filename,'wb'))

# Streamlit app
st.title("Resume Classification App")
st.write("Upload a resume or type text to classify it into predefined categories.")

# Sidebar for training the model
if st.sidebar.button("Train Model"):
    train_model()

# Resume upload or text input
resume_input = st.text_area("Paste Resume Text Here")

if st.button("Classify Resume"):
    if resume_input.strip():
        category = classify_resume(resume_input)
        st.success(f"The resume is classified as: {category}")
    else:
        st.warning("Please provide resume text to classify.")

"""### _______________________________________________________________

# ***PROJECT CONCLUSION***

* In this project we have worked on resume classification, we were given resumes of different categories.Our goal was to classify resumes according to their categories and extract information from the resume through screening.

* We have performed necessary data exploration, text preprocessing, text visualization,feature extraction using count vectorizer,bag of words and explored different entities in the text using Named Entity Recognition.

* Using the spacy library we have found the most common words occuring in the corpus,built worclouds for the whole corpus and particular category from the corpus, we observed the top 10 most occuring words in corpus constituting bigrams and trigrams.

* After complete pipeline of Data exploration we initiated model building by encoding our category labels using label encoder,after encoding we have converted our data into vectors using TFIDF Vectorizer creating a sparse matrix.

* we split our resume data into train and test data with ratio 80:20,80% as training set and 20% as test set,with this we are ready to train our model.

* After splitting the data we have performed model training with classification algorithms and ensemble techniques ,using metrics like KFOLD- validation, confusion Matrix, classification report and accuracy score, we append all our categories of algorithms all at once.

* After observing the confusion matrix and classification report we observed that Random Forest handled well by giving the least missclassification and a excellent equvivalent accuracy with training and testing validation.we got a good score of 1 for precision, recall, F1- score and accuracy.

* After perfoming the evaluation matrix with optimal algorithms with selecting our random forest as final model we performed our predictions on our test data and we got final predictions and classified resumes according to their categories by mapping according to encoded labels.

* After classifying resumes according to categories, we created a predictive system with resume parser techniques and extracted information from the input resume and fit in a dataframe.

* To conclude resume classification or resume screening is a technique to reduce the human effort in human resource management and extract resume with higher level accuracy,In this project we found out the importance of resume screening and how well an ATS system functions.
"""
