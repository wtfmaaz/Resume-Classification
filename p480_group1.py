

"""##### ***6.1 Train and Test Accuracy Plot***"""

import matplotlib.pyplot as plt
import numpy as np

rcParams = {'xtick.labelsize':'12','ytick.labelsize':'14','axes.labelsize':'16'}
fig, axe = plt.subplots(1,1, figsize=(12,6), dpi=500)
x_pos = np.arange(len(table))
model_names = ["KNN", "Decision T", "Random F", "SVM ", "Logistic", "Bagging", "Ada Boost", "Grad Boost","Naive Bayes"]

bar1 = plt.bar(x_pos - 0.2, table['Train_Accuracy(%)'], width=0.4, label='Train', color= "Orange")
bar2 = plt.bar(x_pos + 0.2, table['Test_Accuracy(%)'], width=0.4, label='Test', color= "b")
plt.xticks(x_pos, model_names)

plt.xlabel("Classifiers", fontsize = 16, fontweight = 'bold')
plt.ylabel("Accuracy", fontsize = 16, fontweight = 'bold')
plt.title("Model Accuracy Scores", fontsize = 18, fontweight = 'bold')
plt.xticks(rotation = 10)
plt.legend()

for i, bar in enumerate(bar1):
    plt.text(bar.get_x() + bar.get_width()/2 - 0.1, bar.get_height()*1.02,
             s = '{:.2f}%'.format(Train_accuracies[i]), fontsize = 7)

for i, bar in enumerate(bar2):
    plt.text(bar.get_x() + bar.get_width()/2 - 0.1, bar.get_height()*1.02,
             s = '{:.2f}%'.format(Test_accuracies[i]), fontsize = 7)

pylab.rcParams.update(rcParams)
fig.tight_layout()
plt.show()
fig.savefig('IMG\Mod_Acc_Bar', dpi = 500)

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
