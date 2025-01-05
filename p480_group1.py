


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


