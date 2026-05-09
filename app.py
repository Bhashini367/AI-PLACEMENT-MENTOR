import streamlit as st
from rag import process_pdf, create_chain
from companydata import company_topics

st.set_page_config(page_title="AI Placement Mentor")

st.title("AI Placement Mentor")
selected_company = st.selectbox(
    "Select Target Company",
    list(company_topics.keys())
)
st.subheader("Important Topics")

for topic in company_topics[selected_company]:
    st.write(f"• {topic}")
uploaded_file = st.file_uploader(
    "Upload Study Material PDF",
    type="pdf"
)

if uploaded_file:

    with st.spinner("Processing PDF..."):
        vectorstore = process_pdf(uploaded_file)
        qa_chain = create_chain(vectorstore)

        st.session_state.qa_chain = qa_chain

    st.success("PDF processed successfully!")

st.subheader("Ask Questions")

user_question = st.text_input("Enter your question")

if user_question:

    if "qa_chain" not in st.session_state:
        st.warning("Please upload a PDF first.")

    else:

        relevant = any(
            topic.lower() in user_question.lower()
            for topic in company_topics[selected_company]
        )

        if not relevant:
             st.warning(
            f"This topic may be irrelevant to {selected_company} interview topics."
            )

        response = st.session_state.qa_chain.invoke(
        {
            "question": user_question,
            "company": selected_company,
            "topics": ", ".join(company_topics[selected_company])
        }
        )
        st.write("### Answer")
        st.write(response["answer"])