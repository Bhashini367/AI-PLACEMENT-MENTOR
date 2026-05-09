import streamlit as st

from rag import process_pdfs, create_chain
from companydata import company_topics

st.set_page_config(page_title="AI Placement Mentor")

st.title("AI Placement Mentor")

# Company Selection
selected_company = st.selectbox(
    "Select Target Company",
    list(company_topics.keys())
)

# Show Important Topics
st.subheader("Important Topics")

for topic in company_topics[selected_company]:
    st.write(f"• {topic}")

# Multi PDF Upload
uploaded_files = st.file_uploader(
    "Upload Study Material PDFs",
    type="pdf",
    accept_multiple_files=True
)

# Process PDFs
if uploaded_files:

    with st.spinner("Processing PDFs..."):

        vectorstore = process_pdfs(uploaded_files)

        qa_chain = create_chain(vectorstore)

        st.session_state.qa_chain = qa_chain

    st.success("PDFs processed successfully!")

# Question Section
st.subheader("Ask Questions")

user_question = st.text_input("Enter your question")

if user_question:

    if "qa_chain" not in st.session_state:

        st.warning("Please upload PDFs first.")

    else:

        # Basic relevance check
        relevant = any(
            topic.lower() in user_question.lower()
            for topic in company_topics[selected_company]
        )

        if not relevant:

            st.warning(
                f"This topic may be irrelevant to {selected_company} interview topics."
            )

        # Invoke chain
        response = st.session_state.qa_chain.invoke(
            {
                "question": user_question,
                "company": selected_company,
                "topics": ", ".join(company_topics[selected_company])
            }
        )

        # Show Answer
        st.write("### Answer")
        st.write(response["answer"])

        # Show Sources
        st.write("### Sources")

        source_docs = response["source_documents"]

        shown_sources = set()

        for doc in source_docs:

            source = doc.metadata.get("source", "Unknown File")
            page = doc.metadata.get("page", "Unknown Page")

            source_key = f"{source}-{page}"

            if source_key not in shown_sources:

                shown_sources.add(source_key)

                st.write(f"📄 {source} — Page {page}")