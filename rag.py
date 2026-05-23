import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_groq import ChatGroq

from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")

from langchain.schema import Document
from pdf2image import convert_from_path
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

poppler_path = r"C:\poppler\Library\bin"


def process_pdfs(uploaded_files):

    all_documents = []

    for uploaded_file in uploaded_files:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_pdf_path = temp_file.name

        # Normal PDF extraction
        loader = PDFPlumberLoader(temp_pdf_path)
        documents = loader.load()

        extracted_text = ""

        for doc in documents:
            extracted_text += doc.page_content.strip()

        # If scanned PDF → OCR
        if len(extracted_text) < 50:

            print("Scanned PDF detected. Running OCR...")

            images = convert_from_path(
                temp_pdf_path,
                dpi=150,
                poppler_path=r"C:\Users\BHASHINI\Downloads\Release-26.02.0-0\poppler-26.02.0\Library\bin"
            )

            ocr_text = ""

            for img in images:
                img = img.convert("L")
                text = pytesseract.image_to_string(
                        img
                        # config="--psm 6"
                        )
                ocr_text += text

            documents = [
                Document(
                    page_content=ocr_text,
                    metadata={"source": uploaded_file.name}
                )
            ]

        # Store metadata
        for doc in documents:
            doc.metadata["source"] = uploaded_file.name

        all_documents.extend(documents)

        # Delete temp file
        os.remove(temp_pdf_path)

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(all_documents)

    print("TOTAL CHUNKS:", len(chunks))

    if len(chunks) == 0:
        raise ValueError("No text could be extracted from PDFs")

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector DB
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore

def create_chain(vectorstore):

    # Streamlit Cloud secrets
    api_key = os.getenv("GROQ_API_KEY")

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.1-8b-instant"
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="question",
        output_key="answer"
    )

    prompt_template = """
    You are a strict AI placement interview trainer.

    You MUST follow the relevancy rules before answering.

    STRICT RULES:

    1. Use ONLY the uploaded PDF context.
    If the answer is not present in the PDF, say:
    "The uploaded PDF does not contain information about this."
    and then answer using your own knowledge.

    2. If the user says:
    - stop
    - end
    - quit
    - exit
    - stop asking questions

    then stop asking questions politely.

    3. If user chooses OTHER company,
    ignore relevancy rule.

    RELEVANCY RULE:

    - Important topics for {company} are:
      {topics}

    - Check whether the user's question belongs to the important topics.

    - If irrelevant, first say:
      "This topic is irrelevant to the important topics of {company}."

    - Then still answer the question.

    INTERVIEW FLOW:

    - Ask one interview question at a time.
    - Evaluate user's answer.

    Context:
    {context}

    User:
    {question}

    AI:
    """

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question", "company", "topics"]
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(
        search_kwargs={"k": 2}
    ),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": PROMPT},
    return_source_documents=True
    )
    return qa_chain