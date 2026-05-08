import os
import tempfile

from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_groq import ChatGroq

from langchain.prompts import PromptTemplate

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Load API key from .env
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")

def process_pdf(uploaded_file):

    # Save uploaded PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_pdf_path = temp_file.name

    # Load PDF
    loader = PDFPlumberLoader(temp_pdf_path)
    documents = loader.load()

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    print("TOTAL CHUNKS:", len(chunks))

    # Safety check
    if len(chunks) == 0:
        raise ValueError("No text could be extracted from PDF")

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create vector database
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Delete temporary file
    os.remove(temp_pdf_path)

    return vectorstore


def create_chain(vectorstore):

    api_key = os.getenv("GROQ_API_KEY")
    print("API KEY:", api_key)

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.1-8b-instant"
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": PROMPT}
    )

    prompt_template = """
    You are an AI interview trainer.

    STRICT RULES:
    1. Use ONLY the uploaded PDF context.If the answer is not present in the PDF, say:
    "The uploaded PDF does not contain information about this."and then answer his question with your knowledge.
    2.If the user says:
      - stop
      - end
      - quit
      - exit
      - stop asking questions
      then stop asking interview questions politely and do not ask another question unless the user ask.
    3.if user choose the target company as other then answer the question without considering relevency rule.

    RELEVANCY RULE:
        - Check whether the user's question belongs to the important topics of the selected company.
        - If the question is outside the company's important topics, first say:
        "This topic is irrelevant to the important topics of {company}."
        - Then still answer the question.

    INTERVIEW FLOW:
    - Ask one interview question at a time.
    - Evaluate user's answer.

    Evaluation format:
    1. Verdict:
    Correct / Partially Correct / Incorrect

    2. Explanation

    3. Ideal Answer

    4. Ask next interview question from the PDF.

    Context:
    {context}

    User:
    {question}

    AI:
    """
    PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    combine_docs_chain_kwargs={"prompt": PROMPT}
    )

    return qa_chain