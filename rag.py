import os
import pickle
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

INDEXES_DIR = "indexes"
os.makedirs(INDEXES_DIR, exist_ok=True)

# Load embedding model once
_embeddings = None
def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return _embeddings

# ───────── BUILD RETRIEVER ─────────
def get_retriever(chunks, session_id=None):
    embeddings = _get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    faiss = vectorstore.as_retriever(search_kwargs={"k": 5})
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 5
    retriever = EnsembleRetriever(retrievers=[bm25, faiss], weights=[0.4, 0.6])

    # Save to disk
    if session_id:
        vectorstore.save_local(f"{INDEXES_DIR}/{session_id}")
        with open(f"{INDEXES_DIR}/{session_id}_bm25.pkl", "wb") as f:
            pickle.dump(chunks, f)

    return retriever

# ───────── LOAD RETRIEVER FROM DISK ─────────
def load_retriever(session_id):
    faiss_path = f"{INDEXES_DIR}/{session_id}"
    bm25_path = f"{INDEXES_DIR}/{session_id}_bm25.pkl"

    if not os.path.exists(faiss_path) or not os.path.exists(bm25_path):
        return None

    try:
        embeddings = _get_embeddings()
        vectorstore = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
        faiss = vectorstore.as_retriever(search_kwargs={"k": 5})
        with open(bm25_path, "rb") as f:
            chunks = pickle.load(f)
        bm25 = BM25Retriever.from_documents(chunks)
        bm25.k = 5
        return EnsembleRetriever(retrievers=[bm25, faiss], weights=[0.4, 0.6])
    except:
        return None

# ───────── DELETE RETRIEVER FROM DISK ─────────
def delete_retriever(session_id):
    import shutil
    faiss_path = f"{INDEXES_DIR}/{session_id}"
    bm25_path = f"{INDEXES_DIR}/{session_id}_bm25.pkl"
    if os.path.exists(faiss_path):
        shutil.rmtree(faiss_path)
    if os.path.exists(bm25_path):
        os.remove(bm25_path)

# ───────── LLM ─────────
def get_llm():
    return ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0.3,
        max_tokens=1024
    )

# ───────── CHECK RELEVANCE ─────────
def is_relevant(docs, question, llm):
    if not docs:
        return "NO"

    text = " ".join(d.page_content for d in docs).strip()
    if len(text) < 200:
        return "NO"

    prompt = f"""Is the context below relevant to answer the question?
Reply with one word only: YES, PARTIAL, or NO.

Question: {question}
Context: {text[:800]}"""

    try:
        result = llm.invoke(prompt).content.strip().upper()
        if "PARTIAL" in result: return "PARTIAL"
        if "YES" in result: return "YES"
        return "NO"
    except:
        return "YES"

<<<<<<< HEAD
# ───────── BUILD CONTEXT STRING ─────────
def get_context(docs):
    parts = []
    total = 0
    for doc in docs:
        chunk = f"[{doc.metadata.get('source')} p.{doc.metadata.get('page')}]\n{doc.page_content}"
        if total + len(chunk) > 4000:
            break
        parts.append(chunk)
        total += len(chunk)
    return "\n\n".join(parts)
=======
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