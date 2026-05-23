import streamlit as st
import uuid
from rag import get_retriever, load_retriever, delete_retriever, get_llm, is_relevant, get_context
from pdf_processing import process_pdfs
from history import load_history, save_history
from company_data import company_topics
from evaluate import evaluate_rag

st.set_page_config(page_title="AI Placement Mentor", layout="wide")

# ───────── LOAD HISTORY ONCE ─────────
if "sessions" not in st.session_state:
    st.session_state.sessions = load_history()

if "current_id" not in st.session_state:
    st.session_state.current_id = st.session_state.sessions[-1]["id"]

if "retrievers" not in st.session_state:
    st.session_state.retrievers = {}

if "llm" not in st.session_state:
    st.session_state.llm = get_llm()

# ───────── GET CURRENT SESSION ─────────
def get_session():
    for s in st.session_state.sessions:
        if s["id"] == st.session_state.current_id:
            return s
    return st.session_state.sessions[-1]

session = get_session()

# ───────── SIDEBAR ─────────
with st.sidebar:
    st.title("💬 Chats")

    if st.button("＋ New Chat", use_container_width=True):
        new = {"id": str(uuid.uuid4()), "title": "New Chat", "messages": [], "company": "Google", "progress": {}}
        st.session_state.sessions.append(new)
        st.session_state.current_id = new["id"]
        save_history(st.session_state.sessions)
        st.rerun()

    st.divider()

    to_delete = None
    for s in reversed(st.session_state.sessions):
        is_active = s["id"] == session["id"]
        label = ("▸ " if is_active else "  ") + s["title"][:28]
        col1, col2 = st.columns([5, 1])
        with col1:
            if st.button(label, key=f"s_{s['id']}", use_container_width=True):
                st.session_state.current_id = s["id"]
                st.rerun()
        with col2:
            if st.button("🗑", key=f"d_{s['id']}"):
                to_delete = s["id"]

    if to_delete:
        st.session_state.sessions = [s for s in st.session_state.sessions if s["id"] != to_delete]
        st.session_state.retrievers.pop(to_delete, None)
        delete_retriever(to_delete)
        if not st.session_state.sessions:
                st.session_state.sessions.append({"id": str(uuid.uuid4()), "title": "New Chat", "messages": [], "company": "Google", "progress": {}})
        st.session_state.current_id = st.session_state.sessions[-1]["id"]
        save_history(st.session_state.sessions)
        st.rerun()

# ───────── MAIN PAGE ─────────
st.title("🎯 AI Placement Mentor")
st.caption("Prepare smarter for placement interviews with AI-powered mentoring.")

# Company selector
companies = list(company_topics.keys())
company = st.selectbox(
    "Select Target Company",
    options=companies,
    index=companies.index(session.get("company", "Google"))
)

# Reset progress if company changed
if session.get("company") != company:
    session["company"] = company
    session["progress"] = {}
    save_history(st.session_state.sessions)

topics = company_topics.get(company, [])

# ───────── TOPIC ROADMAP ─────────
if topics:
    st.subheader(f"📌 {company} Prep Roadmap")
    st.caption("Check off topics as you finish studying them.")

    progress_changed = False
    cols_per_row = 3
    for i in range(0, len(topics), cols_per_row):
        cols = st.columns(cols_per_row)
        for col, topic in zip(cols, topics[i:i+cols_per_row]):
            with col:
                current = session["progress"].get(topic, False)
                checked = st.checkbox(topic, value=current, key=f"cb_{session['id']}_{topic}")
                if checked != current:
                    session["progress"][topic] = checked
                    progress_changed = True

    if progress_changed:
        save_history(st.session_state.sessions)



# ───────── PDF UPLOAD ─────────
st.subheader("📄 Upload Study Material")
st.caption("Supported: PDF, TXT, DOCX, PPTX, CSV")

# Load retriever from disk if not in memory
if session["id"] not in st.session_state.retrievers:
    saved = load_retriever(session["id"])
    if saved:
        st.session_state.retrievers[session["id"]] = saved

retriever = st.session_state.retrievers.get(session["id"])

if retriever:
    st.info("✓ PDFs already loaded for this session.")

files = st.file_uploader("Upload files", accept_multiple_files=True, type=["pdf", "txt", "docx", "pptx", "csv"], key=f"up_{session['id']}")

if files:
    with st.spinner("Processing files..."):
        chunks = process_pdfs(files)
        retriever = get_retriever(chunks, session_id=session["id"])
        st.session_state.retrievers[session["id"]] = retriever
    st.success(f"✓ {len(files)} file(s) uploaded successfully!")

st.divider()

# ───────── CHAT HISTORY ─────────
for msg in session["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ───────── CHAT INPUT ─────────
q = st.chat_input(f"Ask your {company} interview question...")

if q:
    session["messages"].append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.write(q)

    llm = st.session_state.llm

    # Check if user wants practice questions
    question_triggers = ["generate questions", "give me questions", "test me", "quiz me", "practice questions", "ask me", "interview questions"]
    wants_questions = any(t in q.lower() for t in question_triggers)
    checked_topics = [t for t, v in session.get("progress", {}).items() if v]

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            # ── Practice question generation ──
            if wants_questions:
                if checked_topics:
                    prompt = (
                        f"You are a {company} interview coach.\n"
                        f"Generate 5 interview questions for each of these topics: {', '.join(checked_topics)}.\n"
                        f"Format with topic as heading and questions as numbered list.\n"
                        f"Focus on what {company} typically asks."
                    )
                    answer = llm.invoke(prompt).content
                    final = f"📝 **Practice Questions for {company}**\n*Topics: {', '.join(checked_topics)}*\n\n{answer}"
                else:
                    final = "⚠️ **No topics checked yet!**\n\nPlease check off the topics you have studied in the roadmap above first."

            # ── RAG answer (PDF uploaded) ──
            elif retriever:
                docs = retriever.invoke(q)
                relevant = is_relevant(docs, q, llm)
                context = get_context(docs)
                company_context = f"\n\nNote: Answer in the context of {company} interviews."

                if relevant == "NO":
                    answer = llm.invoke(q + company_context).content
                    final = f"⚠️ **Not found in PDF**\n\n🤖 **AI Answer**\n\n{answer}"

                elif relevant == "PARTIAL":
                    pdf_ans = llm.invoke(f"Answer only from this:\n{context}\n\nQuestion: {q}").content
                    ai_ans = llm.invoke(q + company_context).content
                    final = f"📘 **From PDF**\n\n{pdf_ans}\n\n---\n\n🤖 **AI Answer**\n\n{ai_ans}"

                else:
                    pdf_ans = llm.invoke(f"Answer using only this:\n{context}\n\nQuestion: {q}").content
                    # Build sources
                    file_pages = {}
                    for d in docs:
                        src = d.metadata.get("source", "?")
                        page = d.metadata.get("page", 1)
                        file_pages.setdefault(src, set()).add(page)
                    sources = "\n".join(f"- **{src}** → {', '.join(f'p.{p}' for p in sorted(pgs))}" for src, pgs in file_pages.items())
                    final = f"📘 **From PDF**\n\n{pdf_ans}\n\n---\n\n📚 **Sources**\n\n{sources}"
                    # Save for evaluation
                    evaluate_rag(
                        questions     = [q],
                        answers       = [final],
                        contexts      = [[d.page_content for d in docs]],
                        ground_truths = [""]
                    )

            # ── Pure AI answer (no PDF) ──
            else:
                company_context = f"\n\nNote: Answer in the context of {company} interviews."
                answer = llm.invoke(q + company_context).content
                final = f"🤖 **AI Answer**\n\n{answer}"

        st.markdown(final)

    session["messages"].append({"role": "assistant", "content": final})

    if session["title"] == "New Chat":
        session["title"] = f"{company} – {q[:30]}"

    save_history(st.session_state.sessions)
    st.rerun()