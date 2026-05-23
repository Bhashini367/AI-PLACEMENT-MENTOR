import os
import sys
import tempfile
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PDFPlumberLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    CSVLoader
)

# ───────── TESSERACT SETUP (for scanned PDFs) ─────────
try:
    import pytesseract
    if sys.platform == "win32":
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
except ImportError:
    pytesseract = None

POPPLER_PATH = r"C:\Users\BHASHINI\Downloads\Release-26.02.0-0\poppler-26.02.0\Library\bin" if sys.platform == "win32" else None

# ───────── OCR FOR SCANNED PDFs ─────────
def _ocr_pdf(path, filename):
    if pytesseract is None:
        return []
    try:
        from pdf2image import convert_from_path
        images = convert_from_path(path, dpi=150, poppler_path=POPPLER_PATH)
        docs = []
        for page_num, img in enumerate(images, start=1):
            text = pytesseract.image_to_string(img.convert("L")).strip()
            if text:
                docs.append(Document(page_content=text, metadata={"source": filename, "page": page_num}))
        return docs
    except:
        return []

# ───────── LOAD FILE BASED ON TYPE ─────────
def _load_file(path, filename):
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pdf":
        pages = PDFPlumberLoader(path).load()
        text = "".join(p.page_content for p in pages).strip()
        if not text:
            return _ocr_pdf(path, filename)
        for i, page in enumerate(pages):
            page.metadata["source"] = filename
            page.metadata["page"] = i + 1
        return pages

    elif ext == ".txt":
        pages = TextLoader(path, encoding="utf-8").load()
        for i, page in enumerate(pages):
            page.metadata["source"] = filename
            page.metadata["page"] = i + 1
        return pages

    elif ext == ".docx":
        pages = Docx2txtLoader(path).load()
        for i, page in enumerate(pages):
            page.metadata["source"] = filename
            page.metadata["page"] = i + 1
        return pages

    elif ext == ".pptx":
        pages = UnstructuredPowerPointLoader(path).load()
        for i, page in enumerate(pages):
            page.metadata["source"] = filename
            page.metadata["page"] = i + 1
        return pages

    elif ext == ".csv":
        pages = CSVLoader(path).load()
        for i, page in enumerate(pages):
            page.metadata["source"] = filename
            page.metadata["page"] = i + 1
        return pages

    else:
        print(f"Unsupported file type: {ext}")
        return []

# ───────── MAIN FUNCTION ─────────
def process_pdfs(uploaded_files):
    all_pages = []

    for file in uploaded_files:
        ext = os.path.splitext(file.name)[1].lower()

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        try:
            pages = _load_file(tmp_path, file.name)
            all_pages.extend(pages)
        except Exception as e:
            print(f"Error reading {file.name}: {e}")
        finally:
            os.remove(tmp_path)

    # Split each page into chunks, keeping source and page number
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = []
    for page in all_pages:
        for chunk_text in splitter.split_text(page.page_content):
            chunks.append(Document(
                page_content=chunk_text,
                metadata={
                    "source": page.metadata.get("source", "unknown"),
                    "page": page.metadata.get("page", 1)
                }
            ))

    return chunks