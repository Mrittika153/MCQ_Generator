import os
import pdfplumber
import docx
from fpdf import FPDF
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.schema import AIMessage

# -----------------------------
# Configuration
# -----------------------------
UPLOAD_FILE = r"C:\Users\USER\MCQ_Genaretor\chapter1.pdf"  # Your file
NUM_QUESTIONS = 5
OUTPUT_FOLDER = "results"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -----------------------------
# LangChain-Groq setup
# -----------------------------
llm = ChatGroq(
    api_key="YOUR API KEY",  # Replace with your actual API key
    model="MODEL NAME", # Replace with your actual model name
    temperature=0.0
)

mcq_prompt = PromptTemplate(
    input_variables=["context", "num_questions"],
    template="""
You are an AI assistant generating multiple-choice questions (MCQs) from the text below:

Text:
{context}

Create {num_questions} MCQs. Each must include:
- A question
- 4 options (A, B, C, D)
- Correct answer at the end

Format exactly:
## MCQ
Question: ...
A) ...
B) ...
C) ...
D) ...
Correct Answer: X
"""
)

# Modern LangChain "RunnableSequence"
mcq_chain = mcq_prompt | llm

# -----------------------------
# Text extraction
# -----------------------------
def extract_text(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = file_path.rsplit('.', 1)[-1].lower()

    if ext == "pdf":
        with pdfplumber.open(file_path) as pdf:
            return ''.join([page.extract_text() or "" for page in pdf.pages])
    elif ext == "docx":
        doc = docx.Document(file_path)
        return ' '.join([para.text for para in doc.paragraphs])
    elif ext == "txt":
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        raise ValueError("Unsupported file type")

# -----------------------------
# Save MCQs as TXT
# -----------------------------
def save_txt(mcqs, filename):
    path = os.path.join(OUTPUT_FOLDER, filename)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(mcqs)
    print(f"Saved text to {path}")

# -----------------------------
# Save MCQs as PDF
# -----------------------------
def save_pdf(mcqs, filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 6, mcqs)
    path = os.path.join(OUTPUT_FOLDER, filename)
    pdf.output(path)
    print(f"Saved PDF to {path}")

# -----------------------------
# Main MCQ generation
# -----------------------------
def main():
    try:
        text = extract_text(UPLOAD_FILE)
        if not text.strip():
            print("No text extracted from the file.")
            return
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    print("Generating MCQs...")

    # Generate MCQs
    result = mcq_chain.invoke({"context": text, "num_questions": NUM_QUESTIONS})

    # Extract text from AIMessage
    if isinstance(result, AIMessage):
        mcqs = result.content.strip()
    else:
        mcqs = str(result).strip()

    # Save outputs
    base_name = os.path.basename(UPLOAD_FILE).rsplit('.', 1)[0]
    save_txt(mcqs, f"generated_mcqs_{base_name}.txt")
    save_pdf(mcqs, f"generated_mcqs_{base_name}.pdf")

    print("\nMCQ Generation Complete!")

# -----------------------------
# Run script
# -----------------------------
if __name__ == "__main__":
    main()
