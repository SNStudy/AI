import os
import PyPDF2
import openai
from dotenv import load_dotenv

load_dotenv('keys.env')

def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()

def read_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip()

def generate_summary(content):
    client = openai.OpenAI(
        api_key=os.getenv('OPENAI_API_KEY')
    )
    
    prompt = f"""
    
You are an expert document analyst. Your task is NOT to summarize or explain content, 
but to segment the provided text corpus into thematic **sections and subsections**.

The input may consist of one or multiple documents, possibly with inconsistent formatting. 
Each document can have multiple pages, and sections/subsections may span across different pages 
and even reappear in non-contiguous places.

### Rules for Section & Subsection Identification:
1. **Prefer the document’s own structure when possible**:
   - If the introduction, table of contents, or early pages contain an outline or section structure, 
     use those headings, section numbers, and subsection numbers as the primary framework.
   - If explicit numbering exists (e.g., "1. Introduction", "2.1 Data Collection"), follow it exactly.
2. If no clear structure exists, or if gaps appear, infer descriptive section/subsection names based on content flow.
3. A section or subsection may reappear across non-contiguous pages (e.g., "Methods" on pages 2–5 and again on page 7). 
   Always merge them under the same entry.

### Your Tasks:
1. Cover the entire text — every line must belong to exactly one section or subsection.
2. For each **section and subsection**, output:
   - **Section/Subsection Name** (prefer source document titles; infer only if necessary).
   - **Page Range(s)** where it appears.
   - **Line Range(s)** (start → end line per page where it occurs).
   - If a section/subsection spans multiple intervals, list all intervals clearly.

### Section Count Control:
- If the variable `num_sections` is provided (e.g., `num_sections=5`), divide the text into exactly that many **top-level sections**, 
  using the document’s own structure as much as possible. Subsections should still be extracted within them.
- If `num_sections` is empty or not given, determine the most logical number of sections/subsections based on the text and its structure.

### Output Format
Return a structured list ONLY, no summaries or commentary. Example:

## Section Breakdown
- Section: "1. Introduction"
  - Pages: 1–2
  - Lines: 1–38

- Section: "2. Search Algorithms"
  - Pages: 2–7
  - Lines: 39–149
  - Subsections:
    - Subsection: "2.1 Linear Search"
      - Pages: 2–3
      - Lines: 39–72
    - Subsection: "2.2 Binary Search"
      - Pages: 4–5, 7
      - Lines: 73–128 (pp.4–5), 120–149 (p.7)

- Section: "3. Binary Trees"
  - Pages: 8–10
  - Lines: 150–210
Text to analyze: {content}"""
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        max_tokens=4000,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message.content

def main():
    file_path = "test_document.pdf"  
    content = read_pdf(file_path)
    summary = generate_summary(content)
    print(summary)

if __name__ == "__main__":
    main()