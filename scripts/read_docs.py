import docx
import sys

def read_docx(file_path):
    try:
        doc = docx.Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return '\n'.join(full_text)
    except Exception as e:
        return f"Error reading {file_path}: {e}"

if __name__ == "__main__":
    for arg in sys.argv[1:]:
        print(f"--- {arg} ---")
        print(read_docx(arg))
        print("\n")
