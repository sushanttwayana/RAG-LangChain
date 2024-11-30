import os
from langchain_community.document_loaders import PyPDFLoader

# Specify the relative path to the PDF file
pdf_path = os.path.join("..", "data", "monopoly.pdf")  # Go one level up from 'src' and then navigate to 'data'

#load pdf
def load_documents():
    document_loader = PyPDFLoader(pdf_path)
    return document_loader.load()

documents = load_documents()
print(documents[0])


# SPLit the documents