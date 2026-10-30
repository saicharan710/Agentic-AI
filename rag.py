from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_classic.chains import RetrievalQA # Mention that this is outdated, and will be removed in the future. 
                                                 # Give students a homework on how they can improve on the RAG pipeline. 
                                                 # Show them the diagram from the handbook, and the improvement should be on the pipeline.

from llm import create_llm

from dotenv import load_dotenv
import os

load_dotenv()

def process_pdf(file_path: str = "./resume.pdf"): # Have students use their own resume, without images or tables. If they do not have one, have them make using Claude
    loader = PyPDFLoader(file_path)
    document = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=150
    ) # Explain the difference between Recursive Character Text Splitter & Character Text Splitter

    chunks = splitter.split_documents(document)

    return chunks

embeddings = HuggingFaceEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2" # If any dll error occurs, check their python and cmake version. Huggingface token warning can be ignored.
)

vector_store = Chroma(
    embedding_function=embeddings,
    collection_name="resume_data_collection",
    persist_directory="./vector_db"
)

def ingest_data():
    chunks = process_pdf()
    vector_store.add_documents(chunks)

    return "Data ingestion completed." # Check if vector_db folder has been created or not. 
                                       # It should have a sqlite file and collection folder. If any is missing the run has failed.

def rag_chain(query: str):
    llm = create_llm()
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )

    return chain.invoke(query)


if __name__ == "__main__":
    # ----------Run this Immediately After creating process_pdf--------------------------
    chunks = process_pdf()

    for chunk in chunks:
        print(chunk)
        print("\n\n\n")

    # # ----------Run this Immediately After creating ingest_data--------------------------
    #  ingest_data() # First run will take time to download the model.

    # ----------Run this Immediately After creating rag_chain--------------------------
    query = input("Enter your query: ")
    response = rag_chain(query)
    print(response) # Ask the students to figure out how to get the clean response (the final response only).
    