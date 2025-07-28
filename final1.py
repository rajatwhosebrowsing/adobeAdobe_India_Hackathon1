import os
import json
import datetime
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic.v1 import BaseModel, Field

# Docker container paths
INPUT_DIR = '/app/input'
OUTPUT_DIR = '/app/output'

# --- Model Setup ---
print("Initializing models...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# This name 'hackathon' will be created by our Dockerfile
llm = ChatOllama(model="hackathon", temperature=0)
print("Models initialized.")

# --- Define Output Structure ---
class ExtractedSection(BaseModel):
    section_title: str = Field(description="A concise, descriptive title for the extracted section.")
    importance_rank: int = Field(description="A rank from 1 (most important) to 5 based on the user's job.")
    refined_text: str = Field(description="A 3-4 sentence summary of the key information in this section.")

# --- RAG Chain Creation Function ---
def create_json_rag_chain(llm):
    parser = JsonOutputParser(pydantic_object=ExtractedSection)
    analyzer_template = """You are an expert analyst acting as a **{persona}**.
Your job is to analyze the text provided in the 'Context' based on the user's job.
From the text below, extract the following information:
1. `section_title`: Create a concise and descriptive title for the section.
2. `importance_rank`: Assign a rank from 1 (most important) to 5 based on its relevance to the user's job.
3. `refined_text`: Summarize the key information from the context in 3-4 sentences.
Return ONLY the JSON object, without any other text or formatting.
Context: {context}
User's Job: {question}
{format_instructions}
"""
    analysis_prompt = PromptTemplate(
        template=analyzer_template,
        input_variables=["persona", "question", "context"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    return analysis_prompt | llm | parser

# --- Main Logic ---
def main():
    print("\n--- Starting RAG Process ---")
    
    try:
        with open(os.path.join(INPUT_DIR, 'config.json'), 'r') as f:
            config = json.load(f)
        persona = config['persona']
        job_to_be_done = config['job_to_be_done']
        print(f"Persona: {persona}\nJob: {job_to_be_done}")
    except FileNotFoundError:
        print(f"Error: 'config.json' not found in '{INPUT_DIR}'.")
        return

    pdf_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"No PDF files found in '{INPUT_DIR}'.")
        return
        
    print(f"Loading {len(pdf_files)} PDF(s)...")
    all_docs = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(os.path.join(INPUT_DIR, pdf_file))
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = pdf_file
        all_docs.extend(docs)
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    splits = text_splitter.split_documents(all_docs)
    
    print("Creating vector store...")
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
    
    relevant_docs = retriever.invoke(job_to_be_done)
    json_rag_chain = create_json_rag_chain(llm)

    print(f"\nAnalyzing {len(relevant_docs)} sections...")
    extracted_sections = []
    for doc in relevant_docs[:10]: # Process top 10 for efficiency
        try:
            result = json_rag_chain.invoke({
                "persona": persona,
                "question": job_to_be_done,
                "context": doc.page_content
            })
            if isinstance(result, dict):
                result['document'] = doc.metadata.get('source', 'Unknown')
                result['page_number'] = doc.metadata.get('page', 0)
                extracted_sections.append(result)
        except Exception as e:
            print(f"Could not process a chunk: {e}")
            
    def get_safe_sort_key(item):
        try:
            rank = int(item.get('importance_rank', 999))
        except (ValueError, TypeError):
            rank = 999
        title = item.get('section_title', '')
        return (rank, title)
        
    sorted_sections = sorted(extracted_sections, key=get_safe_sort_key)

    final_output = {
        "metadata": {
            "input_documents": pdf_files,
            "persona": persona,
            "job_to_be_done": job_to_be_done,
            "processing_timestamp": datetime.datetime.now().isoformat()
        },
        "Extracted Section": sorted_sections
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "result.json")
    with open(output_path, 'w') as f:
        json.dump(final_output, f, indent=4)
    
    print("\n--- Process Complete ---")
    print(f"JSON output saved to: {output_path}")

# --- Script Entry Point ---
if __name__ == "__main__":
    main()