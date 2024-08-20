import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()


genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

def get_pdf_text(file):
    text = ""
    for pdf in file:
        pdf_reader = PdfReader(pdf) # Reading the pdf file
        for page in range(len(pdf_reader.pages)):    # Iterating through the pages
            text += pdf_reader.pages[page].extract_text()  # Extracting the text from the page
    return text

def get_text_chunks(text):  # Function to split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):  # Function to get the vector store
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vector_store = FAISS.from_texts(text_chunks,embedding=embeddings)
    vector_store.save_local('faiss_index')

def getConversationalChain(): # Function to get the conversational chain
    prompt_template = """
    Answer the question in detail as much as possible. Make sure to provide a clear and concise answer. If the answer is not in the context 
    just say "answer is not available in the context". Don't provide the wrong answer.
    Context:\n {context}?\n
    Question:\n{question}\n
    
    Answer:
    """
    chat_chain = ChatGoogleGenerativeAI(model='gemini-1.5-flash',temperature=0.7)
    prompt = PromptTemplate(template=prompt_template,input_variables=['context','question']) # it is provided in langchain 
    chain = load_qa_chain(llm=chat_chain,chain_type="stuff",prompt = prompt)
    return chain

def getUserInput(userQuestion):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    new_db = FAISS.load_local('faiss_index',embeddings=embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(userQuestion)

    chain = getConversationalChain()

    # `input_documents` is used instead of `context`
    response = chain(
        {"input_documents": docs, "question": userQuestion},
        return_only_outputs=True
    )

    st.write(response['output_text'])

# Function to load CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def main():
    
    st.set_page_config("DocBot")
    load_css('styles.css')
    
    st.markdown('<div class="custom-mainheader">Welcome to DocBot!</div>', unsafe_allow_html=True)

    # Use the custom class from CSS file
    st.markdown('<div class="custom-header">Got a question? Let\'s dive into the PDF!</div>', unsafe_allow_html=True)
    user_question = st.text_input("Your Question:")
    if user_question:
        getUserInput(user_question)

    with st.sidebar:
        st.markdown('<div class="custom-menutitle">Menu</div>', unsafe_allow_html=True)
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Processing Completed!")
        else:
            st.warning("Please upload at least one PDF file.")

if __name__ == "__main__":
    main()
