import streamlit as st
import pandas as pd
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOllama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Function to extract text from PDF files
def get_pdf_text(pdf_docs):
    """Extracts text from a list of PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PyPDF2.PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to extract text from Excel files
def get_excel_text(excel_files):
    """Extracts text from a list of Excel files."""
    all_text = ""
    for file in excel_files:
        try:
            df_dict = pd.read_excel(file, sheet_name=None)
            for sheet_name, df in df_dict.items():
                all_text += f"Sheet: {sheet_name}\n{df.to_string()}\n\n"
        except Exception as e:
            st.error(f"Error reading {file.name}: {e}")
    return all_text

#split text into chunks
def get_text_chunks(text):
    """Splits into smaller chunks."""
    text_split = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_split.split_text(text)
    return chunks

#vector store
def get_vector_store(text_chunks):
    """Creates a FAISS vector store"""
    try:
        #using an embedding model 
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

#conversational chain
def get_conversational_chain(vector_store):
    """Creates a conversational retrieval chain."""
    try:
        llm = ChatOllama(model="llama2")
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(),
            memory=memory
        )
        return conversation_chain
    except Exception as e:
        st.error(f"Error creating conversational chain: {e}")
        return None


def handle_user_input(user_question):
    """Handles user input and displays the conversation."""
    if st.session_state.conversation:
        try:
            response = st.session_state.conversation({'question': user_question})
            st.session_state.chat_history = response['chat_history']
            for i, message in enumerate(st.session_state.chat_history):
                if i % 2 == 0:
                    with st.chat_message("user"):
                        st.markdown(message.content)
                else:
                    with st.chat_message("assistant"):
                        st.markdown(message.content)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please upload and process documents before asking questions.")


def main():
    st.set_page_config(page_title="Financial Document")
    st.header("Chat")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    with st.sidebar:
        st.subheader("Your Documents")
        uploaded_files = st.file_uploader(
            "Upload your PDF and Excel files here",
            accept_multiple_files=True,
            type=['pdf', 'xlsx']
        )
        
        if st.button("Process Documents"):
            if uploaded_files:
                with st.spinner("Processing documents..."):
                    try:
                        pdf_files = [f for f in uploaded_files if f.type == "application/pdf"]
                        excel_files = [f for f in uploaded_files if f.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]
                        
                        raw_text = ""
                        if pdf_files:
                            raw_text += get_pdf_text(pdf_files)
                        if excel_files:
                            raw_text += get_excel_text(excel_files)
                        
                        if raw_text:
                            text_chunks = get_text_chunks(raw_text)
                            vector_store = get_vector_store(text_chunks) 
                            if vector_store:
                                st.session_state.conversation = get_conversational_chain(vector_store)
                                st.success("Processing complete")
                            else:
                                st.error("Failed to create the vector store.")
                        else:
                            st.warning("No text")
                    except Exception as e:
                        st.error(f"An error occurred during processing: {e}")
            else:
                st.warning("Please upload document.")


    st.subheader("Ask a Question")
    user_question = st.chat_input("Ask")

    if user_question:
        handle_user_input(user_question)

if __name__ == '__main__':
    main()