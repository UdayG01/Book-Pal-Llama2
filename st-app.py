
import time
import streamlit as st
from PyPDF2 import PdfReader
from langchain.document_loaders import PyPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma, FAISS
# from langchain.vectorstores import Pinecone
from sentence_transformers import SentenceTransformer
from langchain.chains.question_answering import load_qa_chain
# import pinecone
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# importing the files for the llm
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFaceHub
from htmlTemplates import css, bot_template, user_template

import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

from huggingface_hub import hf_hub_download

model_name_or_path = "TheBloke/Llama-2-13B-chat-GGML"
model_basename = "llama-2-13b-chat.ggmlv3.q5_1.bin"
model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)
# model_path = "./drive/MyDrive/Colab Notebooks/llama-2-7b-chat.ggmlv3.q4_0.bin"

def get_pdf_text(pdf_docs):
  text = ""
  for pdf in pdf_docs:
    pdf_reader = PdfReader(pdf)
    # loader = PyPDFLoader(pdf)
    # data = loader.load()
    for page in pdf_reader.pages:
      # text += document.page_content
      text += page.extract_text()
  return text


def get_text_chunks(text):
   text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
  # text_splitter = CharacterTextSplitter(
  #     chunk_size = 20000,
  #     chunk_overlap = 100,
  #     length_function = len
  # )
   docs = text_splitter.split_text(text)
   return docs

def get_vectorstore(docs):
  embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
  #db = Chroma.from_texts(docs, embeddings)
  db = FAISS.from_texts(docs, embeddings)
  return db

def get_conversation_chain(vectorstore):
  n_gpu_layers = 80
  n_batch = 1024

  # Loading model,
  llm = LlamaCpp(
      model_path=model_path,
      max_tokens=512,
      n_gpu_layers=n_gpu_layers,
      n_batch=n_batch,
      n_ctx=4096,
      verbose=False,
  )

  # repo_id = "google/flan-t5-xxl"
  # llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_length": 64})

  memory = ConversationBufferMemory(memory_key = 'chat_history', return_messages=True)
  conversation_chain = ConversationalRetrievalChain.from_llm(
      llm=llm,
      retriever=vectorstore.as_retriever(),
      memory=memory
  )
  return conversation_chain

def handle_userinput(user_question):
  response = st.session_state.conversation({'question': user_question})
  st.session_state.chat_history = response['chat_history']

  for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
  st.set_page_config(page_title="Omnipresent.ai",
                    page_icon=":alien:",
                    layout="wide")
  st.write(css, unsafe_allow_html=True)

  if "conversation" not in st.session_state:
    st.session_state.conversation = None
  if "chat_history" not in st.session_state:
    st.session_state.chat_history = None


  st.header("Have a word with your book :scroll:")
  user_question = st.text_input("What brings you here? ")
  if user_question:
    handle_userinput(user_question)

  with st.sidebar:
    st.subheader("Your documents")
    pdf_docs = st.file_uploader(
        "Upload your pdfs here and click on 'Process'", accept_multiple_files = True)
    if st.button("Process"):
      with st.spinner():
        progress_text = "Processing..."
        myBar = st.progress(0, text=progress_text)
        # Perform loading on the pdf
        text = get_pdf_text(pdf_docs)
        # st.write(text)

        for percent_complete in range(25):
          time.sleep(0.1)
          myBar.progress(percent_complete + 1, text=progress_text)

        # Perform text-splitting
        docs  = get_text_chunks(text)

        for percent_complete in range(25, 51):
          time.sleep(0.1)
          myBar.progress(percent_complete + 1, text=progress_text)

        # Performing embeddings/vectorization using HuggingFaceEmbeddings
        # and I'll be storing these embeddings in Chroma vector store
        db = get_vectorstore(docs)

        for percent_complete in range(51, 75):
          time.sleep(0.1)
          myBar.progress(percent_complete + 1, text=progress_text)

        # Making a conversation chain
        # conversation = get_conversation_chain(db)
        # do the below one in case you want a persistent convo
        st.session_state.conversation = get_conversation_chain(db)

        for percent_complete in range(76, 100):
          time.sleep(0.1)
          myBar.progress(percent_complete + 1, text=progress_text)
        myBar.progess(100, text="Done")

if __name__ == '__main__':
  main()
