import streamlit as st 
from streamlit_chat import message as ms
import pandas as pd
import os
from fpdf import FPDF 
from langchain.llms import OpenAI  # Correct import for OpenAI LLM
from langchain.embeddings import OpenAIEmbeddings  # Correct import for embeddings
from langchain.chains import ConversationChain
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma  # Correct import for Chroma vector store
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredExcelLoader, CSVLoader
from dotenv import load_dotenv 
from html_template import css, bot_template, user_template


# Load environment variables from .env file
load_dotenv()

# Access the OpenAI API key from the environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize vector_store in session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

# Function to clear chat history and memory
def clear_chat_history():
    st.session_state.messages = []
    st.session_state.memory.clear()
    st.success("Chat history and memory cleared!")

# Function to load document
def load_document(file):
    name, extension = os.path.splitext(file)

    if extension.lower() in ['.pdf', '.PDF']: 
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
    elif extension.lower() == '.docx':
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    elif extension.lower() == '.txt':
        print(f'Loading {file}')
        loader = TextLoader(file)
    elif extension.lower() == '.csv':
        print(f'Loading {file}')
        loader = CSVLoader(file)
    else:
        st.error('Unsupported document format!')
        return None

    data = loader.load()
    return data

# Function to chunk data
def chunk_data(data, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks

# Function to create embeddings and Chroma DB
def create_embeddings_chroma(chunks):
    embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')
    try:
        vector_store = Chroma.from_documents(chunks, embeddings)
        print("Chroma vector store created successfully.")
        return vector_store
    except Exception as e:
        print(f"Error creating Chroma vector store: {e}")
        return None

# Initialize session state
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = """

    Eres Kira, un asistente virtual diseñado específicamente para estudiantes de la Universidad de La Sabana. Tu personalidad es amigable y divertida. Te comunicas exclusivamente en español.

Tu Estilo de Respuestas es proporcionar respuestas en español y mantén la información clara y concreta.

Tu primer mensaje al usuario debe ser Comienza con el saludo: "Hola Monito, soy Kira y estoy aquí para facilitar y guiarte en tu vida universitaria en la Sabana. Y Pregunta al user Su nombre, Su carrera ,Su semestre, Su hobby

Cuando conozcas el nombre del usuario, responde: "Hola, [nombre]. ¿En qué puedo ayudarte hoy?"
Ofrece ejemplos de preguntas para orientación, utilizando viñetas:
¿Qué actividades o grupos puedo participar relacionados con el [su hobby]?
¿Qué profesor me recomiendas para mis materias?
Quiero conectar con gente en mi misma carrera: [su carrera].

Acceso a Datos:
Extrae la información necesaria sobre los estudiantes de los documentos cargados en formato CSV. Utiliza estos datos para responder adecuadamente a las consultas de los estudiantes.
                """

# Function to generate response
def generate_response(q, k=3): 
    model = OpenAI(temperature=0.5, api_key=openai_api_key)  # Correct LLM initialization
    
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(st.session_state.system_prompt),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])
    
    if st.session_state.vector_store: 
        # Perform RAG search
        docs = st.session_state.vector_store.similarity_search(q, k=k) if st.session_state.vector_store else []

        # Generate response using RAG if documents are found
        if docs:
            chain = load_qa_chain(model, chain_type="stuff")
            rag_response = chain.run(input_documents=docs, question=q)
        else:
            rag_response = None  # No RAG response available

        # Generate LLM response
        conversation = ConversationChain(
            llm=model,
            memory=st.session_state.memory,
            prompt=prompt_template,
            verbose=True
        )
        llm_response = conversation.predict(input=q)

        # Combine or prioritize responses
        if rag_response:
            # You can combine responses or choose one based on your requirements
            combined_response = f"{llm_response}\n   \n{rag_response}"
        else:
            combined_response = llm_response

        return combined_response 
    
# Generate LLM response
    conversation = ConversationChain(
        llm=model,
        memory=st.session_state.memory,
        prompt=prompt_template,
        verbose=True) 
    response = conversation.predict(input=q) 
    return response

# Streamlit UI 

st.set_page_config(page_title="Amigo bot", layout="centered")
logo_path = "logo_sabana.png"  # Replace with your logo path
st.image(logo_path, width=50)  # Adjust the width as needed
st.title("Kira")
st.write("Tu copiloto para guiarte y ayudarte en la vida Universitaria") 
st.write("Conecta y haz nuevos amigos - Sube tus documentos y estudiemos juntos") 




chunk_size = 256 
k = 3 

st.write(css, unsafe_allow_html=True)

# Sidebar
with st.sidebar: 

    uploaded_file = st.file_uploader("Sube tus documentos acá", type=["pdf", "csv", "txt", "xlsx", "docx"])
    add_data = st.button("Cargar Documento")

    if uploaded_file and add_data:
        with st.spinner("Procesando su Documento... "):
            bytes_data = uploaded_file.read() 
            file_name = os.path.join('./', uploaded_file.name) 
            with open(file_name, "wb") as f: 
                f.write(bytes_data) 
            data = load_document(file_name) 
            chunks = chunk_data(data, chunk_size=chunk_size) 

            st.session_state.vector_store = create_embeddings_chroma(chunks)
            if st.session_state.vector_store:
                st.success("Documento procesado con éxito!")
            else:
                st.error("Failed to process the document. Please try again.")

    if st.checkbox("Personalizar"):
        st.session_state.system_prompt = st.text_area("System Prompt", st.session_state.system_prompt, height=300)
    
    if st.button("Clear Chat History"):
        clear_chat_history()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    if message["role"] == "user":
        st.write(user_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)
    else:
        st.write(bot_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)



# Chat Input 
if prompt := st.chat_input("¿Cómo puedo ayudarte?"):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("T"):
        st.markdown(user_template.replace("{{MSG}}", prompt), unsafe_allow_html=True)

    with st.chat_message("K"):
        message_placeholder = st.empty()
        full_response = generate_response(prompt, k)
        
        # Place response in the correct template
        message_placeholder.markdown(bot_template.replace("{{MSG}}", full_response), unsafe_allow_html=True)

    # Append the assistant's response to the chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
