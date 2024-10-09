import os
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain.memory import StreamlitChatMessageHistory, ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader, CSVLoader
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import tempfile

# 1. Load environment variables
load_dotenv()

# 2. Initialize LLM with API key
groq_api_key = os.environ['GROQ_API_KEY']
llm = ChatGroq(model="mixtral-8x7b-32768", groq_api_key=groq_api_key)

# 3. Define the prompt template
prompt = PromptTemplate(template="Answer the question.\nQuestion: {question}\nHelpful Answers:", 
                        input_variables=['question'])

# 4. Memory for conversation
sthistory = StreamlitChatMessageHistory(key='chatmessage')
hist = ConversationBufferMemory(chat_memory=sthistory)
chain = LLMChain(llm=llm, prompt=prompt, memory=hist)

# 5. Streamlit app title
st.title("Conversational QA Bot with File Support")

# Create a sidebar for file upload
with st.sidebar:
    uploaded_files = st.file_uploader("Upload files (text, CSV, or PDF)", type=["txt", "csv", "pdf"], accept_multiple_files=True)

# 6. Initialize vectorstore and embeddings
if uploaded_files:
    all_docs = []
    
    for uploaded_file in uploaded_files:
        # Load the uploaded file based on its extension
        if uploaded_file.name.endswith(".txt"):
            loader = TextLoader(uploaded_file)
        elif uploaded_file.name.endswith(".csv"):
            loader = CSVLoader(uploaded_file)
        elif uploaded_file.name.endswith(".pdf"):
            with tempfile.NamedTemporaryFile(delete=False) as temp_pdf:
                temp_pdf.write(uploaded_file.read())
                temp_pdf_path = temp_pdf.name
            
            loader = PyPDFLoader(temp_pdf_path)

        # Process the file
        documents = loader.load()
        all_docs.extend(documents)  # Combine all loaded documents
    
    # Split all documents into manageable chunks
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(all_docs)
   

    embeddings = AzureOpenAIEmbeddings(deployment='MAJNU', azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'])

    vectordb = FAISS.from_documents(docs, embeddings)

    st.write("Files successfully uploaded and processed.")
else:
    st.write("Please upload files to proceed.")

# 7. Function to query vector database and LLM, and include sources
def query_file(question):
    docs = vectordb.similarity_search(question, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    response = chain.invoke({"question": f"{question}\n\nContext: {context}"})
    # Only keep the first source
    source = f"Source: {docs[0].page_content[:200]}..." if docs else "No sources available."
    return response["text"], source

# 8. Display chat history and allow the user to input new questions
for msg in sthistory.messages:
    st.chat_message(msg.type).write(msg.content)

# Get user input
if user_input := st.chat_input("Ask your question here:"):
    st.chat_message("human").write(user_input)
    
    if uploaded_files:
        answer, source = query_file(user_input)
        st.chat_message("ai").write(answer)
        
        # Display the first source of the response
        st.write("Source:")
        st.write(source)
    else:
        response = chain.invoke({"question": user_input})
        st.chat_message("ai").write(response['text'])
