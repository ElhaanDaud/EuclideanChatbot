import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os 
import time

load_dotenv()

progress_bar = st.progress(0)
for i in range(100):
    progress_bar.progress(i + 1)
    time.sleep(0.1)


OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Header of the webpage
st.header("Euclidean Chatbot Project")


# Uploading Documents
with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload your file and start asking questions", type="Pdf")

# Extracting the text
if file is not None:
    pdf_reader = PdfReader(file)  # pdf_reader containes the read file
    text = ""
    for page in pdf_reader.pages:  # returning page coordinates

        text = (
            text + page.extract_text()
        )  # to extract the texts from page we use extract_text()
    # st.write(text)

    # Breaking it into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000,  # length of each chunk is 1000 characters
        chunk_overlap=150,  # +- 150 characters of abover and below will be considered
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    # st.write(chunks)

    # Generating Embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # creating vector storage - FAISS (facebook AI semantic search)
    vector_store = FAISS.from_texts(
        chunks, embeddings
    )  # Chiz(chunks) - Its value(embeddings)

    # get user question
    user_question = st.text_input("Type your question here")

    # Create two columns
    col1, col2 = st.columns(2)

    # Add a button to the first column
    with col1:
        button1 = st.button("Summary Of Document")

    # Add a button to the second column
    with col2:
        button2 = st.button("Important Pointers")

    # button clicked
    option = None  # Initialize option outside the button click check
    search_triggered = False  # Initialize search_triggered

    if button1:
        option = "Summary Of Document"
        search_triggered = True
    elif button2:
        option = "Important Pointers"
        search_triggered = True

    # Check if the search should be triggered
    if st.session_state.get("last_option") != option:
        st.session_state["last_option"] = option
        search_triggered = True

    # Do similarity search and generate response
    if user_question or search_triggered:
        match = vector_store.similarity_search(user_question)
        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            temperature=0,  # lower the value more accurate (randomness less)
            max_tokens=1000,  # finetuning stuffs
            model_name="gpt-3.5-turbo",
        )

        # output results
        # chain - take the question , get relevant documents , pass it to the llm, generate the output
        chain = load_qa_chain(llm, chain_type="stuff")

        if option == "Important Pointers":
            pointers_question = "List the important pointers from the document."
            response = chain.run(input_documents=match, question=pointers_question)
        elif option == "Summary Of Document":
            summary_question = "Provide a summary of the document."
            response = chain.run(input_documents=match, question=summary_question)
        else:
            # Default case, use the original user question
            response = chain.run(input_documents=match, question=user_question)

        st.write(response)
