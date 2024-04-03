# 🚀 Euclidean Chatbot 
General Idea of the Project 
------

* This is a [RAG](https://www.youtube.com/watch?v=T-D1OfcDW1M) Chatbot that allows users to upload PDF documents and ask questions about them. The text from the uploaded PDF documents is extracted using streamlit - popular python library for creating interactive web applications with minimal effort, broken into chunks, and then embedded. These embeddings are stored in a vector store, which enables efficient similarity search.Users can ask questions, and the system retrieves relevant chunks of text from the document to provide answers.

Breaking Text into chunks
------

```python
text_splitter = RecursiveCharacterTextSplitter(
            separators="\n",
            chunk_size=1000,  #length of each chunk is 1000 characters
            chunk_overlap=150, #+- 150 characters of abover and below will be considered
            length_function=len 
    )
chunks=text_splitter.split_text(text)
```
* The extracted text from the PDF document is divided into smaller segments or chunks. This is done to manage large texts efficiently, as processing them as a whole might be computationally expensive. It also allows for more precise analysis and retrieval of information.

Embedding Chunks
------

```python
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
```
* After breaking the text into chunks, each chunk is converted into a numerical representation called an [Embedding](https://platform.openai.com/docs/guides/embeddings/what-are-embeddings). Embeddings capture semantic information about the text, enabling comparison and analysis in a numerical space. In this code, OpenAI's embedding service is used to generate embeddings for the text chunks.

OpenAI's Embedding System
------

* OpenAI provides pre-trained language models and embedding systems that capture rich semantic information from text. These embeddings can be used for various natural language processing tasks, including similarity search, language understanding, and generation.


Vector Storage
------

```python
vector_store=FAISS.from_texts(chunks,embeddings)  
```
* Once the embeddings are generated, they are stored in a [vector store](https://python.langchain.com/docs/integrations/vectorstores/faiss). The vector store is a data structure optimized for fast similarity search and retrieval. FAISS (Facebook AI Similarity Search) is used here for efficient storage and retrieval of embeddings.

Semantic Search and Similarity Search
------

```python
match = vector_store.similarity_search(user_question)
```
* Semantic search involves understanding the meaning of text and retrieving information based on its semantic relevance rather than just keyword matching.
* Similarity search, in this context, involves finding chunks of text that are most similar to a given query.
* The embeddings of the query and the text chunks are compared using similarity measures to find the most relevant chunks.

LLM (Language Model)
------

```python
llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            temperature=0,  # lower the value more accurate (randomness less)
            max_tokens=1000,  # finetuning stuffs
            model_name="gpt-3.5-turbo"
        )
# chain - take the question , get relevant documents , pass it to the llm, generate the output
chain = load_qa_chain(llm, chain_type="stuff")
```
* The [LLM](https://aws.amazon.com/what-is/large-language-model/) used is based on OpenAI's GPT (Generative Pre-trained Transformer) architecture. The LLM is used for generating responses to user queries based on the context provided by the relevant text chunks. It can understand and generate human-like text, making it suitable for natural language understanding and generation tasks.

Architecture
------

<img src="https://github.com/ElhaanDaud/EuclideanChatbot/assets/165207315/2d9ab5a4-4844-46bf-bdbe-60f101c662d7" alt="Architecture" style="width:900px;"/>


Usage
------

<img src="https://github.com/ElhaanDaud/EuclideanChatbot/assets/165207315/2949f752-8af2-442f-b956-786f9d53ba76" alt="use1" style="width:900px;"/>

<img src="https://github.com/ElhaanDaud/EuclideanChatbot/assets/165207315/c7c2f4ca-ce9a-4759-aaa6-d4a5d447db44" alt="use1.1" style="width:900px;"/>




