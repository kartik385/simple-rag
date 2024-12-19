from langchain_ollama.llms import OllamaLLM
# from ollama import embeddings as Embedder
from langchain_community.document_loaders import PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
import os
# Load the model
llm = OllamaLLM(model="tinyllama")

# Load the document
document = PDFMinerLoader("./elon.pdf").load()

# Print the attributes of the document to find the correct one


# Assuming the correct attribute is 'content'


# Define RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(document)




embeddings = OllamaEmbeddings(model="nomic-embed-text")


persist_directory = 'db'

# Check if the database directory exists
if os.path.exists(persist_directory):
    # Load the existing database
    my_database = Chroma(persist_directory=persist_directory,embedding_function=embeddings)
else:
    # Create a new database and embed the documents
    my_database = Chroma.from_documents(
        documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )

from langchain_core.prompts import PromptTemplate

template = """
This is a wikipedia article of celebrity, give the answer to qustions about the article. 
If you don't know the answer, just say you don't know. 
You answer with short and concise answer, no longer than 2 sentences.

Context: {context}
Question: {question}
Answer: 

"""

prompt = PromptTemplate(template=template, input_variables=["context", "question"])

from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

rag_chain = (
{"context": my_database.as_retriever(),  "question": RunnablePassthrough()} 
| prompt 
| llm
| StrOutputParser() 
)


input = input("Enter the question:\n ")

answer = rag_chain.invoke({"question": input})

print(answer)
