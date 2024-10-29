from flask import Flask, request, jsonify
import os
import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Initialize Flask app
app = Flask("RAG_api")

# Load API key and model configurations
config_data = json.load(open("fashion_tech/config.json"))
GROQ_API_KEY = config_data["GROQ_API_KEY"]
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Setup vectorstore and model chain
def setup_vectorstore():
    persist_directory = "fashion_tech/vector_db_dir"
    embeddings = HuggingFaceEmbeddings()
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return vectorstore

def chat_chain(vectorstore):
    llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0.4)
    retriever = vectorstore.as_retriever()
    memory = ConversationBufferMemory(
        max_token_limit=4000,  # Adjust based on the required context length
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"  # Set the output key explicitly in memory
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        memory=memory,
        verbose=True,
        return_source_documents=False,
        output_key="answer"  # Set the output key explicitly in the chain
    )
    return chain

# Initialize model
vectorstore = setup_vectorstore()
conversational_chain = chat_chain(vectorstore)

@app.route('/generate-response', methods=['POST'])
def generate_response():
    data = request.get_json()
    question = data.get("query", "")

    # Generate response from the RAG model
    response = conversational_chain({"question": question})
    answer = response.get("answer", "I'm sorry, I couldn't retrieve an answer.")

    # Return the answer as JSON
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)