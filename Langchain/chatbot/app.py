from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


# -----------------------------
# 1. Load Your Text File
# -----------------------------
print(1)
loader = TextLoader("about_me.txt", encoding="utf-8")
docs = loader.load()


# -----------------------------
# 2. Split into Chunks
# -----------------------------
print(2)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100
)
chunks = splitter.split_documents(docs)


# -----------------------------
# 3. Embeddings Model (local but tiny)
# -----------------------------
print(3)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
# NOTE: This model is small and downloads fast (<30MB).
# It is only for creating embeddings, not generating text.


# -----------------------------
# 4. Vector Database
# -----------------------------
print(4)
db = FAISS.from_documents(chunks, embeddings)
retriever = db.as_retriever()


# -----------------------------
# 5. Cloud HuggingFace Model (No Download!)
# -----------------------------
print(5)
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.5,
    max_new_tokens=300,
    huggingfacehub_api_token="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"
)


# -----------------------------
# 6. Conversational RAG Chain
# -----------------------------
print(6)
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

rag_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)


# -----------------------------
# 7. Chat Loop
# -----------------------------
print(7)
print("🌐 Cloud RAG Chatbot Ready! (No local model)\n")

while True:
    question = input("You: ")

    if question.lower() in ["exit", "quit", "stop"]:
        print("Chatbot: Goodbye!")
        break

    answer = rag_chain.invoke({"question":question})
    print("\nChatbot:", answer["answer"], "\n")
