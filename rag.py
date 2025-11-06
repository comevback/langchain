from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from dotenv import load_dotenv
import os

# === 1️⃣ 加载环境变量 (.env 文件) ===
load_dotenv()

# === 2️⃣ 从 .env 读取配置 ===
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
AZURE_EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")

# === 3️⃣ 加载 PDF 文档 ===
loader = PyPDFLoader("./白话机器学习的数学_图灵图书_立石贤吾_Z_Library.pdf")
docs = loader.load()

# === 4️⃣ 切块 ===
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# === 5️⃣ 向量化（Azure Embedding）===
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=AZURE_EMBED_DEPLOYMENT,
    openai_api_version=OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY
)
vectorstore = FAISS.from_documents(chunks, embeddings)

# === 6️⃣ 构建问答链（Azure Chat 模型）===
llm = AzureChatOpenAI(
    azure_deployment=AZURE_CHAT_DEPLOYMENT,
    openai_api_version=OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY
)
qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

# === 7️⃣ 提问 ===
query = "根据文档，美绪是谁，有什么特色，学习到了什么？"
print("Q:", query)
print("A:", qa.invoke(query))
