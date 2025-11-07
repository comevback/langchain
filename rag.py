import time
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import AzureChatOpenAI  # 仍保留 Azure 作为问答模型
from langchain_huggingface import HuggingFaceEmbeddings


# === 1️⃣ 加载环境变量 ===
load_dotenv()
index_file = "faiss_index_local"

# === 2️⃣ Azure 聊天模型配置（用于问答阶段） ===
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")

# === 3️⃣ 加载 PDF 文档 ===
loader = PyPDFLoader("./Python核心编程 (Wesley Chun) (Z-Library).pdf")
docs = loader.load()

# === 4️⃣ 文本切块 ===
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# === 5️⃣ 初始化本地 Embedding 模型 ===
# 可选模型：
# "intfloat/multilingual-e5-base"  → 推荐 (中英双语)
# "BAAI/bge-base-zh"               → 中文表现更强（稍慢）
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")

# === 6️⃣ 检查已有索引，准备断点续跑 ===
all_texts, all_metas = [], []
processed_count = 0
vectorstore = None

if os.path.exists(index_file):
    try:
        vectorstore = FAISS.load_local(
            index_file, embeddings, allow_dangerous_deserialization=True
        )
        processed_count = len(vectorstore.docstore._dict)
        print(f"✅ 已处理 {processed_count} 个 chunk，跳过这些，继续处理剩余部分。")
        all_texts = [chunks[i].page_content for i in range(processed_count)]
        all_metas = [chunks[i].metadata for i in range(processed_count)]
    except Exception as e:
        print(f"⚠️ 加载现有索引失败，将重新创建: {e}")

# === 7️⃣ 批量生成 Embedding 并保存索引 ===
batch_size = 10
chunks_to_process = chunks[processed_count:]
print(f"👉 共有 {len(chunks_to_process)} 个待处理 chunk。")

for i in range(0, len(chunks_to_process), batch_size):
    batch = chunks_to_process[i:i + batch_size]
    texts = [doc.page_content for doc in batch]
    metas = [doc.metadata for doc in batch]
    all_texts.extend(texts)
    all_metas.extend(metas)

    print(f"🔹 处理进度: {processed_count + i + len(batch)}/{len(chunks)}")

    # 本地 embedding 不需要重试，也不会 429
    embeddings.embed_documents(texts)

    # 每批保存一次进度
    try:
        vectorstore = FAISS.from_texts(
            all_texts, embeddings, metadatas=all_metas)
        vectorstore.save_local(index_file)
        print(f"💾 已保存进度到 {index_file}")
    except Exception as e:
        print(f"⚠️ 保存本地进度失败: {e}")

# === 8️⃣ 向量化完成 ===
print("✅ 本地向量化完成，正在构建问答系统...")

# === 9️⃣ 构建 QA 链 ===
llm = AzureChatOpenAI(
    azure_deployment=AZURE_CHAT_DEPLOYMENT,
    openai_api_version=OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# === 🔟 提问 ===
query = "《Python核心编程》中第8章扩展Python具体讲了什么，我会学到什么，学会之后我能用这个发挥什么作用。"
print("\nQ:", query)
result = qa.invoke(query)
print("\nA:", result['result'])
print("\n📖 检索片段：")
for i, doc in enumerate(result['source_documents']):
    print(f"\n--- 片段{i+1} ---\n{doc.page_content[:500]}...")
