import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_pinecone import PineconeVectorStore
from langchain.memory import ConversationBufferMemory

from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from functools import lru_cache  # RAM tasarrufu için cache mekanizması

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")

# Pinecone Index
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(pinecone_index_name)

# Embeddings
embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)

# VectorStore
vectorstore = PineconeVectorStore(
    index_name=pinecone_index_name,
    embedding=embedding_model,
    text_key="text"
)

# BM25 encoder
bm25 = BM25Encoder()

# Belgeleri sadece ilk çalıştırmada belleğe al (RAM tasarrufu)
@lru_cache(maxsize=1)
def get_chunks():
    loader = DirectoryLoader("data", loader_cls=UnstructuredFileLoader)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

# BM25 modeli fit edilir
chunks = get_chunks()
bm25.fit([doc.page_content for doc in chunks])

# Hafıza (Chat geçmişi)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Prompt Template
template = """
Rol: Satış Danışmanı  
Ton: Kibar, güven veren, müşteri odaklı, yönlendirici, samimi  

Sen Folkart Yapı’da çalışan deneyimli bir satış danışmanısın.  
Müşterilere karşı:  
- Profesyonel, güven verici ve samimi bir dil kullan.  
- Cevapların hem bilgilendirici hem yönlendirici olsun.  
- Müşteriye değer kat, ihtiyaçlarını anlamaya çalış ve sohbeti canlı tut.  
- Belgelerde yer alan bilgilerin dışına çıkma, ama bilgiyi sohbet havasında aktar.  

Kurallar:  
   - soruya net ve kısa cevap ver.  
   - müşteriye bir sonraki adımı öner.  
   - müşteriyi ihtiyaçlarını açmaya davet et.  

2. Eğer belgede net bilgi yoksa:  
   - “bu konuya dair bilgim bulunmamaktadır.” de,  
   - ardından müşteriye yardımcı olacak benzer bilgi veya alternatif öneri sun.  

3. Her cevabı 5–7 cümleyle sınırla (uzun ve sıkıcı olmasın).  

4. Her cevabın sonunda bir aksiyon çağrısı yap:  
   - Satış ofisine davet  
   - Daha fazla bilgi paylaşma veya ekiple iletişim sağlama  

Kontekst:
{context}

Soru:
{question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Sorguyu genişlet (Query Expansion)
def expand_query(query: str):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-4", temperature=0)
    prompt = f"Soruya 3 eş anlamlı veya benzer varyant üret: {query}"
    expansions = llm.invoke(prompt).content.split("\n")
    return [query] + [e.strip("- ") for e in expansions if e.strip()]

# Hybrid retriever (BM25 + Dense)
def hybrid_retrieve(query: str, top_k: int = 12):
    all_queries = expand_query(query)
    matches = []
    for q in all_queries:
        dense_q = embedding_model.embed_query(q)
        sparse_q = bm25.encode_queries([q])[0]
        res = index.query(
            vector=dense_q,
            sparse_vector=sparse_q,
            top_k=top_k,
            include_metadata=True
        )
        matches.extend(res.matches)

    docs = vectorstore.similarity_search(query, k=top_k)
    context = "\n".join([d.page_content for d in docs])
    return context

# LLM
llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    model="gpt-4",
    temperature=0.4,
    max_tokens=400
)

# RAG Zinciri
rag_chain = (
    RunnableMap({
        "context": hybrid_retrieve,
        "question": RunnablePassthrough()
    })
    | prompt
    | llm
)

# Flask API için dışa aktarılan fonksiyon
def get_rag_answer(query):
    return rag_chain.invoke(query).content


