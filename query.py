import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_pinecone import PineconeVectorStore
from langchain.memory import ConversationBufferMemory

from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder

# load environment variables
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

# BM25 encoder (önceden fit edilip kaydedilmeli; burada dummy kullanım var)
bm25 = BM25Encoder()
bm25.fit(["dummy"])  # RAM taşmasını önlemek için sadece boş init

# Hafıza
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
   - “ bu konuya dair Bilgim Bulunmamaktadır.” de,  
   - ama ardından müşteriye yardımcı olacak **benzer bilgi veya alternatif öneri** sun.  

3. Her cevabı 5–7 cümleyle sınırla (uzun ve sıkıcı olmasın).  

4. Her cevabın sonunda bir **aksiyon çağrısı** yap:  
   - Satış ofisine davet  
   - Daha fazla bilgi paylaşma veya ekiple iletişim sağlama  

Kontekst:
{context}

Soru:
{question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Sorguyu genişlet (query expansion)
def expand_query(query: str):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-4", temperature=0)
    prompt = f"Soruya 3 eş anlamlı veya benzer varyant üret: {query}"
    expansions = llm.invoke(prompt).content.split("\n")
    return [query] + [e.strip("- ") for e in expansions if e.strip()]

# Hybrid retriever (dense + sparse)
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

    # Vectorstore üzerinden context'i getir
    docs = vectorstore.similarity_search(query, k=top_k)
    context = "\n".join([d.page_content for d in docs])
    return context

# LLM + zincir
llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    model="gpt-4.1-mini",
    temperature=0.4,
    max_tokens=400
)

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
