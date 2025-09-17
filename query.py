import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain.memory import ConversationBufferWindowMemory

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

# BM25 encoder (dummy init)
bm25 = BM25Encoder()
bm25.fit(["dummy"])

# Hafıza kullanıcı bazlı (user_id -> memory)
user_memories = {}

# Kullanıcıya özel proje bilgisi (user_id -> last_project)
user_projects = {}

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
   - “bu konuya dair Bilgim Bulunmamaktadır.” de,  
   - ama ardından müşteriye yardımcı olacak **benzer bilgi veya alternatif öneri** sun.  

3. Her cevabı 5–7 cümleyle sınırla.  

4. Her cevabın sonunda bir **aksiyon çağrısı** yap:  
   - Satış ofisine davet  
   - Daha fazla bilgi paylaşma veya ekiple iletişim sağlama  
   - Randevu oluşturamıyorsun; iletişim bilgilerine yönlendir  

Kontekst:
{context}

Soru:
{question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Kullanıcıya özel proje tespiti
def detect_project(query: str, user_id: str) -> str:
    q = query.lower()
    if "mona" in q:
        project = "Mona"
    elif "nova" in q:
        project = "Nova"
    elif "terra" in q:
        project = "Terra"
    else:
        project = user_projects.get(user_id, "Genel")

    user_projects[user_id] = project
    return project

# Hybrid retriever
def hybrid_retrieve(query: str, user_id: str, top_k: int = 12):
    project = detect_project(query, user_id)

    dense_q = embedding_model.embed_query(query)
    sparse_q = bm25.encode_queries([query])[0]

    res = index.query(
        vector=dense_q,
        sparse_vector=sparse_q,
        top_k=top_k,
        include_metadata=True,
        filter={"project": project}
    )

    docs = vectorstore.similarity_search(query, k=top_k, filter={"project": project})
    context = "\n".join([d.page_content for d in docs])
    return context

# LLM
llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    model="gpt-4.1-mini",
    temperature=0.4,
    max_tokens=400
)

# Flask API için dışa aktarılan fonksiyon
def get_rag_answer(query, user_id="local_test"):
    if user_id not in user_memories:
        user_memories[user_id] = ConversationBufferWindowMemory(
            memory_key="chat_history", return_messages=True, k=5
        )

    context = hybrid_retrieve(query, user_id)
    result = llm.invoke(prompt.format(context=context, question=query)).content

    user_memories[user_id].save_context({"question": query}, {"answer": result})
    return result





