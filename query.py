import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain.memory import ConversationBufferWindowMemory

from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder

# 1. Ortam değişkenlerini yükle
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")

# 2. Pinecone bağlantısı
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(pinecone_index_name)

# 3. Embedding modeli
embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)

# 4. Vektör deposu
vectorstore = PineconeVectorStore(
    index_name=pinecone_index_name,
    embedding=embedding_model,
    text_key="text"
)

# 5. BM25 encoder (dummy init)
bm25 = BM25Encoder()
bm25.fit(["dummy"])

# 6. Hafıza (kullanıcı bazlı)
user_memories = {}

# 7. Proje takibi (kullanıcı bazlı)
user_projects = {}

# 8. Satış ofisi konumları
PROJECT_LOCATIONS = {
    "Mona": {
        "address": "Folkart Towers, Manas Bulvarı, Bayraklı / İzmir",
        "maps_url": "https://www.google.com/maps?q=Folkart+Towers,+Manas+Bulvar%C4%B1,+Bayrakl%C4%B1,+%C4%B0zmir"
    },
    "Nova": {
        "address": "Folkart Vega, Mersinli Mah. 2819/3 Sk. No:1 / İzmir",
        "maps_url": "https://www.google.com/maps?q=Kaz%C4%B1mdirik,+296.+Sk.+No:18,+Bornova,+İzmir"
    },
    "Terra": {
        "address": "Şifne Mah. Reisdere Mevkii No: 7 Çeşme/İzmir",
        "maps_url": "https://www.google.com/maps?q=Folkart+Terra+Projesi,+Şifne,+Çeşme,+İzmir"
    }
}

# 9. Prompt şablonu
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
    - Eğer kullanıcı proje belirtmemişse, Mona, Nova ve Terra projeleri hakkında genel bilgi ver.
    - Her Cümlenin Başına Merhaba Ekleme.
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
   - Randevu oluşturamıyorsun; iletişim bilgilerine yönlendir.  

Kontekst:
{context}

Soru:
{question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 10. Proje tespiti
def detect_project(query: str, user_id: str) -> str:
    q = query.lower()
    if "mona" in q:
        project = "Mona"
    elif "nova" in q:
        project = "Nova"
    elif "terra" in q:
        project = "Terra"
    else:
        project = user_projects.get(user_id, "Mona")  # Mona varsayılan

    user_projects[user_id] = project
    return project

# 11. Hybrid Retriever
def hybrid_retrieve(query: str, user_id: str, top_k: int = 12):
    project = detect_project(query, user_id)

    dense_q = embedding_model.embed_query(query)
    sparse_q = bm25.encode_queries([query])[0]

    index.query(
        vector=dense_q,
        sparse_vector=sparse_q,
        top_k=top_k,
        include_metadata=True,
        filter={"project": project}
    )

    docs = vectorstore.similarity_search(query, k=top_k, filter={"project": project})
    context = "\n".join([doc.page_content for doc in docs])
    return context

# 12. Konum isteğini algılayan fonksiyon
def is_location_request(query: str) -> bool:
    q = query.lower()
    return any(word in q for word in ["konum", "adres", "satış ofisi", "lokasyon", "nerede", "nasıl gelirim"])

# ✅ 13. LLM
llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    model="gpt-4.1-mini",
    temperature=0.4,
    max_tokens=400
)

# 14. Ana cevap fonksiyonu
def get_rag_answer(query, user_id="local_test"):
    if user_id not in user_memories:
        user_memories[user_id] = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=5
        )

    if is_location_request(query):
        project = detect_project(query, user_id)
        loc = PROJECT_LOCATIONS.get(project)
        if loc:
            address = loc["address"]
            maps_url = loc["maps_url"]
            html_link = f'<a href="{maps_url}" target="_blank"><b>{project} Satış Ofisi Konumu</b></a>'
            return (
                f"📍 <b>{project}</b> projesinin satış ofisi:<br>"
                f"{address}<br>"
                f"🗺️ Harita: {html_link}"
            )
        else:
            return "❌ Şu an bu proje için konum bilgisine ulaşamıyorum."

    context = hybrid_retrieve(query, user_id)
    formatted_prompt = prompt.format(context=context, question=query)
    result = llm.invoke(formatted_prompt).content
    user_memories[user_id].save_context({"question": query}, {"answer": result})
    return result




