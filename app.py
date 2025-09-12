from flask import Flask, request, jsonify, render_template
from query import get_rag_answer
from twilio.twiml.messaging_response import MessagingResponse

app = Flask(__name__)

# Ana sayfa (formlu HTML)
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# JSON API endpoint (web arayüzünden gelen sorular için)
@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({"error": "Geçersiz istek: 'question' alanı gerekli."}), 400

        question = data['question']
        response = get_rag_answer(question)
        return jsonify({"answer": response})
    except Exception as e:
        return jsonify({"error": f"❌ Sunucu hatası: {str(e)}"}), 500

# Twilio WhatsApp webhook endpoint
@app.route("/whatsapp", methods=["POST"])
def whatsapp_webhook():
    incoming_msg = request.form.get("Body")
    if not incoming_msg:
        return "No message received", 400

    try:
        response_text = get_rag_answer(incoming_msg)
    except Exception as e:
        response_text = f"❌ Sistem hatası: {str(e)}"

    resp = MessagingResponse()
    msg = resp.message()
    msg.body(response_text)
    return str(resp)

# Sağlık kontrol endpoint'i
@app.route("/status", methods=["GET"])
def status():
    return jsonify({"status": "Folkart Chatbot API aktif ✅"})

# Render'da gunicorn ile çalışacaksa bu blok gerekmez ama lokal çalışmada lazım
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)





