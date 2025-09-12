from flask import Flask, request, jsonify, render_template
from query import get_rag_answer
from twilio.twiml.messaging_response import MessagingResponse

app = Flask(__name__)

# Ana sayfa (formlu HTML)
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# JSON API endpoint (Web için)
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

# Twilio WhatsApp Webhook Endpoint
@app.route("/whatsapp", methods=["POST"])
def whatsapp():
    try:
        incoming_msg = request.form.get("Body", "")
        if not incoming_msg:
            return "Boş mesaj", 400

        answer = get_rag_answer(incoming_msg)

        # Twilio yanıt formatı
        resp = MessagingResponse()
        resp.message(answer)
        return str(resp)
    except Exception as e:
        resp = MessagingResponse()
        resp.message("❌ Bir hata oluştu: " + str(e))
        return str(resp), 500

# Status kontrol
@app.route("/status", methods=["GET"])
def status():
    return jsonify({"status": "Folkart Chatbot API aktif ✅"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)





