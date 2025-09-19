from flask import Flask, request, jsonify, render_template
from query import get_rag_answer
from twilio.twiml.messaging_response import MessagingResponse
import logging
from logging.handlers import RotatingFileHandler

app = Flask(__name__)

# ---------------------------
# Yardımcı fonksiyonlar
# ---------------------------

def _get_client_ip():
    xff = request.headers.get("X-Forwarded-For", "")
    if xff:
        return xff.split(",")[0].strip()
    return request.remote_addr or "0.0.0.0"

def _extract_user_id_for_web(data):
    user_id = None
    if isinstance(data, dict):
        user_id = data.get("user_id")
    if not user_id:
        user_id = request.args.get("user_id")
    if not user_id:
        user_id = request.headers.get("X-User-Id")
    return (user_id or "local_test").strip()

def _extract_user_id_for_twilio():
    from_id = request.form.get("From")
    waid = request.form.get("WaId")
    user_id = from_id or (f"whatsapp:{waid}" if waid else None) or "unknown_user"
    return user_id.strip()

# ---------------------------
# Routes
# ---------------------------

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json(silent=True) or {}
        question = (data.get("question") or "").strip()
        if not question:
            return jsonify({"error": "Geçersiz istek: 'question' alanı gerekli."}), 400

        user_id = _extract_user_id_for_web(data)
        app.logger.info(f"[WEB] user_id={user_id} ip={_get_client_ip()} q='{question[:80]}...'")

        answer = get_rag_answer(question, user_id=user_id)
        return jsonify({"answer": answer, "user_id": user_id})
    except Exception as e:
        app.logger.exception("ASK endpoint error")
        return jsonify({"error": f"❌ Sunucu hatası: {str(e)}"}), 500

@app.route("/whatsapp", methods=["POST"])
def whatsapp():
    try:
        incoming_msg = (request.form.get("Body") or "").strip()
        if not incoming_msg:
            return "Boş mesaj", 400

        user_id = _extract_user_id_for_twilio()
        app.logger.info(f"[WHATSAPP] user_id={user_id} ip={_get_client_ip()} q='{incoming_msg[:80]}...'")

        answer = get_rag_answer(incoming_msg, user_id=user_id)

        resp = MessagingResponse()
        resp.message(answer)
        return str(resp)
    except Exception as e:
        app.logger.exception("WHATSAPP endpoint error")
        resp = MessagingResponse()
        resp.message("❌ Bir hata oluştu: " + str(e))
        return str(resp), 500

@app.route("/status", methods=["GET"])
def status():
    return jsonify({"status": "Folkart Chatbot API aktif ✅"})

if __name__ == "__main__":
    handler = RotatingFileHandler("chatbot.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    handler.setLevel(logging.INFO)
    app.logger.addHandler(handler)
    app.config["JSON_AS_ASCII"] = False

    app.run(host="0.0.0.0", port=5000, debug=True)









