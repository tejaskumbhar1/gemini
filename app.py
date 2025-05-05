
from flask import Flask, request
import google.generativeai as genai
import os

app = Flask(__name__)

# Configure Gemini API key
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

@app.route('/query', methods=['POST'])
def ask():
    question = request.data.decode("utf-8")
    try:
        response = model.generate_content(question)
        return response.text
    except Exception as e:
        return f"Error generating response: {e}"

@app.route('/ping', methods=['GET'])
def ping():
    return "pong"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)  # Explicitly bind to 0.0.0.0    