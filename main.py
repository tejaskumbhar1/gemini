from flask import Flask, request
import google.generativeai as genai
import os

app = Flask(__name__)

# Configure Gemini API key
GEMINI_API_KEY = "AIzaSyBMyqVdwhbJ3OjLpNVfXs_8oAQ4YEOZhSw"


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

if __name__ == '__main__':
    app.run(debug=False, port=5000)