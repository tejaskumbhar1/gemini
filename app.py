from flask import Flask, request
import google.generativeai as genai

app = Flask(__name__)

# Configure Gemini API key
genai.configure(api_key="AIzaSyBMyqVdwhbJ3OjLpNVfXs_8oAQ4YEOZhSw")
model = genai.GenerativeModel('gemini-2.0-flash')

# Route 1: Accepts a POST request with a question and returns Gemini's response
@app.route('/query', methods=['POST'])
def ask():
    question = request.data.decode("utf-8")
    response = model.generate_content(question)
    return response.text

# Route 2: Returns a static paragraph of simple text
@app.route('/intro', methods=['GET'])
def intro():
    return "Welcome to the Gemini API server! This service lets you ask natural language questions and get AI-generated responses. Use /query to ask your question via a POST request."
