from flask import Flask, request, jsonify,render_template
from langchain_huggingface import HuggingFaceEndpoint
import os

app = Flask(__name__)

# Set the Hugging Face API token
sec_key = os.getenv('HUGGINGFACEHUB_API_TOKEN') 

# Initialize the Hugging Face endpoint
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=128, temperature=0.7, token=sec_key)
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    # Get response from the LLM
    response = llm.invoke(user_input)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)

