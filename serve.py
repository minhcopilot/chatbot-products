from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
from flask_cors import CORS
from rag.core import RAG

# Load environment variables from .env file
load_dotenv()

# Access the key
MYSQL_HOST = os.getenv('MYSQL_HOST')
MYSQL_PORT = os.getenv('MYSQL_PORT')
MYSQL_USER = os.getenv('MYSQL_USER')
MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD')
MYSQL_DATABASE = os.getenv('MYSQL_DATABASE')
MYSQL_TABLE = os.getenv('MYSQL_TABLE')
LLM_KEY = os.getenv('GEMINI_KEY')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL') or 'keepitreal/vietnamese-sbert'

app = Flask(__name__)
CORS(app)

# Initialize RAG with MySQL
rag = RAG(
    mysqlHost=MYSQL_HOST,
    mysqlPort=int(MYSQL_PORT),
    mysqlUser=MYSQL_USER,
    mysqlPassword=MYSQL_PASSWORD,
    mysqlDatabase=MYSQL_DATABASE,
    mysqlTable=MYSQL_TABLE,
    llmApiKey=LLM_KEY,
    llmName='gemini-1.5-pro',
    embeddingName=EMBEDDING_MODEL
)

def process_query(query):
    return query.lower()

@app.route('/api/search', methods=['POST'])
def handle_query():
    data = request.get_json()
    query = process_query(data.get('content'))

    if not query:
        return jsonify({'error': 'No query provided'}), 400

    # Retrieve data from MySQL database using RAG
    source_information = rag.enhance_prompt(query).replace('<br>', '\n')
    combined_information = f"Hãy trở thành chuyên gia tư vấn bán hàng cho một cửa hàng bán đồ gia dụng. Câu hỏi của khách hàng: {query}\nTrả lời câu hỏi dựa vào các thông tin sản phẩm dưới đây: {source_information}."

    response = rag.generate_content(combined_information)

    return jsonify({
        'content': response.text,
        'role': 'system'
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
