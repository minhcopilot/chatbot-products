from flask import Flask, request, jsonify, session
from dotenv import load_dotenv
import os
from flask_cors import CORS
from rag.core import RAG
from datetime import datetime

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
app.secret_key = 'your-secret-key-here'
CORS(app, supports_credentials=True)

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

def format_chat_history(chat_history):
    formatted_history = ""
    for message in chat_history:
        role = "Khách hàng" if message['role'] == 'user' else "Tư vấn viên"
        formatted_history += f"{role}: {message['content']}\n"
    return formatted_history

@app.route('/chatbot', methods=['POST'])
def handle_query():
    data = request.get_json()
    query = process_query(data.get('content'))
    chat_history = data.get('chatHistory', [])

    if not query:
        return jsonify({'error': 'No query provided'}), 400

    source_information = rag.enhance_prompt(query).replace('<br>', '\n')
    combined_information = f"""
        Bạn là trợ lý tư vấn giày thông minh trên website cửa hàng RENO. Hãy trả lời dựa trên cuộc trò chuyện và thông tin sau:

        LỊCH SỬ TRÒ CHUYỆN:
        {format_chat_history(chat_history)}

        CÂU HỎI HIỆN TẠI: {query}

        THÔNG TIN SẢN PHẨM:
        {source_information}

        Yêu cầu khi trả lời:
        1. Phân tích context của cuộc trò chuyện để hiểu đúng nhu cầu khách hàng
        2. Trả lời ngắn gọn, tối đa 3-4 sản phẩm phù hợp nhất
        3. Mỗi sản phẩm chỉ nêu: tên, giá, size và 1-2 ưu điểm nổi bật
        4. Nếu khách đã hỏi về sản phẩm trước đó, hãy tập trung trả lời về sản phẩm đó
        5. Nếu cần thêm thông tin, hỏi ngắn gọn 1-2 câu
        """
    response = rag.generate_content(combined_information)

    response_data = {
        'replyContent': response.text,
        'role': 'system',
        'timestamp': datetime.now().isoformat()
    }

    return jsonify(response_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
