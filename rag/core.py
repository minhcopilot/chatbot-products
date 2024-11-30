import numpy as np
import mysql.connector
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from IPython.display import Markdown
import textwrap
class RAG():
    def __init__(self, 
            mysqlHost: str,
            mysqlPort: int,
            mysqlUser: str,
            mysqlPassword: str,
            mysqlDatabase: str,
            mysqlTable: str,
            llmApiKey: str,
            llmName: str ='gemini-1.5-pro',
            embeddingName: str ='keepitreal/vietnamese-sbert',
        ):
        self.conn = mysql.connector.connect(
            host=mysqlHost,
            port=mysqlPort,
            user=mysqlUser,
            password=mysqlPassword,
            database=mysqlDatabase
        )
        self.table = mysqlTable
        self.embedding_model = SentenceTransformer(embeddingName)

        # config llm
        genai.configure(api_key=llmApiKey)
        self.llm = genai.GenerativeModel(llmName)
# Tạo embedding cho văn bản đầu vào sử dụng mô hình SentenceTransformer.
    def get_embedding(self, text):
        if not text.strip():
            return []

        embedding = self.embedding_model.encode(text)
        return embedding.tolist()
# Tính toán độ tương đồng cosine giữa hai vector.
    def cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
# Tìm kiếm sản phẩm từ cơ sở dữ liệu MySQL dựa trên embedding của truy vấn người dùng. Sắp xếp các sản phẩm theo độ tương đồng với truy vấn.
    def vector_search(self, user_query, limit=4):
        # Generate embedding for the user query
        query_embedding = self.get_embedding(user_query)

        if not query_embedding:
            return "Invalid query or embedding generation failed."

        # Retrieve all embeddings from the database
        query = f"""
                SELECT 
                    p.id,
                    p.name,
                    p.description,
                    p.price,
                    p.category_id,
                    p.stock,
                    p.created_at,
                    p.updated_at,
                    p.deleted_at,
                    c.name as category_name,
                    GROUP_CONCAT(DISTINCT s.name ORDER BY s.name) as sizes
                FROM products p
                FORCE INDEX (idx_deleted_at, idx_category_id)
                LEFT JOIN categories c 
                    ON p.category_id = c.id 
                    AND c.deleted_at IS NULL
                LEFT JOIN product_size ps 
                    ON p.id = ps.product_id
                LEFT JOIN sizes s 
                    ON ps.size_id = s.id 
                    AND s.deleted_at IS NULL
                WHERE p.deleted_at IS NULL
                GROUP BY p.id, p.name, p.description, p.price, p.category_id, p.stock, 
                        p.created_at, p.updated_at, p.deleted_at, c.name;
        """
        cursor = self.conn.cursor(dictionary=True)
        cursor.execute(query)
        results = cursor.fetchall()
        cursor.close()

        # Calculate cosine similarity and sort results
        for result in results:
            # Kết hợp nhiều trường thông tin thành một văn bản
            product_text = f"""
            Tên sản phẩm: {result['name']}
            Mô tả: {result['description']}
            Danh mục: {result['category_name']}
            Giá: {result['price']}
            Kích thước có sẵn: {result['sizes']}
            """
            # Tạo embedding từ văn bản kết hợp
            product_embedding = self.get_embedding(product_text)
            result['score'] = self.cosine_similarity(query_embedding, product_embedding)
        results = sorted(results, key=lambda x: x['score'], reverse=True)
        return results[:limit]
#  Tạo một prompt nâng cao từ kết quả tìm kiếm để đưa vào mô hình LLM.
    def enhance_prompt(self, query):
        get_knowledge = self.vector_search(query, 10)
        enhanced_prompt = ""
        i = 0
        for result in get_knowledge:
            if result.get('price'):
                i += 1
                enhanced_prompt += f"\n {i}) Tên: {result.get('name')}"
                
                if result.get('price'):
                    enhanced_prompt += f", Giá: {result.get('price'):,.0f}đ"
                else:
                    enhanced_prompt += f", Giá: Liên hệ để trao đổi thêm!"
                
                if result.get('category_name'):
                    enhanced_prompt += f", Danh mục: {result.get('category_name')}"
                
                if result.get('stock'):
                    enhanced_prompt += f", Số lượng tồn: {result.get('stock')}"
                
                if result.get('sizes'):
                    enhanced_prompt += f", Kích thước: {result.get('sizes')}"
                
                if result.get('description'):
                    enhanced_prompt += f"\n   Mô tả: {result.get('description')}"
        
        return enhanced_prompt
# Sử dụng mô hình LLM để tạo câu trả lời từ prompt nâng cao.
    def generate_content(self, prompt):
        return self.llm.generate_content(prompt)

    def _to_markdown(text):
        text = text.replace('•', '  *')
        return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))
