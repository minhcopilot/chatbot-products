### Set up

#### 1. Installation

This code requires Python >= 3.9.

```
pip install -r requirements.txt
```

#### 2. Environment Variables

Create a file named .env and add the following lines, replacing placeholders with your actual values:

```
MYSQL_HOST=
MYSQL_PORT=
MYSQL_USER=
MYSQL_PASSWORD=
MYSQL_DATABASE=
GEMINI_KEY=
```

#### 3. Data

Prepare your data

#### 4. Edit your Prompt in serve.py

In the serve.py file, you can see that we used the prompt like this. This prompt was enhanced by adding information about your products to it.

```
 combined_information = f"Hãy trở thành chuyên gia tư vấn bán hàng cho một cửa hàng bán đồ gia dụng. Câu hỏi của khách hàng: {query}\nTrả lời câu hỏi dựa vào các thông tin sản phẩm dưới đây: {source_information}."


- query: Query from the user.
- source_information: Information we get from our database.

The full prompt will look like this:

```




The prompt is then fed to LLMs.

#### 5. Run server

```

python serve.py

```

#### 6. Testing API
http://localhost:5002/api/search
POST
{
    "content":"Kệ gầm bếp Tokyo "
}
```
