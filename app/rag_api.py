from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from google import genai
from pydantic import BaseModel

from config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
    EMBEDDING_MODEL,
    GEMINI_API_KEY
)


# =========================================================
# Create the FastAPI application
# =========================================================

app = FastAPI(
    title="Harry Potter RAG API",
    description="RAG-based chatbot for the Harry Potter books",
    version="1.0.0"
)


# =========================================================
# Connect to Qdrant Cloud
# =========================================================

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)


# =========================================================
# Connect to Gemini
# =========================================================

gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)


# =========================================================
# Load the embedding model
# =========================================================

print("Starting embedding model download/loading...")

embedding_model = SentenceTransformer(EMBEDDING_MODEL)

print("Embedding model loaded successfully!")
print("FastAPI application initialized.")
print("Connected to Qdrant Cloud.")
print("Embedding model loaded.")


# =========================================================
# Health Check
# =========================================================

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "message": "Harry Potter RAG API is running"
    }


# =========================================================
# Test Gemini
# =========================================================

@app.get("/test-llm")
def test_llm():

    response = gemini_client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents="Say hello in one short sentence."
    )

    return {
        "response": response.text
    }


# =========================================================
# Request Model
# =========================================================

class QueryRequest(BaseModel):
    question: str


# =========================================================
# Query Router
# =========================================================

def route_query(question: str):

    question_lower = question.lower().strip()

    general_phrases = [
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "thanks",
        "thank you"
    ]

    if question_lower in general_phrases:
        return "general"

    return "rag"


# =========================================================
# RAG Function
# =========================================================

def rag_answer(question: str):

    # -----------------------------------------------------
    # Create an embedding for the user's question
    # -----------------------------------------------------

    query_text = "query: " + question

    query_embedding = embedding_model.encode(
        query_text
    ).tolist()


    # -----------------------------------------------------
    # Search Qdrant for relevant chunks
    # -----------------------------------------------------

    search_results = qdrant_client.query_points(
        collection_name=QDRANT_COLLECTION_NAME,
        query=query_embedding,
        limit=8
    ).points


    # -----------------------------------------------------
    # Prepare the retrieved sources
    # -----------------------------------------------------

    sources = []

    for result in search_results:

        sources.append({
            "book": result.payload["book_name"],
            "page": result.payload["page_number"],
            "content": result.payload["content"],
            "score": result.score
        })


    # -----------------------------------------------------
    # Combine retrieved chunks into context
    # -----------------------------------------------------

    context = "\n\n".join(
        f"Book: {source['book']}, Page: {source['page']}\n"
        f"{source['content']}"
        for source in sources
    )


    # -----------------------------------------------------
    # Create the RAG prompt
    # -----------------------------------------------------

    prompt = f"""
You are a helpful Harry Potter book assistant.

Answer the user's question using ONLY the information provided
in the retrieved context below.

Give a direct and complete answer. If the context contains
specific names, events, reasons, or details that answer the
question, include those details in your answer.

Do not invent information that is not supported by the context.

If the answer cannot be found in the context, say:
"I could not find the answer in the provided Harry Potter books."

User question:
{question}

Retrieved context:
{context}
"""


    # -----------------------------------------------------
    # Generate answer using Gemini
    # -----------------------------------------------------

    print("Sending RAG prompt to Gemini...")

    response = gemini_client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    print("Gemini response received!")


    return {
        "answer": response.text,
        "sources": sources
    }

# =========================================================
# Main Query Endpoint
# =========================================================

@app.post("/query")
def query(request: QueryRequest):

    route = route_query(request.question)

    print(f"Query route: {route}")


    # General conversation
    if route == "general":

        response = gemini_client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=request.question
        )

        return {
            "question": request.question,
            "route": "general",
            "answer": response.text,
            "sources": []
        }


    # RAG route
    result = rag_answer(request.question)

    return {
        "question": request.question,
        "route": "rag",
        "answer": result["answer"],
        "sources": result["sources"]
    }


# =========================================================
# User Interface
# =========================================================

@app.get("/", response_class=HTMLResponse)
def home():

    return """
<!DOCTYPE html>
<html lang="en">

<head>

    <meta charset="UTF-8">

    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <title>Harry Potter RAG Assistant</title>

    <style>

        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            padding: 35px 20px;
            font-family: Arial, sans-serif;
            background: #11182f;
            color: #222;
        }

        .container {
            max-width: 850px;
            margin: auto;
            background: white;
            border-radius: 18px;
            padding: 32px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.25);
        }

        .header {
            text-align: center;
            margin-bottom: 25px;
        }

        .header h1 {
            margin: 0 0 8px;
            font-size: 32px;
        }

        .header p {
            margin: 0;
            font-size: 16px;
            color: #666;
        }

        .input-row {
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
        }

        #question {
            flex: 1;
            padding: 14px 16px;
            border: 2px solid #ddd;
            border-radius: 10px;
            font-size: 16px;
            outline: none;
        }

        #question:focus {
            border-color: #777;
        }

        #askButton {
            padding: 0 24px;
            border: none;
            border-radius: 10px;
            background: #222;
            color: white;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
        }

        #askButton:hover {
            background: #444;
        }

        .health-row {
            text-align: center;
            margin-bottom: 25px;
        }

        #healthButton {
            padding: 9px 16px;
            border: 1px solid #ccc;
            border-radius: 8px;
            background: white;
            cursor: pointer;
            font-size: 14px;
        }

        #healthStatus {
            margin-top: 8px;
            font-size: 14px;
            font-weight: bold;
        }

        .section {
            margin-top: 22px;
        }

        .section h2 {
            font-size: 22px;
            margin-bottom: 12px;
        }

        .answer-box {
            background: #f5f5f5;
            border-radius: 12px;
            padding: 20px;
            line-height: 1.6;
            font-size: 16px;
            min-height: 70px;
        }

        .source-card {
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 16px;
            margin-bottom: 12px;
            background: #fafafa;
        }

        .source-book {
            font-size: 17px;
            font-weight: bold;
            margin-bottom: 5px;
        }

        .source-page {
            color: #666;
            font-size: 14px;
            margin-bottom: 10px;
        }

        .source-content {
            font-size: 14px;
            line-height: 1.5;
        }

        .score {
            margin-top: 8px;
            color: #777;
            font-size: 13px;
        }

        .loading {
            color: #666;
            font-style: italic;
        }

        @media (max-width: 700px) {

            .container {
                padding: 22px;
            }

            .header h1 {
                font-size: 26px;
            }

            .input-row {
                flex-direction: column;
            }

            #askButton {
                padding: 12px;
            }

        }

    </style>

</head>


<body>

<div class="container">

    <div class="header">

        <h1>📚 Harry Potter RAG Assistant</h1>

        <p>Ask questions about the Harry Potter books</p>

    </div>


    <div class="input-row">

        <input
            id="question"
            type="text"
            placeholder="Ask a question about Harry Potter..."
        >

        <button id="askButton" onclick="askQuestion()">
            Ask
        </button>

    </div>


    <div class="health-row">

        <button id="healthButton" onclick="checkHealth()">
            🔍 Check Server Health
        </button>

        <div id="healthStatus"></div>

    </div>


    <div class="section">

        <h2>Answer</h2>

        <div id="answer" class="answer-box">
            Ask a question to get an answer.
        </div>

    </div>


    <div class="section">

        <h2>📚 Sources</h2>

        <div id="sources">
            Sources will appear here.
        </div>

    </div>

</div>


<script>

async function askQuestion() {

    const questionInput = document.getElementById("question");
    const question = questionInput.value.trim();

    if (!question) {
        alert("Please enter a question.");
        return;
    }

    const answerBox = document.getElementById("answer");
    const sourcesBox = document.getElementById("sources");
    const askButton = document.getElementById("askButton");

    answerBox.innerHTML = '<span class="loading">Thinking...</span>';
    sourcesBox.innerHTML = "";
    askButton.disabled = true;

    try {

        const response = await fetch("/query", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                question: question
            })

        });


        const data = await response.json();


        if (!response.ok) {
            throw new Error(data.detail || "Request failed");
        }


        answerBox.textContent = data.answer;


        if (data.sources && data.sources.length > 0) {

            data.sources.forEach(source => {

                const card = document.createElement("div");

                card.className = "source-card";

                card.innerHTML = `
                    <div class="source-book">
                        ${source.book}
                    </div>

                    <div class="source-page">
                        Page ${source.page}
                    </div>

                    <div class="source-content">
                        ${source.content}
                    </div>

                    <div class="score">
                        Similarity score: ${source.score.toFixed(4)}
                    </div>
                `;

                sourcesBox.appendChild(card);

            });

        } else {

            sourcesBox.innerHTML = "No sources available for this question.";

        }


    } catch (error) {

        answerBox.textContent =
            "An error occurred: " + error.message;

    }


    askButton.disabled = false;

}


async function checkHealth() {

    const status = document.getElementById("healthStatus");

    status.textContent = "Checking...";

    try {

        const response = await fetch("/health");

        const data = await response.json();

        if (response.ok) {

            status.textContent = "✅ Server is healthy";

        } else {

            status.textContent = "❌ Server error";

        }

    } catch (error) {

        status.textContent = "❌ Server is not reachable";

    }

}


document.getElementById("question").addEventListener(
    "keydown",
    function(event) {

        if (event.key === "Enter") {
            askQuestion();
        }

    }
);

</script>

</body>

</html>
"""


# =========================================================
# Run the application
# =========================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )