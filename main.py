from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from models import (
    ChatRequest, ChatResponse, Source, TokenUsage,
    TranscribeResponse, HealthResponse, ErrorResponse
)
from tools import get_rag_tool, get_translation_tool, get_transcription_tool
from openai import OpenAI
import os
from typing import List
import time
import psutil
import traceback

# Configuration
VERSION = "1.0.0"
START_TIME = time.time()

# Initialize FastAPI app
app = FastAPI(
    title="Police Website Assistant API",
    description="Stateless backend for Police Website Chatbot with RAG",
    version=VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware - configure for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize tools (cached singletons)
rag_tool = get_rag_tool()
translation_tool = get_translation_tool()
transcription_tool = get_transcription_tool()

# Configuration
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o")
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "10"))


def get_memory_usage() -> float:
    """Get current memory usage in MB"""
    try:
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    except:
        return 0.0


def generate_answer(query: str, search_results: List[dict], history: List[dict]) -> tuple:
    """
    Generate answer using GPT with RAG results and conversation history
    
    Args:
        query: User's question (in English)
        search_results: List of relevant documents from RAG
        history: Conversation history
        
    Returns:
        Tuple of (answer, token_usage_dict)
    """
    
    # Format RAG context
    context = "\n\n".join([
        f"Source: {r['url']}\nContent: {r['content'][:500]}...\nRelevance: {r['score']:.3f}"
        for r in search_results[:5]  # Use top 5 sources
    ])
    
    # Build messages with system prompt
    messages = [
        {
            "role": "system",
            "content": """You are a helpful AI assistant for a police website in India.

Your responsibilities:
- Answer questions about police services, procedures, and information
- Use the provided context from the police website to give accurate answers
- Cite source URLs naturally in your responses (e.g., "According to [service page]...")
- Be concise, clear, and helpful
- If the context doesn't contain relevant information, politely say so
- Always respond in English (translation is handled separately)
- Be professional yet friendly and approachable
- Focus on helping citizens access police services efficiently

Guidelines:
- Provide actionable information when possible
- If a question requires visiting a police station or calling, mention that
- For emergency situations, always mention emergency helplines (100, 112)
- Be sensitive to potentially stressful situations citizens may be facing"""
        }
    ]
    
    # Add conversation history (limited to last N messages)
    history_to_include = history[-MAX_HISTORY_MESSAGES:] if history else []
    for msg in history_to_include:
        messages.append({
            "role": msg.get("role", "user"),
            "content": msg.get("content", "")
        })
    
    # Add current query with RAG context
    messages.append({
        "role": "user",
        "content": f"""Question: {query}

Relevant information from police website:
{context}

Please answer the question based on the information provided above. If the information isn't sufficient, let me know what additional details I should look for."""
    })
    
    # Generate response
    response = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.7,
        max_tokens=800
    )
    
    answer = response.choices[0].message.content
    token_usage = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens
    }
    
    return answer, token_usage


# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    error_detail = traceback.format_exc() if os.getenv("DEBUG") == "true" else str(exc)
    print(f"Error: {error_detail}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred",
            "detail": str(exc) if os.getenv("DEBUG") == "true" else None
        }
    )


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - basic health check"""
    return HealthResponse(
        status="running",
        version=VERSION,
        services={
            "openai": bool(os.getenv("OPENAI_API_KEY")),
            "qdrant": bool(os.getenv("QDRANT_URL")),
            "sarvam": bool(os.getenv("SARVAM_API_KEY"))
        },
        uptime_seconds=time.time() - START_TIME,
        memory_usage_mb=get_memory_usage()
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Health check endpoint for Render
    Tests connectivity to all dependent services
    """
    try:
        services = {
            "openai": bool(os.getenv("OPENAI_API_KEY")),
            "sarvam": bool(os.getenv("SARVAM_API_KEY"))
        }
        
        # Test Qdrant connectivity
        try:
            services["qdrant"] = rag_tool.health_check()
        except Exception as e:
            print(f"Qdrant health check error: {e}")
            services["qdrant"] = False
        
        # Determine overall status
        all_healthy = all(services.values())
        status = "healthy" if all_healthy else "degraded"
        
        return HealthResponse(
            status=status,
            version=VERSION,
            services=services,
            uptime_seconds=time.time() - START_TIME,
            memory_usage_mb=get_memory_usage()
        )
    except Exception as e:
        print(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint - completely stateless
    
    Receives:
    - Current user message
    - Conversation history from Streamlit
    - Optional language hint
    
    Returns:
    - Assistant response
    - Sources used
    - Token usage
    - Processing metrics
    """
    start_time = time.time()
    
    try:
        # Validate input
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Step 1: Translate query to English if needed
        english_query, detected_lang = translation_tool.translate_to_english(request.message)
        
        if not english_query.strip():
            raise HTTPException(status_code=400, detail="Translation failed")
        
        # Step 2: Search RAG for relevant content
        search_results = rag_tool.search(english_query, top_k=request.max_sources)
        
        if not search_results:
            # No relevant content found
            return ChatResponse(
                response="I couldn't find relevant information in the police website database to answer your question. Please try rephrasing your question or contact the police directly for assistance.",
                sources=[],
                tokens_used=TokenUsage(input_tokens=0, output_tokens=0, total_tokens=0),
                detected_language=detected_lang,
                processing_time_ms=round((time.time() - start_time) * 1000, 2)
            )
        
        # Step 3: Generate answer with context
        answer, token_usage = generate_answer(
            english_query, 
            search_results,
            request.history
        )
        
        # Step 4: Format response
        sources = [
            Source(
                content=r["content"][:300] + "..." if len(r["content"]) > 300 else r["content"],
                url=r["url"],
                score=r["score"]
            )
            for r in search_results
        ]
        
        processing_time = round((time.time() - start_time) * 1000, 2)
        
        return ChatResponse(
            response=answer,
            sources=sources,
            tokens_used=TokenUsage(**token_usage),
            detected_language=detected_lang,
            processing_time_ms=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Chat error: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(audio: UploadFile = File(...)):
    """
    Transcribe audio file to text
    
    Accepts:
    - Audio file upload (WAV, MP3, etc.)
    
    Returns:
    - Transcribed text
    - Detected language
    """
    try:
        # Validate file
        if not audio.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="File must be an audio file")
        
        # Read audio data
        audio_data = await audio.read()
        
        if len(audio_data) == 0:
            raise HTTPException(status_code=400, detail="Audio file is empty")
        
        if len(audio_data) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(status_code=400, detail="Audio file too large (max 10MB)")
        
        # Transcribe
        transcript, detected_lang = transcription_tool.transcribe(audio_data)
        
        if not transcript:
            raise HTTPException(status_code=400, detail="Transcription failed - could not extract text from audio")
        
        return TranscribeResponse(
            text=transcript,
            detected_language=detected_lang
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Transcription error: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@app.get("/metrics")
async def metrics():
    """
    Basic metrics endpoint for monitoring
    """
    return {
        "uptime_seconds": time.time() - START_TIME,
        "memory_usage_mb": get_memory_usage(),
        "version": VERSION
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=port,
        log_level="info"
    )
