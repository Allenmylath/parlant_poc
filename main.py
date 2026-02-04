import asyncio
import os
import time
from typing import List, Dict
import traceback

from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import parlant.sdk as p

from models import (
    ChatRequest, ChatResponse, Source, TokenUsage,
    TranscribeResponse, HealthResponse
)
from tools import (
    get_rag_tool, get_translation_tool, get_transcription_tool,
    ParlantRAGTool, ParlantTranslationTool
)
import psutil

# Configuration
VERSION = "1.0.0"
START_TIME = time.time()
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("PORT", "8000"))

# Create FastAPI app directly
app = FastAPI(title="Kerala Police Assistant")

# Global agent
police_agent: p.Agent = None
parlant_server: p.Server = None

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_memory_usage() -> float:
    try:
        return psutil.Process().memory_info().rss / 1024 / 1024
    except:
        return 0.0


@app.on_event("startup")
async def startup():
    """Initialize Parlant on startup - non-blocking"""
    print(f"Starting Kerala Police Assistant v{VERSION}...")
    print(f"Server will bind to {SERVER_HOST}:{SERVER_PORT}")
    
    # Start Parlant initialization in background
    asyncio.create_task(initialize_parlant())


async def initialize_parlant():
    """Initialize Parlant in background"""
    global police_agent, parlant_server
    
    try:
        print("Initializing Parlant...")
        
        # Create Parlant server (internal only)
        parlant_server = p.Server(
            host="127.0.0.1",
            port=8818,
            nlp_service=p.NLPServices.openai,
            session_store='transient',
            customer_store='transient',
            variable_store='transient',
        )
        
        await parlant_server.ready.wait()
        print("Parlant server ready")
        
        # Create agent
        agents = await parlant_server.list_agents()
        if agents:
            police_agent = agents[0]
        else:
            police_agent = await parlant_server.create_agent(
                name="Kerala Police Assistant",
                description="AI assistant for Kerala Police"
            )
            
            await police_agent.add_guideline(
                condition="always",
                action="Use search_police_website tool. Be helpful."
            )
            
            await police_agent.add_guideline(
                condition="user message is not in English",
                action="Use translate_to_english tool first."
            )
            
            await police_agent.add_tool(ParlantRAGTool())
            await police_agent.add_tool(ParlantTranslationTool())
        
        print(f"Agent ready with {len(await police_agent.list_tools())} tools")
    except Exception as e:
        print(f"Failed to initialize Parlant: {e}")
        import traceback
        traceback.print_exc()


async def process_chat(message: str, history: List[Dict], max_sources: int = 5) -> ChatResponse:
    """Process chat"""
    start_time = time.time()
    
    if not police_agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        session_id = f"req_{int(time.time() * 1000000)}"
        session = await police_agent.create_session(
            customer_id="web_user",
            session_id=session_id
        )
        
        await session.send_message(message)
        await session.wait_for_completion()
        events = await session.get_events()
        
        assistant_message = ""
        sources = []
        detected_language = "en"
        
        for event in events:
            if hasattr(event, 'message') and event.message:
                if hasattr(event.message, 'content') and event.message.content:
                    assistant_message = event.message.content
            
            if hasattr(event, 'tool_result') and event.tool_result:
                result = event.tool_result
                
                if result.tool_name == "search_police_website":
                    try:
                        import json
                        data = json.loads(result.output) if isinstance(result.output, str) else result.output
                        if isinstance(data, list):
                            for item in data[:max_sources]:
                                sources.append(Source(
                                    content=item.get("content", "")[:300],
                                    url=item.get("url", ""),
                                    score=item.get("score", 0.0)
                                ))
                    except:
                        pass
                
                if result.tool_name == "translate_to_english":
                    try:
                        import json
                        data = json.loads(result.output) if isinstance(result.output, str) else result.output
                        detected_language = data.get("language", "en")
                    except:
                        pass
        
        try:
            await session.delete()
        except:
            pass
        
        return ChatResponse(
            response=assistant_message or "I couldn't generate a response.",
            sources=sources,
            tokens_used=TokenUsage(input_tokens=0, output_tokens=0, total_tokens=0),
            detected_language=detected_language,
            processing_time_ms=round((time.time() - start_time) * 1000, 2)
        )
    except Exception as e:
        print(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", response_model=HealthResponse)
async def root():
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
    try:
        services = {
            "openai": bool(os.getenv("OPENAI_API_KEY")),
            "sarvam": bool(os.getenv("SARVAM_API_KEY"))
        }
        
        try:
            services["qdrant"] = get_rag_tool().health_check()
        except:
            services["qdrant"] = False
        
        return HealthResponse(
            status="healthy" if all(services.values()) else "degraded",
            version=VERSION,
            services=services,
            uptime_seconds=time.time() - START_TIME,
            memory_usage_mb=get_memory_usage()
        )
    except:
        raise HTTPException(status_code=503, detail="Service unhealthy")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    return await process_chat(
        message=request.message,
        history=[msg.dict() for msg in request.history],
        max_sources=request.max_sources
    )


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(audio: UploadFile = File(...)):
    try:
        if not audio.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="Must be audio file")
        
        audio_data = await audio.read()
        
        if not audio_data or len(audio_data) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Invalid file")
        
        transcript, lang = get_transcription_tool().transcribe(audio_data)
        
        if not transcript:
            raise HTTPException(status_code=400, detail="Transcription failed")
        
        return TranscribeResponse(text=transcript, detected_language=lang)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics():
    return {
        "uptime_seconds": time.time() - START_TIME,
        "memory_usage_mb": get_memory_usage(),
        "version": VERSION
    }


if __name__ == "__main__":
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT, log_level="info")
