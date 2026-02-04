import asyncio
import os
import time
from typing import List, Dict
import traceback

import parlant.sdk as p
from fastapi import HTTPException, UploadFile, File, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

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

# Global agent reference
police_agent: p.Agent = None


def get_memory_usage() -> float:
    """Get current memory usage in MB"""
    try:
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except:
        return 0.0


async def setup_agent(server: p.Server) -> p.Agent:
    """Set up the police website assistant agent"""
    
    # Check if agent already exists
    agents = await server.list_agents()
    if agents:
        agent = agents[0]
        print(f"Using existing agent: {agent.name}")
        return agent
    
    # Create new agent
    agent = await server.create_agent(
        name="Kerala Police Assistant",
        description="AI assistant for Kerala Police website"
    )
    
    # Add guidelines
    await agent.add_guideline(
        condition="always",
        action="Use search_police_website tool to find information. Be concise and helpful."
    )
    
    await agent.add_guideline(
        condition="user message is not in English",
        action="Use translate_to_english tool first."
    )
    
    # Register tools
    rag_tool = ParlantRAGTool()
    translation_tool = ParlantTranslationTool()
    
    await agent.add_tool(rag_tool)
    await agent.add_tool(translation_tool)
    
    print(f"Agent created with {len(await agent.list_tools())} tools")
    return agent


async def process_chat_stateless(
    message: str,
    history: List[Dict],
    max_sources: int = 5
) -> ChatResponse:
    """Process chat using Parlant agent"""
    start_time = time.time()
    
    global police_agent
    
    if not police_agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        # Create temporary session
        session_id = f"req_{int(time.time() * 1000000)}"
        customer_id = "web_user"
        
        session = await police_agent.create_session(
            customer_id=customer_id,
            session_id=session_id
        )
        
        # Send message
        await session.send_message(message)
        await session.wait_for_completion()
        
        # Get events
        events = await session.get_events()
        
        # Extract response
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
                    except Exception as e:
                        print(f"Error parsing RAG results: {e}")
                
                if result.tool_name == "translate_to_english":
                    try:
                        import json
                        data = json.loads(result.output) if isinstance(result.output, str) else result.output
                        detected_language = data.get("language", "en")
                    except Exception as e:
                        print(f"Error parsing translation: {e}")
        
        # Clean up
        try:
            await session.delete()
        except:
            pass
        
        processing_time = round((time.time() - start_time) * 1000, 2)
        
        return ChatResponse(
            response=assistant_message or "I couldn't generate a response.",
            sources=sources,
            tokens_used=TokenUsage(input_tokens=0, output_tokens=0, total_tokens=0),
            detected_language=detected_language,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        print(f"Chat processing error: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


async def main():
    """Initialize and run Parlant server"""
    global police_agent
    
    print(f"Starting Kerala Police Assistant v{VERSION}...")
    
    # Use async context manager pattern (recommended by Parlant docs)
    async with p.Server(
        host=SERVER_HOST,
        port=SERVER_PORT,
        nlp_service=p.NLPServices.openai,
        session_store='transient',  # In-memory for stateless deployment
        customer_store='transient',
        variable_store='transient',
    ) as server:
        
        # Wait for server to be ready
        await server.ready.wait()
        print(f"Parlant server ready on {SERVER_HOST}:{SERVER_PORT}")
        
        # Setup agent
        police_agent = await setup_agent(server)
        
        # Get FastAPI app
        app = server.api
        
        # Add CORS
        app.add_middleware(
            CORSMiddleware,
            allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Custom endpoints
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
            """Fast health check for Render"""
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
            except Exception as e:
                raise HTTPException(status_code=503, detail="Service unhealthy")
        
        @app.post("/chat", response_model=ChatResponse)
        async def chat(request: ChatRequest):
            if not request.message.strip():
                raise HTTPException(status_code=400, detail="Message cannot be empty")
            
            return await process_chat_stateless(
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
                
                if not audio_data:
                    raise HTTPException(status_code=400, detail="Empty file")
                
                if len(audio_data) > 10 * 1024 * 1024:
                    raise HTTPException(status_code=400, detail="File too large (max 10MB)")
                
                transcript, lang = get_transcription_tool().transcribe(audio_data)
                
                if not transcript:
                    raise HTTPException(status_code=400, detail="Transcription failed")
                
                return TranscribeResponse(text=transcript, detected_language=lang)
                
            except HTTPException:
                raise
            except Exception as e:
                print(f"Transcription error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/metrics")
        async def metrics():
            return {
                "uptime_seconds": time.time() - START_TIME,
                "memory_usage_mb": get_memory_usage(),
                "version": VERSION
            }
        
        @app.exception_handler(Exception)
        async def global_exception_handler(request: Request, exc: Exception):
            print(f"Error: {exc}")
            if os.getenv("DEBUG") == "true":
                print(traceback.format_exc())
            
            return JSONResponse(
                status_code=500,
                content={
                    "error": "InternalServerError",
                    "message": str(exc) if os.getenv("DEBUG") == "true" else "An error occurred"
                }
            )
        
        print(f"Server running. Keeping process alive...")
        
        # Keep the process alive - Parlant server is already running
        # The context manager handles cleanup on exit
        try:
            while True:
                await asyncio.sleep(3600)  # Sleep for 1 hour, repeat
        except (KeyboardInterrupt, asyncio.CancelledError):
            print("Shutting down gracefully...")


if __name__ == "__main__":
    asyncio.run(main())
