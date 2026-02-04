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
        print("=" * 60)
        print("PARLANT INITIALIZATION STARTED")
        print("=" * 60)
        
        # Create Parlant server (internal only)
        print("Step 1: Creating Parlant server...")
        parlant_server = p.Server(
            host="127.0.0.1",
            port=8818,
            nlp_service=p.NLPServices.openai,
            session_store='transient',
            customer_store='transient',
            variable_store='transient',
        )
        print("✓ Parlant server object created")
        
        print("Step 2: Waiting for server to be ready...")
        await parlant_server.ready.wait()
        print("✓ Parlant server is ready")
        
        # List existing agents
        print("Step 3: Listing existing agents...")
        agents = await parlant_server.list_agents()
        print(f"✓ Found {len(agents)} existing agent(s)")
        
        if agents:
            print(f"Step 4: Using existing agent: {agents[0].name}")
            police_agent = agents[0]
            print(f"✓ Agent assigned: {police_agent.name} (ID: {police_agent.id})")
        else:
            print("Step 4: No existing agents found, creating new agent...")
            
            try:
                police_agent = await parlant_server.create_agent(
                    name="Kerala Police Assistant",
                    description="AI assistant for Kerala Police"
                )
                print(f"✓ Agent created: {police_agent.name} (ID: {police_agent.id})")
            except Exception as create_error:
                print(f"✗ AGENT CREATION FAILED:")
                print(f"  Error type: {type(create_error).__name__}")
                print(f"  Error message: {str(create_error)}")
                traceback.print_exc()
                raise
            
            # Add guidelines
            print("Step 5: Adding guidelines...")
            try:
                await police_agent.add_guideline(
                    condition="always",
                    action="Use search_police_website tool. Be helpful."
                )
                print("✓ Guideline 1 added")
                
                await police_agent.add_guideline(
                    condition="user message is not in English",
                    action="Use translate_to_english tool first."
                )
                print("✓ Guideline 2 added")
            except Exception as guideline_error:
                print(f"✗ GUIDELINE ADDITION FAILED: {guideline_error}")
                traceback.print_exc()
                # Continue anyway - agent is created
            
            # Add tools
            print("Step 6: Adding tools...")
            try:
                await police_agent.add_tool(ParlantRAGTool())
                print("✓ RAG tool added")
                
                await police_agent.add_tool(ParlantTranslationTool())
                print("✓ Translation tool added")
            except Exception as tool_error:
                print(f"✗ TOOL ADDITION FAILED: {tool_error}")
                traceback.print_exc()
                # Continue anyway - agent is created
        
        # Verify tools
        print("Step 7: Verifying agent setup...")
        tools = await police_agent.list_tools()
        print(f"✓ Agent has {len(tools)} tool(s)")
        for tool in tools:
            print(f"  - {tool.name}")
        
        # Final verification
        print("Step 8: Final verification...")
        print(f"  police_agent is None: {police_agent is None}")
        print(f"  parlant_server is None: {parlant_server is None}")
        
        if police_agent is None:
            print("✗ CRITICAL: police_agent is still None after initialization!")
            raise RuntimeError("Agent initialization failed - police_agent is None")
        
        print("=" * 60)
        print("PARLANT INITIALIZATION COMPLETE ✓")
        print(f"Agent: {police_agent.name} (ID: {police_agent.id})")
        print("=" * 60)
        
    except Exception as e:
        print("=" * 60)
        print("PARLANT INITIALIZATION FAILED ✗")
        print("=" * 60)
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        print("=" * 60)
        
        # Set both to None to indicate failure
        police_agent = None
        parlant_server = None


async def process_chat(message: str, history: List[Dict], max_sources: int = 5) -> ChatResponse:
    """Process chat"""
    start_time = time.time()
    
    if not police_agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        session_id = f"req_{int(time.time() * 1000000)}"
        
        print(f"Creating session: {session_id}")
        session = await police_agent.create_session(
            customer_id="web_user",
            session_id=session_id
        )
        print(f"✓ Session created")
        
        print(f"Sending message: {message[:50]}...")
        await session.send_message(message)
        print(f"✓ Message sent")
        
        print(f"Waiting for completion...")
        await session.wait_for_completion()
        print(f"✓ Processing complete")
        
        print(f"Fetching events...")
        events = await session.get_events()
        print(f"✓ Got {len(events)} events")
        
        assistant_message = ""
        sources = []
        detected_language = "en"
        
        for i, event in enumerate(events):
            print(f"Event {i}: {type(event).__name__}")
            
            if hasattr(event, 'message') and event.message:
                if hasattr(event.message, 'content') and event.message.content:
                    assistant_message = event.message.content
                    print(f"  - Got assistant message: {len(assistant_message)} chars")
            
            if hasattr(event, 'tool_result') and event.tool_result:
                result = event.tool_result
                print(f"  - Tool result: {result.tool_name}")
                
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
                            print(f"  - Extracted {len(sources)} sources")
                    except Exception as e:
                        print(f"  - Failed to parse RAG results: {e}")
                
                if result.tool_name == "translate_to_english":
                    try:
                        import json
                        data = json.loads(result.output) if isinstance(result.output, str) else result.output
                        detected_language = data.get("language", "en")
                        print(f"  - Detected language: {detected_language}")
                    except Exception as e:
                        print(f"  - Failed to parse translation: {e}")
        
        print(f"Cleaning up session...")
        try:
            await session.delete()
            print(f"✓ Session deleted")
        except Exception as e:
            print(f"  - Session cleanup failed: {e}")
        
        response = ChatResponse(
            response=assistant_message or "I couldn't generate a response.",
            sources=sources,
            tokens_used=TokenUsage(input_tokens=0, output_tokens=0, total_tokens=0),
            detected_language=detected_language,
            processing_time_ms=round((time.time() - start_time) * 1000, 2)
        )
        
        print(f"✓ Chat complete in {response.processing_time_ms}ms")
        return response
        
    except Exception as e:
        print(f"✗ Chat error: {e}")
        traceback.print_exc()
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
    
    # Check if agent is ready
    if police_agent is None:
        print("WARNING: Chat request received but agent not initialized yet")
        raise HTTPException(
            status_code=503, 
            detail="Agent is still initializing. Please wait a moment and try again."
        )
    
    try:
        return await process_chat(
            message=request.message,
            history=[msg.dict() for msg in request.history],
            max_sources=request.max_sources
        )
    except Exception as e:
        print(f"Chat endpoint error: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Chat processing failed: {str(e)}"
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


@app.get("/agent-status")
async def agent_status():
    """Debug endpoint to check agent initialization status"""
    status = {
        "agent_initialized": police_agent is not None,
        "parlant_server_initialized": parlant_server is not None,
        "uptime_seconds": time.time() - START_TIME,
    }
    
    # Add agent details if available
    if police_agent:
        try:
            tools = await police_agent.list_tools()
            status["agent_details"] = {
                "name": police_agent.name,
                "id": police_agent.id,
                "tools_count": len(tools),
                "tools": [tool.name for tool in tools]
            }
        except Exception as e:
            status["agent_details_error"] = str(e)
    
    return status


if __name__ == "__main__":
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT, log_level="info")
