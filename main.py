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


@app.on_event("shutdown")
async def shutdown():
    """Cleanup Parlant on shutdown"""
    print("Shutting down application...")
    if parlant_server:
        try:
            await parlant_server.__aexit__(None, None, None)
        except:
            pass


async def initialize_parlant():
    """Initialize Parlant in background using proper async context manager"""
    global police_agent, parlant_server
    
    try:
        print("=" * 60)
        print("PARLANT INITIALIZATION STARTED")
        print("=" * 60)
        
        print("Step 1: Creating Parlant server with async context manager...")
        
        # Use async with as shown in the healthcare example
        server = p.Server(
            host="127.0.0.1",
            port=8818,
            nlp_service=p.NLPServices.openai,
            session_store='transient',
            customer_store='transient',
            variable_store='transient',
        )
        
        # Enter the context manager
        await server.__aenter__()
        parlant_server = server
        print("✓ Parlant server context entered")
        
        print("Step 2: Creating agent...")
        try:
            # Use create_agent as shown in healthcare example
            agent = await parlant_server.create_agent(
                name="Kerala Police Assistant",
                description="A helpful AI assistant for Kerala Police services. Provides accurate information about police procedures, emergency contacts, and services."
            )
            police_agent = agent
            print(f"✓ Agent created: {police_agent.name}")
        except Exception as create_error:
            print(f"✗ AGENT CREATION FAILED:")
            print(f"  Error type: {type(create_error).__name__}")
            print(f"  Error message: {str(create_error)}")
            traceback.print_exc()
            raise
        
        # Add guidelines (use create_guideline, not add_guideline)
        print("Step 3: Adding guidelines...")
        try:
            await police_agent.create_guideline(
                condition="always",
                action="Search the Kerala Police website database to provide accurate information. Be helpful and informative."
            )
            print("✓ Guideline 1 added")
            
            await police_agent.create_guideline(
                condition="The user's message is not in English",
                action="First detect and translate the user's message to English, then proceed with answering their question."
            )
            print("✓ Guideline 2 added")
            
            await police_agent.create_guideline(
                condition="The user asks about emergency services or urgent situations",
                action="Provide emergency contact numbers immediately and prioritize their safety."
            )
            print("✓ Guideline 3 added")
            
            await police_agent.create_guideline(
                condition="The user asks about something unrelated to Kerala Police services",
                action="Politely inform them that you can only assist with Kerala Police-related inquiries."
            )
            print("✓ Guideline 4 added")
        except Exception as guideline_error:
            print(f"⚠ GUIDELINE ADDITION FAILED: {guideline_error}")
            traceback.print_exc()
        
        print("Step 4: Tools registered via decorators")
        print("✓ RAG and Translation tools available")
        
        print("=" * 60)
        print("PARLANT INITIALIZATION COMPLETE ✓")
        print(f"Agent: {police_agent.name}")
        print("=" * 60)
        
        # Keep the context alive indefinitely
        await asyncio.Event().wait()
        
    except asyncio.CancelledError:
        print("Parlant initialization cancelled")
        if parlant_server:
            try:
                await parlant_server.__aexit__(None, None, None)
            except:
                pass
    except Exception as e:
        print("=" * 60)
        print("PARLANT INITIALIZATION FAILED ✗")
        print("=" * 60)
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        print("=" * 60)
        
        if parlant_server:
            try:
                await parlant_server.__aexit__(None, None, None)
            except:
                pass
        
        police_agent = None
        parlant_server = None


async def process_chat(message: str, history: List[Dict], max_sources: int = 5) -> ChatResponse:
    """Process chat using Parlant agent"""
    start_time = time.time()
    
    if not police_agent or not parlant_server:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        # Create a unique customer ID for this session
        customer_id = f"web_user_{int(time.time() * 1000)}"
        
        print(f"Processing message for customer: {customer_id}")
        print(f"Message: {message[:100]}...")
        
        # Create a customer through the SERVER (not the agent)
        try:
            customer = await parlant_server.create_customer(customer_id=customer_id)
            print(f"✓ Customer created: {customer_id}")
        except Exception as e:
            print(f"Customer creation note: {e}")
            # Customer might already exist, that's okay
        
        # Create a session through the SERVER with the agent ID
        try:
            session = await parlant_server.create_session(
                customer_id=customer_id,
                agent_id=police_agent.id
            )
            print(f"✓ Session created: {session.id}")
        except Exception as e:
            print(f"✗ Session creation failed: {e}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")
        
        # Send the message through the session
        try:
            await session.send_message(message)
            print(f"✓ Message sent")
        except Exception as e:
            print(f"✗ Failed to send message: {e}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Failed to send message: {str(e)}")
        
        # Wait for the agent to respond
        try:
            await session.wait_for_completion()
            print(f"✓ Agent processing complete")
        except Exception as e:
            print(f"✗ Failed waiting for completion: {e}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Agent processing failed: {str(e)}")
        
        # Get the events from the session
        try:
            events = await session.get_events()
            print(f"✓ Got {len(events)} events")
        except Exception as e:
            print(f"✗ Failed to get events: {e}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Failed to retrieve events: {str(e)}")
        
        # Extract the assistant's response and tool results
        assistant_message = ""
        sources = []
        detected_language = "en"
        
        for i, event in enumerate(events):
            event_type = type(event).__name__
            print(f"Event {i}: {event_type}")
            
            # Check for agent message
            if hasattr(event, 'message') and event.message:
                if hasattr(event.message, 'content') and event.message.content:
                    assistant_message = event.message.content
                    print(f"  - Got assistant message: {len(assistant_message)} chars")
            
            # Check for tool results
            if hasattr(event, 'tool_result') and event.tool_result:
                result = event.tool_result
                tool_name = result.tool_name if hasattr(result, 'tool_name') else 'unknown'
                print(f"  - Tool result from: {tool_name}")
                
                # Extract RAG sources
                if tool_name == "search_police_website":
                    try:
                        import json
                        data = result.data if hasattr(result, 'data') else (result.output if hasattr(result, 'output') else None)
                        if isinstance(data, str):
                            data = json.loads(data)
                        
                        if isinstance(data, list):
                            for item in data[:max_sources]:
                                if isinstance(item, dict):
                                    sources.append(Source(
                                        content=item.get("content", "")[:300],
                                        url=item.get("url", ""),
                                        score=item.get("score", 0.0)
                                    ))
                            print(f"  - Extracted {len(sources)} sources")
                    except Exception as e:
                        print(f"  - Failed to parse RAG results: {e}")
                
                # Extract language detection
                if tool_name == "translate_to_english":
                    try:
                        import json
                        data = result.data if hasattr(result, 'data') else (result.output if hasattr(result, 'output') else None)
                        if isinstance(data, str):
                            data = json.loads(data)
                        if isinstance(data, dict):
                            detected_language = data.get("language", "en")
                            print(f"  - Detected language: {detected_language}")
                    except Exception as e:
                        print(f"  - Failed to parse translation: {e}")
        
        # Cleanup session
        print(f"Cleaning up session...")
        try:
            await session.delete()
            print(f"✓ Session deleted")
        except Exception as e:
            print(f"  - Session cleanup warning: {e}")
        
        response = ChatResponse(
            response=assistant_message or "I couldn't generate a response. Please try again.",
            sources=sources,
            tokens_used=TokenUsage(input_tokens=0, output_tokens=0, total_tokens=0),
            detected_language=detected_language,
            processing_time_ms=round((time.time() - start_time) * 1000, 2)
        )
        
        print(f"✓ Chat complete in {response.processing_time_ms}ms")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"✗ Chat error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")


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
    except HTTPException:
        raise
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
            status["agent_details"] = {
                "name": police_agent.name if hasattr(police_agent, 'name') else "unknown",
                "id": police_agent.id if hasattr(police_agent, 'id') else "unknown",
            }
        except Exception as e:
            status["agent_details_error"] = str(e)
    
    return status


if __name__ == "__main__":
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT, log_level="info")
