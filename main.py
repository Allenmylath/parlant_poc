import asyncio
import os
import time
from typing import List, Dict
import traceback

import parlant.sdk as p
from parlant.core.sessions import InMemorySessionStore
from fastapi import HTTPException, UploadFile, File, Request
from fastapi.responses import JSONResponse

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

# NLP Service configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NLP_SERVICE = p.NLPServices.openai

# Global Parlant server instance
parlant_server = None
police_agent = None


def get_memory_usage() -> float:
    """Get current memory usage in MB"""
    try:
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    except:
        return 0.0


async def configure_container(container: p.Container) -> p.Container:
    """Configure Parlant container - no auth, in-memory session store"""
    
    # Use in-memory session store (no MongoDB)
    container[p.SessionStore] = InMemorySessionStore()
    
    # No authentication - accept all requests
    container[p.AuthorizationPolicy] = p.PassThroughAuthorizationPolicy()
    
    return container


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
        description="AI assistant for Kerala Police website helping citizens with information about police services, procedures, and emergency contacts"
    )
    
    # Add guidelines for agent behavior
    await agent.add_guideline(
        condition="always",
        action="""You are a helpful AI assistant for the Kerala Police website in India.

Your responsibilities:
- Answer questions about police services, procedures, and information
- Use the RAG search tool to find relevant information from the police website
- Be concise, clear, and helpful
- Cite source URLs naturally in your responses
- Be professional yet friendly and approachable
- Focus on helping citizens access police services efficiently

Guidelines:
- Provide actionable information when possible
- If a question requires visiting a police station or calling, mention that clearly
- For emergency situations, always mention emergency helplines (100, 112)
- Be sensitive to potentially stressful situations citizens may be facing
- If you don't have relevant information from the website, say so politely

Always use the search_police_website tool to find relevant information before answering."""
    )
    
    # Add guideline for using translation
    await agent.add_guideline(
        condition="user message is not in English",
        action="Use the translate_to_english tool first to understand the user's message, then respond in English. The frontend will handle translation back to the user's language."
    )
    
    # Add guideline for citing sources
    await agent.add_guideline(
        condition="you find relevant information from search results",
        action="Cite the source URLs naturally in your response, for example: 'According to the [service page](url), ...' This helps users find more detailed information."
    )
    
    print(f"Agent '{agent.name}' created and configured")
    return agent


async def initialize_parlant() -> tuple:
    """Initialize Parlant server and agent"""
    
    # Create Parlant server with in-memory storage (no MongoDB)
    server = p.Server(
        host=SERVER_HOST,
        port=SERVER_PORT,
        nlp_service=NLP_SERVICE,
        configure_container=configure_container,
        # No MongoDB configuration - using in-memory store
    )
    
    await server.initialize()
    
    # Set up the agent
    agent = await setup_agent(server)
    
    # Register tools with the agent
    rag_tool = ParlantRAGTool()
    translation_tool = ParlantTranslationTool()
    
    await agent.add_tool(rag_tool)
    await agent.add_tool(translation_tool)
    
    print(f"Parlant server initialized on {SERVER_HOST}:{SERVER_PORT}")
    print(f"Agent tools registered: {len(await agent.list_tools())}")
    
    return server, agent


async def process_chat_stateless(
    message: str,
    history: List[Dict],
    max_sources: int = 5
) -> ChatResponse:
    """
    Process chat in stateless manner using Parlant
    
    Args:
        message: User's current message
        history: Conversation history from Streamlit
        max_sources: Maximum number of sources to return
        
    Returns:
        ChatResponse with answer and sources
    """
    start_time = time.time()
    
    global police_agent
    
    if not police_agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        # Create a temporary session for this request
        session_id = f"temp_{int(time.time() * 1000)}"
        customer_id = "streamlit_user"
        
        # Create session
        session = await police_agent.create_session(
            customer_id=customer_id,
            session_id=session_id
        )
        
        # Add conversation history to session context
        for msg in history[-10:]:  # Last 10 messages
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                # Don't actually send these as messages, just for context
                pass
            elif role == "assistant":
                pass
        
        # Send current message and get response
        response_event = await session.send_message(message)
        
        # Wait for completion
        await session.wait_for_completion()
        
        # Get the response
        events = await session.get_events()
        
        # Extract response and tool results
        assistant_message = ""
        sources = []
        detected_language = "en"
        
        for event in events:
            if hasattr(event, 'message') and event.message:
                if hasattr(event.message, 'content'):
                    assistant_message = event.message.content
            
            # Extract tool results for sources
            if hasattr(event, 'tool_result') and event.tool_result:
                result = event.tool_result
                if result.tool_name == "search_police_website":
                    # Extract sources from RAG results
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
                    # Extract detected language
                    try:
                        import json
                        data = json.loads(result.output) if isinstance(result.output, str) else result.output
                        detected_language = data.get("language", "en")
                    except:
                        pass
        
        # Delete temporary session
        await session.delete()
        
        processing_time = round((time.time() - start_time) * 1000, 2)
        
        return ChatResponse(
            response=assistant_message or "I'm sorry, I couldn't generate a response.",
            sources=sources,
            tokens_used=TokenUsage(input_tokens=0, output_tokens=0, total_tokens=0),
            detected_language=detected_language,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        print(f"Chat processing error: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")


# Initialize FastAPI app from Parlant server
async def startup():
    """Startup event handler"""
    global parlant_server, police_agent
    parlant_server, police_agent = await initialize_parlant()


async def shutdown():
    """Shutdown event handler"""
    global parlant_server
    if parlant_server:
        await parlant_server.close()


# Main function to run the server
async def main():
    """Initialize and run the Parlant server"""
    
    global parlant_server, police_agent
    
    # Initialize Parlant
    parlant_server, police_agent = await initialize_parlant()
    
    # Get the FastAPI app from Parlant server
    app = parlant_server.app
    
    # Add CORS middleware
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add custom endpoints
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
        """Health check endpoint"""
        try:
            services = {
                "openai": bool(os.getenv("OPENAI_API_KEY")),
                "sarvam": bool(os.getenv("SARVAM_API_KEY"))
            }
            
            # Test Qdrant connectivity
            try:
                rag_tool = get_rag_tool()
                services["qdrant"] = rag_tool.health_check()
            except Exception as e:
                print(f"Qdrant health check error: {e}")
                services["qdrant"] = False
            
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
        Main chat endpoint - stateless
        Receives history from Streamlit, processes with Parlant, returns response
        """
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        return await process_chat_stateless(
            message=request.message,
            history=[msg.dict() for msg in request.history],
            max_sources=request.max_sources
        )
    
    @app.post("/transcribe", response_model=TranscribeResponse)
    async def transcribe_audio(audio: UploadFile = File(...)):
        """Transcribe audio file to text"""
        try:
            if not audio.content_type.startswith('audio/'):
                raise HTTPException(status_code=400, detail="File must be an audio file")
            
            audio_data = await audio.read()
            
            if len(audio_data) == 0:
                raise HTTPException(status_code=400, detail="Audio file is empty")
            
            if len(audio_data) > 10 * 1024 * 1024:  # 10MB limit
                raise HTTPException(status_code=400, detail="Audio file too large (max 10MB)")
            
            transcription_tool = get_transcription_tool()
            transcript, detected_lang = transcription_tool.transcribe(audio_data)
            
            if not transcript:
                raise HTTPException(
                    status_code=400,
                    detail="Transcription failed - could not extract text from audio"
                )
            
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
        """Basic metrics endpoint for monitoring"""
        return {
            "uptime_seconds": time.time() - START_TIME,
            "memory_usage_mb": get_memory_usage(),
            "version": VERSION
        }
    
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
    
    # Start the server
    await parlant_server.serve()


if __name__ == "__main__":
    asyncio.run(main())
