from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class Message(BaseModel):
    """Single message in conversation"""
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")

class ChatRequest(BaseModel):
    """Request for chat endpoint"""
    message: str = Field(..., description="User's current message", min_length=1)
    history: List[Message] = Field(
        default_factory=list, 
        description="Conversation history from Streamlit (last 10 messages recommended)"
    )
    language: Optional[str] = Field(default="en", description="Detected language code")
    max_sources: Optional[int] = Field(default=5, ge=1, le=10, description="Number of RAG sources to retrieve")

class Source(BaseModel):
    """Single source from RAG search"""
    content: str = Field(..., description="Content snippet from source")
    url: str = Field(..., description="Source URL")
    score: float = Field(..., description="Relevance score (0-1)")

class TokenUsage(BaseModel):
    """Token usage information"""
    input_tokens: int
    output_tokens: int
    total_tokens: int

class ChatResponse(BaseModel):
    """Response from chat endpoint"""
    response: str = Field(..., description="Assistant's response")
    sources: List[Source] = Field(..., description="Sources used to generate response")
    tokens_used: TokenUsage = Field(..., description="Token usage information")
    detected_language: Optional[str] = Field(None, description="Detected input language")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")

class TranscribeRequest(BaseModel):
    """Request for transcription from base64 audio"""
    audio_base64: str = Field(..., description="Base64 encoded audio file")

class TranscribeResponse(BaseModel):
    """Response from transcription endpoint"""
    text: str = Field(..., description="Transcribed text")
    detected_language: Optional[str] = Field(None, description="Detected language code")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status: 'healthy', 'degraded', or 'unhealthy'")
    version: str = Field(..., description="API version")
    services: Dict[str, bool] = Field(..., description="Status of dependent services")
    uptime_seconds: Optional[float] = Field(None, description="Service uptime in seconds")
    memory_usage_mb: Optional[float] = Field(None, description="Current memory usage in MB")

class ErrorResponse(BaseModel):
    """Error response"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")
