from typing import List, Dict, Tuple
from openai import OpenAI
from qdrant_client import QdrantClient
import os
from functools import lru_cache
import httpx
import json
import time
import parlant.sdk as p


class RAGTool:
    """Tool for searching police website content using RAG"""
    
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            timeout=10.0  # 10 second timeout
        )
        self.collection_name = os.getenv("QDRANT_COLLECTION", "website_content")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using OpenAI"""
        try:
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=[text]
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Embedding generation error: {e}")
            raise
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for relevant content in Qdrant
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of dictionaries with content, url, and score
        """
        try:
            # Generate query embedding
            query_embedding = self.get_embedding(query)
            
            # Search Qdrant
            results = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=top_k,
                with_payload=True
            )
            
            # Format results
            formatted_results = []
            for hit in results.points:
                formatted_results.append({
                    "content": hit.payload.get("content", ""),
                    "url": hit.payload.get("url", ""),
                    "score": float(hit.score)
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"RAG search error: {e}")
            return []
    
    def health_check(self) -> bool:
        """Check if Qdrant is accessible"""
        try:
            self.qdrant_client.get_collections()
            return True
        except Exception as e:
            print(f"Qdrant health check failed: {e}")
            return False


class TranslationTool:
    """Tool for translating text to English and detecting language"""
    
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("TRANSLATION_MODEL", "gpt-4o-mini")
    
    def translate_to_english(self, text: str) -> Tuple[str, str]:
        """
        Translate text to English and detect source language
        
        Args:
            text: Input text in any language
            
        Returns:
            Tuple of (translated_text, detected_language_code)
        """
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are a language detection and translation expert.

Task:
1. Detect the language of the input text
2. If not in English, translate it to English
3. If already in English, return it as-is

Respond ONLY with valid JSON in this exact format:
{
    "translated": "the english text here",
    "language": "language_code"
}

Language codes: en, ml, hi, ta, te, kn, bn, gu, mr, pa, ur, etc.
Use ISO 639-1 codes where possible."""
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                temperature=0.3,
                max_tokens=500,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            translated = result.get("translated", text)
            language = result.get("language", "en")
            
            return translated, language
            
        except Exception as e:
            print(f"Translation error: {e}")
            # Return original text if translation fails
            return text, "unknown"


class TranscriptionTool:
    """Tool for transcribing audio using Sarvam AI"""
    
    def __init__(self):
        self.api_key = os.getenv("SARVAM_API_KEY")
        self.api_url = "https://api.sarvam.ai/speech-to-text-translate"
        self.timeout = 30.0
    
    def transcribe(self, audio_data: bytes) -> Tuple[str, str]:
        """
        Transcribe audio to text using Sarvam AI
        
        Args:
            audio_data: Raw audio file bytes
            
        Returns:
            Tuple of (transcribed_text, detected_language)
        """
        try:
            files = {
                'file': ('audio.wav', audio_data, 'audio/wav')
            }
            
            data = {
                'model': 'saarika:v2.5',
                'language_code': 'unknown'  # Auto-detect language
            }
            
            headers = {
                'api-subscription-key': self.api_key
            }
            
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    self.api_url,
                    files=files,
                    data=data,
                    headers=headers
                )
            
            if response.status_code == 200:
                result = response.json()
                transcript = result.get('transcript', '')
                language = result.get('language_code', 'unknown')
                return transcript, language
            else:
                print(f"Sarvam API error: {response.status_code} - {response.text}")
                return "", "unknown"
                
        except httpx.TimeoutException:
            print("Transcription timeout")
            return "", "unknown"
        except Exception as e:
            print(f"Transcription error: {e}")
            return "", "unknown"


# ============================================================================
# PARLANT TOOL WRAPPERS
# ============================================================================

class ParlantRAGTool(p.Tool):
    """Parlant Tool wrapper for RAG search"""
    
    def __init__(self):
        super().__init__(
            name="search_police_website",
            description="Search the Kerala Police website for relevant information about police services, procedures, emergency contacts, and other official information. Use this tool to find accurate answers from the official website."
        )
        self.rag = get_rag_tool()
    
    async def call(self, query: str, top_k: int = 5) -> str:
        """
        Search the police website
        
        Args:
            query: Search query in English
            top_k: Number of results to return (default: 5)
            
        Returns:
            JSON string with search results
        """
        try:
            results = self.rag.search(query, top_k=top_k)
            
            if not results:
                return json.dumps({
                    "status": "no_results",
                    "message": "No relevant information found in the police website database."
                })
            
            # Format results for the agent
            formatted = []
            for r in results:
                formatted.append({
                    "content": r["content"],
                    "url": r["url"],
                    "score": r["score"]
                })
            
            return json.dumps(formatted, ensure_ascii=False)
            
        except Exception as e:
            print(f"Parlant RAG tool error: {e}")
            return json.dumps({
                "status": "error",
                "message": f"Search failed: {str(e)}"
            })
    
    def get_parameters(self) -> Dict:
        """Define tool parameters for Parlant"""
        return {
            "query": {
                "type": "string",
                "description": "The search query in English",
                "required": True
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results to return (1-10)",
                "required": False,
                "default": 5
            }
        }


class ParlantTranslationTool(p.Tool):
    """Parlant Tool wrapper for translation"""
    
    def __init__(self):
        super().__init__(
            name="translate_to_english",
            description="Translate user's message to English and detect the source language. Use this when the user's message appears to be in a language other than English (like Malayalam, Hindi, Tamil, etc.)"
        )
        self.translator = get_translation_tool()
    
    async def call(self, text: str) -> str:
        """
        Translate text to English
        
        Args:
            text: Text in any language
            
        Returns:
            JSON string with translation and detected language
        """
        try:
            translated, language = self.translator.translate_to_english(text)
            
            return json.dumps({
                "translated": translated,
                "language": language,
                "original": text
            }, ensure_ascii=False)
            
        except Exception as e:
            print(f"Parlant translation tool error: {e}")
            return json.dumps({
                "translated": text,
                "language": "unknown",
                "error": str(e)
            })
    
    def get_parameters(self) -> Dict:
        """Define tool parameters for Parlant"""
        return {
            "text": {
                "type": "string",
                "description": "Text to translate to English",
                "required": True
            }
        }


# Singleton instances (cached)
@lru_cache()
def get_rag_tool() -> RAGTool:
    """Get cached RAG tool instance"""
    return RAGTool()


@lru_cache()
def get_translation_tool() -> TranslationTool:
    """Get cached translation tool instance"""
    return TranslationTool()


@lru_cache()
def get_transcription_tool() -> TranscriptionTool:
    """Get cached transcription tool instance"""
    return TranscriptionTool()
