import parlant.sdk as p
from parlant.sdk import NLPServices
from qdrant_client import QdrantClient
from openai import OpenAI
from typing import List, Dict
import os

# Configuration from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = "website_content"
EMBEDDING_MODEL = "text-embedding-3-small"

# Initialize clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

def get_embedding(text: str) -> List[float]:
    """Generate embedding for text"""
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[text]
    )
    return response.data[0].embedding

def search_qdrant(query: str, top_k: int = 5) -> List[Dict]:
    """Search Qdrant for relevant content"""
    query_embedding = get_embedding(query)
    
    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=top_k,
        with_payload=True
    )
    
    formatted_results = []
    for hit in results.points:
        formatted_results.append({
            "content": hit.payload["content"],
            "url": hit.payload["url"],
            "score": hit.score
        })
    
    return formatted_results

@p.tool
async def search_police_website(context: p.ToolContext, query: str) -> p.ToolResult:
    """Search the police website for relevant information.
    
    Args:
        query: The user's question or search query
    
    Returns:
        Relevant content from the police website with sources
    """
    try:
        # Search for relevant content
        search_results = search_qdrant(query, top_k=5)
        
        if not search_results:
            return p.ToolResult("No relevant information found on the police website.")
        
        # Format results for the agent
        formatted_context = "Here's what I found on the police website:\n\n"
        
        for idx, result in enumerate(search_results, 1):
            formatted_context += f"Source {idx} (Relevance: {result['score']:.3f}):\n"
            formatted_context += f"URL: {result['url']}\n"
            formatted_context += f"Content: {result['content']}\n\n"
        
        return p.ToolResult(formatted_context)
    
    except Exception as e:
        return p.ToolResult(f"Error searching police website: {str(e)}")

async def main():
    # Use OpenAI service
    async with p.Server(nlp_service=NLPServices.openai) as server:
        agent = await server.create_agent(
            name="PoliceAssistant",
            description="Helpful assistant for Kerala Police website queries in English and Malayalam"
        )
        
        # Add glossary term for PCC
        await agent.create_term(
            name="Police Clearance Certificate",
            description="Official document issued by police certifying that a person has no criminal record or pending cases",
            synonyms=["PCC", "police clearance", "clearance certificate", "character certificate", "no objection certificate"]
        )
        
        # HIGH CRITICALITY: Reject off-topic questions
        await agent.create_guideline(
            condition="User asks about celebrities, political personalities, sports, entertainment, general knowledge, or any topic unrelated to police services",
            action="Politely inform the user: 'I am a Police Assistant Bot designed to help with information from the Kerala Police website only. I can answer questions about police services, procedures, certificates, complaints, and other police-related matters. Please ask me about police services.'",
            criticality=p.Criticality.HIGH
        )
        
        # Guideline 1: Search police website for questions
        await agent.create_guideline(
            condition="User asks any question about police services, procedures, or information",
            action="Search the police website using the search_police_website tool and provide a helpful answer based on the retrieved information. Always cite the source URLs in your response.",
            tools=[search_police_website]
        )
        
        # Guideline 2: Reply in Malayalam if user asks in Malayalam
        await agent.create_guideline(
            condition="User asks a question in Malayalam language",
            action="Respond to the user in Malayalam. Use the search_police_website tool to get information, then translate your entire response to Malayalam while keeping the source URLs intact.",
            tools=[search_police_website]
        )
        
        print(f"🚔 Police Assistant Agent running at http://localhost:8800")
        print(f"Agent ID: {agent.id}")
        print("\nThe agent will:")
        print("- Automatically search the police website to answer your questions")
        print("- Reply in Malayalam if you ask in Malayalam")
        print("- Reply in English if you ask in English")
        print("- Understand PCC and related terms")
        print("- Politely decline questions about celebrities, politics, and off-topic queries")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
