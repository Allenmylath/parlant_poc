import parlant.sdk as p  
from parlant.sdk import NLPServices  
  
@p.tool  
async def get_weather(context: p.ToolContext, city: str) -> p.ToolResult:  
    """Get weather information for a city."""  
    return p.ToolResult(f"Weather in {city}: Sunny, 72°F")  
  
async def main():  
    # Explicitly use OpenAI service  
    async with p.Server(nlp_service=NLPServices.openai) as server:  
        agent = await server.create_agent(  
            name="WeatherBot",  
            description="Helpful weather assistant"  
        )  
          
        await agent.create_guideline(  
            condition="User asks about weather",  
            action="Get current weather and provide tips",  
            tools=[get_weather]  
        )  
          
        print(f"Agent running at http://localhost:8800")  
        print(f"Agent ID: {agent.id}")  
  
if __name__ == "__main__":  
    import asyncio  
    asyncio.run(main())
