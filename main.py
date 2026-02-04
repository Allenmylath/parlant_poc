import parlant.sdk as p  
  
@p.tool  
async def get_weather(context: p.ToolContext, city: str) -> p.ToolResult:  
    """Get weather information for a city."""  
    # Mock weather data - replace with real API call  
    return p.ToolResult(f"Weather in {city}: Sunny, 72°F")  
  
async def main():  
    async with p.Server() as server:  
        agent = await server.create_agent(  
            name="WeatherBot",  
            description="Helpful weather assistant"  
        )  
          
        await agent.create_guideline(  
            condition="User asks about weather",  
            action="Get current weather and provide tips",  
            tools=[get_weather]  
        )  
          

        print(f"Agent ID: {agent.id}")  
  
if __name__ == "__main__":  
    import asyncio  
    asyncio.run(main())
