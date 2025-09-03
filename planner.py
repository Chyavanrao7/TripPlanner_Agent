import os
import json
from typing import List, Dict, Any, Optional, Literal, TypedDict, Annotated
from datetime import datetime, timedelta
import asyncio
from pydantic import BaseModel, Field


from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# LangChain imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI 
from langchain_core.tools import tool
from dotenv import load_dotenv

# Import Firecrawl tools
from enhanced_tools_firecrawl import (
    firecrawl_hotel_search_tool,
    firecrawl_flight_search_tool,
    are_firecrawl_tools_available
)


load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.7,
)

class TripPlannerState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    current_task: str


class ItineraryGeneratorInput(BaseModel):
    destination: str = Field(description="Destination city/country")
    start_date: str = Field(description="Trip start date in YYYY-MM-DD format")
    end_date: str = Field(description="Trip end date in YYYY-MM-DD format")
    travelers: int = Field(description="Number of travelers")
    interests: Optional[List[str]] = Field(default_factory=list, description="Travel interests")
    budget: Optional[str] = Field(None, description="Budget range")


@tool("generate_itinerary", args_schema=ItineraryGeneratorInput)
def generate_itinerary(
    destination: str,
    start_date: str,
    end_date: str,
    travelers: int,
    interests: Optional[List[str]] = None,
    budget: Optional[str] = None
) -> str:
    """Generate a comprehensive itinerary"""
    try:
        print(f" Generating itinerary for {destination}")
        
        # Calculate duration
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        duration_days = (end - start).days + 1
        
        interests_str = ', '.join(interests) if interests else 'general sightseeing'
        
        # Destination-specific attractions
        attractions = {
            "paris": {
                "art": ["Louvre Museum", "Musée d'Orsay", "Centre Pompidou", "Rodin Museum"],
                "food": ["Le Marais Food Tour", "Cooking Class at La Cuisine Paris", "Wine Tasting in Montmartre", "Market Visit at Marché des Enfants Rouges"],
                "general": ["Eiffel Tower", "Arc de Triomphe", "Notre-Dame", "Sacré-Cœur"],
                "neighborhoods": ["Montmartre", "Le Marais", "Saint-Germain", "Latin Quarter"]
            },
            "rome": {
                "history": ["Colosseum & Roman Forum", "Vatican Museums", "Pantheon", "Borghese Gallery"],
                "food": ["Trastevere Food Tour", "Pasta Making Class", "Wine Tasting", "Campo de' Fiori Market"],
                "general": ["Trevi Fountain", "Spanish Steps", "Piazza Navona", "Villa Borghese"],
                "neighborhoods": ["Trastevere", "Centro Storico", "Testaccio", "Monti"]
            },
            "los angeles": {
                "entertainment": ["Hollywood Walk of Fame", "Universal Studios", "Getty Center", "Griffith Observatory"],
                "food": ["Santa Monica Pier Food Tour", "Korean BBQ in Koreatown", "Taco Tour in East LA", "Farmers Market"],
                "general": ["Santa Monica Beach", "Venice Beach", "Hollywood Sign", "Rodeo Drive"],
                "neighborhoods": ["Hollywood", "Beverly Hills", "Santa Monica", "Venice"]
            }
        }
        
        dest_key = destination.lower()
        dest_attractions = attractions.get(dest_key, attractions["paris"])
        
        # Build itinerary
        itinerary = f"# {duration_days}-Day {destination} Itinerary\n\n"
        itinerary += f"**Travelers:** {travelers} {'person' if travelers == 1 else 'people'}\n"
        itinerary += f"**Dates:** {start_date} to {end_date}\n"
        itinerary += f"**Interests:** {interests_str}\n"
        if budget:
            itinerary += f"**Budget:** {budget}\n"
        itinerary += "\n---\n\n"
        
        # Day-by-day plan
        for day in range(duration_days):
            current_date = start + timedelta(days=day)
            date_str = current_date.strftime("%A, %B %d, %Y")
            
            itinerary += f"## Day {day + 1} - {date_str}\n\n"
            
            if day == 0:  # Arrival day
                itinerary += "### Morning\n"
                itinerary += "- Arrival at airport\n"
                itinerary += f"- Transfer to hotel (consider pre-booking airport transfer)\n"
                itinerary += "- Hotel check-in and freshen up\n\n"
                
                itinerary += "### Afternoon\n"
                itinerary += f"- Light lunch at a café near your hotel\n"
                itinerary += f"- Gentle walking tour of {dest_attractions['neighborhoods'][0]}\n"
                itinerary += "- Get oriented with the city\n\n"
                
                itinerary += "### Evening\n"
                itinerary += "- Welcome dinner at a local restaurant\n"
                itinerary += "- Early rest to overcome jet lag\n\n"
                
            elif day == duration_days - 1:  # Departure day
                itinerary += "### Morning\n"
                itinerary += "- Hotel checkout (store luggage if late flight)\n"
                itinerary += "- Last-minute souvenir shopping\n"
                itinerary += "- Visit a local café for breakfast\n\n"
                
                itinerary += "### Afternoon\n"
                itinerary += "- Light lunch\n"
                itinerary += "- Departure to airport (arrive 3 hours before international flight)\n\n"
                
            else:  # Regular days
                itinerary += "### Morning (9:00 AM - 12:30 PM)\n"
                
                # Add interest-specific activities
                if interests and "art" in interests and "art" in dest_attractions:
                    itinerary += f"- Visit {dest_attractions['art'][day % len(dest_attractions['art'])]}\n"
                elif interests and "food" in interests and "food" in dest_attractions:
                    itinerary += f"- {dest_attractions['food'][day % len(dest_attractions['food'])]}\n"
                elif interests and "history" in interests and "history" in dest_attractions:
                    itinerary += f"- Explore {dest_attractions['history'][day % len(dest_attractions['history'])]}\n"
                elif interests and "entertainment" in interests and "entertainment" in dest_attractions:
                    itinerary += f"- Visit {dest_attractions['entertainment'][day % len(dest_attractions['entertainment'])]}\n"
                else:
                    itinerary += f"- Visit {dest_attractions['general'][day % len(dest_attractions['general'])]}\n"
                
                itinerary += "- Coffee break at a local café\n\n"
                
                itinerary += "### Afternoon (12:30 PM - 6:00 PM)\n"
                itinerary += "- Lunch at a recommended restaurant\n"
                itinerary += f"- Explore {dest_attractions['neighborhoods'][(day+1) % len(dest_attractions['neighborhoods'])]}\n"
                itinerary += "- Shopping or additional sightseeing\n"
                itinerary += "- Afternoon break at a local café\n\n"
                
                itinerary += "### Evening (6:00 PM - 10:00 PM)\n"
                itinerary += "- Aperitif at a wine bar\n"
                itinerary += "- Dinner at a local restaurant\n"
                itinerary += "- Evening stroll or cultural performance\n\n"
            
            itinerary += "**🚇 Transportation:** Metro day pass recommended (~€8-15)\n"
            itinerary += "**💰 Estimated Daily Cost:** €100-150 per person (meals, transport, attractions)\n\n"
            itinerary += "---\n\n"
        
        # Add practical information
        itinerary += "## 📍 Practical Information\n\n"
        itinerary += "### Getting Around\n"
        itinerary += f"- {destination} has excellent public transportation\n"
        itinerary += "- Consider buying a multi-day transport pass\n"
        itinerary += "- Download offline maps and transport apps\n\n"
        
        itinerary += "### Budget Breakdown (per person)\n"
        if budget:
            try:
                budget_amount = int(budget.replace("$", "").replace(",", "").replace("USD", "").strip())
                per_person_budget = budget_amount / travelers
                daily_budget = per_person_budget / duration_days
                
                itinerary += f"- Total Budget: ${budget_amount} ({travelers} people)\n"
                itinerary += f"- Per Person: ${per_person_budget:.0f}\n"
                itinerary += f"- Daily Budget: ${daily_budget:.0f}/person/day\n"
                itinerary += f"- Suggested allocation:\n"
                itinerary += f"  - Accommodation: 30-40% (${daily_budget * 0.35:.0f}/day)\n"
                itinerary += f"  - Food: 30-35% (${daily_budget * 0.32:.0f}/day)\n"
                itinerary += f"  - Activities: 20-25% (${daily_budget * 0.22:.0f}/day)\n"
                itinerary += f"  - Transport/Misc: 10-15% (${daily_budget * 0.11:.0f}/day)\n"
            except:
                itinerary += "- Budget allocation depends on your travel style\n"
        
        itinerary += "\n### Tips\n"
        itinerary += f"- Book {destination} museum tickets online in advance\n"
        itinerary += "- Many museums offer free entry on first Sunday of month\n"
        itinerary += "- Restaurant reservations recommended for dinner\n"
        itinerary += "- Keep copies of important documents\n"
        
        return itinerary
        
    except Exception as e:
        print(f" Itinerary generation error: {e}")
        return f"Error generating itinerary: {str(e)}"

# Main agent node - LLM handles context directly
async def agent_node(state: TripPlannerState) -> TripPlannerState:
    """Main LLM agent that handles context management directly"""
    
    # Check if Firecrawl tools are available
    firecrawl_status = " Available" if are_firecrawl_tools_available() else " Not Available (requires FIRECRAWL_API_KEY)"
    
    system_prompt = f"""You are TripGenie, an expert AI travel assistant with perfect memory of our conversation.

FIRECRAWL WEB SCRAPING STATUS: {firecrawl_status}

IMPORTANT BEHAVIORS:
1. **Memory**: Remember ALL details from our conversation. Never ask for information already provided.
2. **Context Management**: You maintain context internally. Extract and remember:
   - Destination and origin cities
   - Travel dates (start and end)
   - Number of travelers
   - Interests and preferences
   - Budget information
   - Any other trip details mentioned
3. **Smart Responses**: 
   - For general travel questions → Answer directly without tools
   - For flight searches → Use firecrawl_flight_search (real-time flight data)
   - For hotel searches → Use firecrawl_hotel_search (real web scraping)
   - For itinerary requests → Use generate_itinerary tool
4. **Progressive Building**: Build understanding across messages. Reference previous information naturally.

TOOL USAGE RULES:
- Flight/Hotel searches now use REAL web scraping via Firecrawl
- Only use tools when user explicitly asks for searches or planning
- Extract ALL parameters from conversation context when available
- If critical information is missing, ask for it before calling tools

REAL DATA NOTICE:
- Flight searches scrape live data from Kayak/other sources
- Hotel searches scrape live data from Booking.com/Hotels.com
- Results are real, current prices and availability
- Response times may be longer due to live web scraping

CONTEXT EXTRACTION EXAMPLES:
- "I want to go to Paris" → Remember destination: Paris
- "From New York" → Remember origin: New York
- "July 15-20" → Remember dates: 2024-07-15 to 2024-07-20
- "For 2 people" → Remember travelers: 2
- "I love art and food" → Remember interests: art, food
- "Budget is $3000" → Remember budget: $3000

Remember: You have access to the entire conversation history. Act like a human assistant who remembers everything discussed and builds context naturally."""

    # Prepare messages with system prompt
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    
    # Bind tools to LLM - use Firecrawl tools
    if are_firecrawl_tools_available():
        tools = [firecrawl_flight_search_tool, firecrawl_hotel_search_tool, generate_itinerary]
        print(" Using Firecrawl tools for real web scraping")
    else:
        # Fallback to just itinerary generation if Firecrawl not available
        tools = [generate_itinerary]
        print("⚠ Firecrawl not available, limited to itinerary generation only")
    
    llm_with_tools = llm.bind_tools(tools)
    
    # Get LLM response
    response = await llm_with_tools.ainvoke(messages)
    
    # Add response to state
    state["messages"].append(response)
    
    # Determine current task
    if response.tool_calls:
        state["current_task"] = "execute_tools"
    else:
        state["current_task"] = "respond"
    
    return state


async def tool_node(state: TripPlannerState) -> TripPlannerState:
    """Execute tools and return results to main agent"""
    
    last_message = state["messages"][-1]
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})
            tool_id = tool_call.get("id")
            
            print(f" Executing {tool_name} with args: {tool_args}")
            
            # Execute the appropriate tool
            if tool_name == "firecrawl_flight_search":
                print("✈ Scraping live flight data from Firecrawl...")
                tool_args = dict(tool_args)
                tool_args['formatted_output'] = True
                result = await firecrawl_flight_search_tool.ainvoke(tool_args)
            elif tool_name == "firecrawl_hotel_search":
                print(" Scraping live hotel data...")
                result = await firecrawl_hotel_search_tool.ainvoke(tool_args)
            elif tool_name == "generate_itinerary":
                print(" Generating custom itinerary...")
                result = generate_itinerary.invoke(tool_args)
            else:
                result = json.dumps({"error": f"Unknown tool: {tool_name}"})
            
            # Add tool result
            tool_message = ToolMessage(
                content=result,
                tool_call_id=tool_id
            )
            state["messages"].append(tool_message)
    
    # Return to agent for final response
    state["current_task"] = "finalize_response"
    return state

# Response finalization node
async def response_node(state: TripPlannerState) -> TripPlannerState:
    """Finalize and format the response with real booking data"""
    
    # Have the agent create a final formatted response
    if state["current_task"] == "finalize_response":
        # Check for tool results
        tool_results = []
        for i, msg in enumerate(state["messages"]):
            if isinstance(msg, ToolMessage):
                tool_results.append((i, msg))
        
        if tool_results:
            # Create formatting prompt based on tool results
            formatting_prompt = """Based on the REAL scraped data above, provide a well-formatted response.

FORMATTING RULES:

For REAL FLIGHT RESULTS (scraped from Skyscanner):
- Present flights in a clean, organized format
- Include all available details (airline, flight number, times, duration, price)
- Show multiple booking options (Skyscanner, MakeMyTrip, Cleartrip, GoIbibo)
- Format booking links as clickable URLs
- Highlight best value flights or shortest flights
- If no flights found, explain what happened and suggest alternatives

For REAL HOTEL RESULTS (scraped from Booking.com/Hotels.com):
- Create organized cards for each hotel
- Include name, location, rating, price, amenities
- Highlight best value options or highest-rated hotels
- If no hotels found, explain and suggest alternatives

For ITINERARIES:
- Present the complete itinerary as provided
- Highlight key activities for each day
- Include budget breakdown if available

IMPORTANT NOTES:
- This data is scraped live from travel websites
- Prices and availability are current as of the search
- Results may vary based on website content and availability
- Suggest users verify details directly on booking sites

Make the response conversational and helpful. Acknowledge that real-time data was used."""

            messages = state["messages"] + [SystemMessage(content=formatting_prompt)]
            final_response = await llm.ainvoke(messages)
            state["messages"].append(final_response)
    
    return state

# Routing function
def route_next(state: TripPlannerState) -> Literal["tools", "response", "__end__"]:
    """Determine next node based on current task"""
    
    if state["current_task"] == "execute_tools":
        return "tools"
    elif state["current_task"] == "finalize_response":
        return "response"
    else:
        return "__end__"

# Build the graph
def create_trip_planner_graph():
    """Create the trip planner graph with proper flow"""
    
    # Initialize with memory
    memory = MemorySaver()
    
    graph = StateGraph(TripPlannerState)
    
    # Add nodes
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("response", response_node)
    
    # Add edges
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        route_next,
        {
            "tools": "tools",
            "response": "response",
            "__end__": END
        }
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("response", END)
    
    return graph.compile(checkpointer=memory)


class LLMContextTripPlanner:
    """Trip planner where LLM handles context management directly"""
    
    def __init__(self):
        self.graph = create_trip_planner_graph()
        self.config = {"configurable": {"thread_id": "main"}}
        
        if are_firecrawl_tools_available():
            print(" LLM Context Trip Planner initialized - REAL web scraping enabled")
        else:
            print("⚠ LLM Context Trip Planner initialized - Firecrawl not available, limited functionality")
            print("   Add FIRECRAWL_API_KEY to environment for full web scraping features")
    
    async def process_message(self, user_input: str, session_id: str = "main") -> Dict[str, Any]:
        """Process user message with LLM-managed context"""
        
        print(f"�� Processing: '{user_input}'")
        
        # Set config for this session
        config = {"configurable": {"thread_id": session_id}}
        
        # Get current state
        current_state = self.graph.get_state(config)
        
        # Initialize state if new session
        if not current_state.values:
            initial_state = {
                "messages": [],
                "current_task": ""
            }
        else:
            initial_state = current_state.values
        
        # Add user message
        initial_state["messages"].append(HumanMessage(content=user_input))
        
        try:
            # Run the graph
            result = await self.graph.ainvoke(initial_state, config)
            
            # Get the final response
            final_messages = result["messages"]
            last_ai_message = None
            
            for msg in reversed(final_messages):
                if isinstance(msg, AIMessage) and not isinstance(msg, ToolMessage):
                    last_ai_message = msg
                    break
            
            response_content = last_ai_message.content if last_ai_message else "I'm here to help with your trip planning!"
            
            # Extract tools used
            tools_used = []
            for msg in final_messages:
                if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls'):
                    for tool_call in msg.tool_calls:
                        tools_used.append(tool_call.get("name"))
            
            return {
                "success": True,
                "response": response_content,
                "tools_used": tools_used,
                "session_id": session_id,
                "firecrawl_enabled": are_firecrawl_tools_available()
            }
            
        except Exception as e:
            print(f" Error: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "success": False,
                "response": f"I apologize, I encountered an error: {str(e)}",
                "tools_used": [],
                "session_id": session_id,
                "firecrawl_enabled": are_firecrawl_tools_available()
            }


async def run_llm_context_trip_planner(user_input: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
    """Main entry point for the LLM context trip planner"""
    planner = LLMContextTripPlanner()

    if conversation_history:
        session_id = f"session_{hash(str(conversation_history))}"
        # Initialize with history
        initial_messages = []
        for msg in conversation_history:
            if msg["role"] == "user":
                initial_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                initial_messages.append(AIMessage(content=msg["content"]))
        
        # Set initial state with history
        config = {"configurable": {"thread_id": session_id}}
        initial_state = {
            "messages": initial_messages,
            "current_task": ""
        }
        planner.graph.update_state(config, initial_state)
    else:
        session_id = f"session_{datetime.now().timestamp()}"
    
    result = await planner.process_message(user_input, session_id)
    

    return {
        "success": result["success"],
        "assistant_responses_for_chat": [{
            "role": "assistant",
            "content": result["response"]
        }],
        "structured_trip_plan": None,
        "updated_conversation_history": conversation_history or [],
        "tool_calls_made": result.get("tools_used", []),
        "error": None if result["success"] else result.get("response"),
        "firecrawl_enabled": result.get("firecrawl_enabled", False)
    }



if __name__ == "__main__":
    print(" LLM Context Trip Planner - LLM Handles Context Directly")
    print("=" * 60)
