from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
from datetime import datetime


from enhanced_redis_chat_manager import EnhancedChatManager
from fixed_trip_planner import run_dynamic_trip_planner


chat_manager = EnhancedChatManager(redis_url="redis://localhost:6379/0")


app = FastAPI(
    title="Fixed Trip Planner API",
    description="AI-powered trip planning with working tools and real booking links",
    version="4.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: Optional[str] = None
    tool_call_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class ChatRequest(BaseModel):
    message: str
    conversation_history: Optional[List[ChatMessage]] = []
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    success: bool
    response: str
    trip_plan: Optional[Dict[str, Any]] = None
    conversation_history: List[ChatMessage]
    session_id: str
    tool_calls_made: Optional[List[str]] = []
    context_analysis: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint with fixed trip planner"""
    try:
   
        session_id = request.session_id
        if not session_id:
            session_id = chat_manager.create_session("user")
            print(f" Created new session: {session_id}")
        

        chat_manager.add_message(
            session_id, 
            "user", 
            request.message,
            metadata={"timestamp": datetime.now().isoformat()}
        )
        

        redis_history = chat_manager.get_session_messages(session_id, include_metadata=True)
        

        formatted_history = []
        for msg in redis_history[:-1]:  
            formatted_msg = {
                "role": msg["role"],
                "content": msg["content"]
            }
            if msg.get("tool_call_id"):
                formatted_msg["tool_call_id"] = msg["tool_call_id"]
            formatted_history.append(formatted_msg)
        
        print(f" Session {session_id}: {len(formatted_history)} previous messages")
        result = await run_dynamic_trip_planner(request.message, formatted_history)
      
        assistant_response = result["assistant_responses_for_chat"][0]["content"]

        chat_manager.add_message(
            session_id,
            "assistant",
            assistant_response,
            metadata={
                "tools_used": result.get("tool_calls_made", []),
                "timestamp": datetime.now().isoformat()
            }
        )
        
        if result.get("context_analysis"):
            context = result["context_analysis"]
            context_update = {}
            
            if context.get("destination"):
                context_update["destination"] = context["destination"]
            if context.get("origin"):
                context_update["origin"] = context["origin"]
            if context.get("dates"):
                context_update["travel_dates"] = context["dates"]
            if context.get("travelers"):
                context_update["travelers"] = context["travelers"]
            if context.get("interests"):
                context_update["interests"] = context["interests"]
            if context.get("budget"):
                context_update["budget"] = context["budget"]
            
            if context_update:
                chat_manager.update_session_context(session_id, context_update)
                print(f" Updated session context: {context_update}")
        

        final_history = chat_manager.get_session_messages(session_id)
        

        response_history = []
        for msg in final_history:
            if msg["role"] in ["user", "assistant"]:
                chat_msg = ChatMessage(
                    role=msg["role"],
                    content=msg["content"],
                    timestamp=msg.get("timestamp")
                )
                response_history.append(chat_msg)
        
        return ChatResponse(
            success=True,
            response=assistant_response,
            conversation_history=response_history,
            session_id=session_id,
            tool_calls_made=result.get("tool_calls_made", []),
            context_analysis=result.get("context_analysis", {}),
            error=None
        )
        
    except Exception as e:
        print(f" Error in chat endpoint: {e}")
        import traceback
        traceback.print_exc()
        
        if 'session_id' in locals() and session_id:
            error_msg = "I apologize, but I encountered an error. Please try rephrasing your request."
            chat_manager.add_message(session_id, "assistant", error_msg)
        
        return ChatResponse(
            success=False,
            response=error_msg if 'error_msg' in locals() else "An error occurred",
            conversation_history=request.conversation_history or [],
            session_id=session_id if 'session_id' in locals() else "error",
            tool_calls_made=[],
            context_analysis={},
            error=str(e)
        )


@app.get("/api/chat/history/{session_id}")
async def get_chat_history(session_id: str):
    """Get chat history for a session"""
    try:
        messages = chat_manager.get_session_messages(session_id)
        context = chat_manager.get_session_context(session_id)
        session_info = chat_manager.get_session_info(session_id)
        
        return {
            "success": True,
            "session_id": session_id,
            "messages": messages,
            "context": context,
            "session_info": session_info,
            "message_count": len(messages)
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Session not found: {str(e)}")

@app.post("/api/chat/new-session")
async def create_new_session(user_id: str = "anonymous"):
    try:
        session_id = chat_manager.create_session(user_id)
        
        welcome_msg = """Hello! I'm TripGenie, your AI travel assistant. I can help you:

        - Search for flights with real booking links
        - Find hotels with direct booking options
        - Create detailed itineraries
        - Answer travel questions

        Just tell me where you'd like to go and when, and I'll help you plan your perfect trip!"""
        
        chat_manager.add_message(session_id, "assistant", welcome_msg)
        
        return {
            "success": True,
            "session_id": session_id,
            "message": "New session created"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fixed_main:app", host="0.0.0.0", port=8000, reload=True)
