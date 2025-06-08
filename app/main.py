from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import os
from datetime import datetime
import json

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from typing_extensions import TypedDict

# Groq imports
from groq import Groq

from dotenv import load_dotenv

# Get environmental variabls
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Multi-Tool Restaurant Chatbot", version="1.0.0")

# Initialize Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Mini Knowledge Base - General World Knowledge
KNOWLEDGE_BASE = {
    "capital_australia": "The capital of Australia is Canberra",
    "largest_ocean": "The Pacific Ocean is the largest ocean in the world, covering about one-third of Earth's surface.",
    "speed_of_light": "The speed of light in a vacuum is approximately 299,792,458 meters per second (about 300,000 km/s).",
    "tallest_mountain": "Mount Everest is the tallest mountain in the world, standing at 8,848.86 meters (29,031.7 feet) above sea level.",
    "human_bones": "An adult human body has 206 bones, while babies are born with about 270 bones that fuse together as they grow.",
    "largest_planet": "Jupiter is the largest planet in our solar system, with a mass greater than all other planets combined.",
    "water_formula": "The chemical formula for water is H2O, meaning it consists of two hydrogen atoms and one oxygen atom.",
    "longest_river": "The Nile River is traditionally considered the longest river in the world at approximately 6,650 kilometers (4,130 miles).",
    "fastest_animal": "The peregrine falcon is the fastest animal, capable of reaching speeds over 240 mph (386 km/h) when diving.",
    "smallest_country": "Vatican City is the smallest country in the world, with an area of just 0.17 square miles (0.44 square kilometers).",
    "deepest_ocean": "The Mariana Trench in the Pacific Ocean is the deepest part of Earth's oceans, reaching depths of about 36,200 feet (11,000 meters).",
    "photosynthesis": "Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen.",
    "gravity_earth": "Earth's gravity accelerates objects at approximately 9.8 meters per second squared (9.8 m/s²) at sea level.",
    "dna_structure": "DNA has a double helix structure, discovered by Watson and Crick, consisting of two complementary strands of nucleotides.",
    "boiling_water": "Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at standard atmospheric pressure at sea level.",
    "moon_landing": "The first successful manned moon landing was by NASA's Apollo 11 mission in 1969, with Neil Armstrong and Buzz Aldrin.",
    "human_body_water": "About 60 percent of the adult human body is composed of water, which is essential for all bodily functions.",
    "earth_orbit": "Earth takes approximately 365.25 days to complete one orbit around the Sun, which is why we have a leap year every 4 years.",
    "invention_internet": "The modern internet evolved from ARPANET, developed in the late 1960s and early 1970s in the United States.",
    "volcano_active": "Mount Etna in Italy is one of the most active volcanoes in the world and has frequent eruptions."
}


# State definition for LangGraph
class ChatbotState(TypedDict):
    messages: list
    user_message: str
    intent: str  # 'reservation', 'knowledge', 'gibberish', 'contradiction', 'profanity'
    date: Optional[str]
    time: Optional[str]
    persons: Optional[int]
    reservation_complete: bool
    conversation_history: list

# Pydantic models for API
class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    response: str
    intent: str
    reservation_status: Dict[str, Any]

# In-memory storage for sessions
sessions: Dict[str, ChatbotState] = {}

def get_llm_response(prompt: str, model: str = "llama3-70b-8192") -> str:
    """Get response from Groq LLM"""
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0,
            max_tokens=500
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Error getting LLM response: {str(e)}"

def analyze_message(state: ChatbotState) -> ChatbotState:
    """Analyze user message and route to appropriate tool"""
    user_msg = state["user_message"]
    
    # Get available knowledge base topics
    kb_topics = ", ".join(KNOWLEDGE_BASE.keys())
    
    analysis_prompt = f"""
    Analyze the following user message and classify it into ONE of these categories:

    User message: "{user_msg}"

    Categories:
    1. "reservation" - Messages about booking/reserving tables, providing date/time/persons, availability questions, confirming reservations
    
    2. "knowledge" - General world knowledge questions about science, geography, history, facts, trivia, or educational topics (NOT related to restaurants)
    
    3. "gibberish" - Messages that are nonsensical, random characters, unclear mumbling, or completely incomprehensible text
    
    4. "contradiction" - Messages containing obviously false statements, contradictory claims, or statements that go against common knowledge/facts
    
    5. "profanity" - Messages containing rude language, insults, offensive content, or disrespectful behavior
    
    
    Any greeting message like Hello, Hey, Hi then consider "reservation".

    Examples:
    - "Book table for 4 tomorrow" → reservation
    - "What's the capital of France?" → knowledge  
    - "How fast is light?" → knowledge
    - "asdflkj qwerty zxcvbn" → gibberish
    - "Is ice hot?" → contradiction
    - "You're stupid" → profanity
    

    Respond with ONLY one word: reservation, knowledge, gibberish, contradiction, or profanity
    """
    
    intent = get_llm_response(analysis_prompt).lower().strip()
    valid_intents = ["reservation", "knowledge", "gibberish", "contradiction", "profanity"]
    
    if intent not in valid_intents:
        intent = "reservation"  # Default fallback
    
    state["intent"] = intent
    return state

def handle_reservation(state: ChatbotState) -> ChatbotState:
    """Handle reservation-related conversation with persistent context"""
    user_msg = state["user_message"]
    current_date = state.get("date")
    current_time = state.get("time")
    current_persons = state.get("persons")
    
    # Build conversation context from history
    recent_messages = state.get("conversation_history", [])  # Last 6 messages for context
    conversation_context = ""
    if recent_messages:
        conversation_context = "Recent conversation:\n" + "\n".join([
            f"{'User' if i % 2 == 0 else 'Assistant'}: {msg}" 
            for i, msg in enumerate(recent_messages)
        ]) + "\n\n"
    
    reservation_context = f"""
    Current reservation status:
    - Date: {current_date if current_date else 'Not provided'}
    - Time: {current_time if current_time else 'Not provided'}
    - Number of persons: {current_persons if current_persons else 'Not provided'}
    """
    
    reservation_prompt = f"""
    You are a helpful restaurant reservation assistant. Your job is to collect three required pieces of information:
    1. Date (when they want to dine)
    2. Time (what time they prefer)  
    3. Number of persons (how many people)

    {conversation_context}
    {reservation_context}
    
    -if user has provided all three information, then ask to confirm these details.
    -if user confirms all the details, then return only YES in response. otherwise continue the conversation.
    
    User message: "{user_msg}"
    
    Instructions:
    
    Parse “fuzzy” time expressions (e.g., “book a table this weekend or maybe
    Monday morning”) into a structured “BookReservation” intent with candidate slots
    (date_candidates = [“this Saturday”, “this Sunday”, “Monday morning”]).
    
    -If multiple candidates remain unresolved, the chatbot must proactively ask one
    clarifying question (“Do you prefer Saturday, Sunday, or Monday morning?”)
    rather than assuming or failing silently
    
    - Use the conversation history to maintain context
    - If the user message contains any missing information, extract and acknowledge it
    - Ask for the next missing piece of information naturally
    - If you have all three pieces, confirm the reservation
    - Be conversational and remember what was discussed before
    - Don't ask for information you already have

    -if user has provided all three details and confirms all the details, then return only YES in response. otherwise continue the conversation.
    Respond naturally and ask for the next needed information if incomplete.
    """
    
    response = get_llm_response(reservation_prompt)
    # print("AI MESSAGE:",response)
    if state.get("date") and state.get("time") and state.get("persons") and response.strip() == "YES":
        state["reservation_complete"] = True
        response = "Thanks for confirmation, The table will be reserved for you. See you soon."
        state["messages"].append(AIMessage(content=response))
        return state
    
    # Extract information from user message
    extraction_prompt = f"""
    You are an information extractor.

    Your task is to extract reservation details from this user message: "{user_msg}"

    Follow these rules strictly:
    1. If the message contains more than one date or time, set both 'date' and 'time' to null.
    2. Extract 'date' and 'time' only if they are mentioned once and clearly.
    3. If any of the fields (date, time, persons) are not found, set them to null.
    4. Return ONLY a raw JSON object, without any explanation, markdown, or labels.
    5. Do not include keys with invalid values like empty strings; use `null`.

    Expected JSON format:
    {{
    "date": "date or day of the week, or null",
    "time": "time in readable format or null",
    "persons": "number of persons as string or null"
    }}

    Examples:
    Input: Book on this Monday or Tuesday  
    Output: {{"date": None, "time": None, "persons": None}}

    Input: Book on this Monday  
    Output: {{"date": "Monday", "time": None, "persons": None}}

    Input: Book for 12pm, 2 people  
    Output: {{"date": None, "time": "12 pm", "persons": "2"}}

    Now extract from this:
    "{user_msg}"
    """
    

    try:
        extraction_result = get_llm_response(extraction_prompt)
        # print("user:",user_msg,"extraction_result:",extraction_result)
        
        import re
        json_match = re.search(r'\{.*\}', extraction_result, re.DOTALL)
        if json_match:
            extracted_info = json.loads(json_match.group())
            
            if extracted_info.get("date"):
                state["date"] = extracted_info["date"]
            if extracted_info.get("time"):
                state["time"] = extracted_info["time"]
            if extracted_info.get("persons"):
                state["persons"] = extracted_info["persons"]
    except:
        pass
    
    # Check completion
    if state.get("date") and state.get("time") and state.get("persons"):
        
        
        confirmation_prompt = f"""
        Reservation details:
        Date: {state['date']},
        Time: {state['time']},
        Persons: {state['persons']}
        
        Please respond with "yes please confirm" or let me know for any change.
        """

        response = confirmation_prompt
    
    state["messages"].append(AIMessage(content=response))
    return state

def handle_knowledge(state: ChatbotState) -> ChatbotState:
    """Handle general world knowledge queries (off-topic from reservations)"""
    user_msg = state["user_message"]
    
    # Convert knowledge base to string for LLM
    kb_content = "\n".join([f"- {key.replace('_', ' ').title()}: {value}" for key, value in KNOWLEDGE_BASE.items()])
    
    knowledge_prompt = f"""
    You are a helpful AI assistant with access to general world knowledge. Use the following knowledge base to answer the user's question:

    KNOWLEDGE BASE:
    {kb_content}

    User question: "{user_msg}"

    Instructions:
    - Use the information from the knowledge base above to answer questions
    - If the question isn't covered in the knowledge base, politely say I dont have information on that.
    - After answering the knowledge question, politely remind the user to redirect them back to table reservations without extra question added.

    Provide a helpful short response and then redirect to reservations.
    """
    
    response = get_llm_response(knowledge_prompt)
    state["messages"].append(AIMessage(content=response))
    return state

def handle_gibberish(state: ChatbotState) -> ChatbotState:
    """Handle gibberish/unclear messages"""
    # user_msg = state["user_message"]
    
    # gibberish_prompt = f"""
    # The user sent a message that appears to be gibberish or unclear: "{user_msg}"
    
    # Respond politely asking them to rephrase their message. Be helpful and mention that you can assist with:
    # - Table reservations
    # - Restaurant information
    # - General questions
    
    # Keep it friendly and brief.
    # """
    
    # response = get_llm_response(gibberish_prompt)
    state["messages"].append(AIMessage(content="I’m sorry, I didn’t catch that—could you rephrase?"))
    return state

def handle_contradiction(state: ChatbotState) -> ChatbotState:
    """Handle contradictory or false statements"""
    user_msg = state["user_message"]
    
    contradiction_prompt = f"""
    The user made a statement that contains a contradiction or false information: "{user_msg}"
    
    Politely correct the misinformation with the accurate facts in just one sentence. 
    After the correction, Tell customer to continue reservation conversaton with no extra question added.
    
    Keep the response very precise and short; and redirect to restaurant services.
    """
    
    response = get_llm_response(contradiction_prompt)
    state["messages"].append(AIMessage(content=response))
    return state

def handle_profanity(state: ChatbotState) -> ChatbotState:
    """Handle profanity or disrespectful messages"""
    profanity_responses = [
        "Let's keep our conversation respectful, please. I'm here to help with your restaurant needs.",
        "I'd appreciate if we could maintain a polite conversation. How can I assist you with our restaurant services?",
        "Let's focus on how I can help you today. I can assist with reservations or answer questions about our restaurant.",
        "I'm here to provide helpful service. Please let me know how I can assist you with restaurant-related questions."
    ]
    
    import random
    response = random.choice(profanity_responses)
    state["messages"].append(AIMessage(content=response))
    return state


def determine_next_node(state: ChatbotState) -> str:
    """Route to appropriate handler based on intent"""
    intent = state["intent"]
    
    if state.get("reservation_complete") and intent == "reservation":
        return END
    
    route_map = {
        "reservation": "reservation",
        "knowledge": "knowledge", 
        "gibberish": "gibberish",
        "contradiction": "contradiction",
        "profanity": "profanity",
    }
    
    return route_map.get(intent, "reservation")  # Default to knowledge

def create_multi_tool_workflow():
    """Create the multi-tool workflow"""
    workflow = StateGraph(ChatbotState)
    
    # Add all nodes
    workflow.add_node("analyze_message", analyze_message)
    workflow.add_node("reservation", handle_reservation)
    workflow.add_node("knowledge", handle_knowledge)
    workflow.add_node("gibberish", handle_gibberish)
    workflow.add_node("contradiction", handle_contradiction)
    workflow.add_node("profanity", handle_profanity)
    
    # Set entry point
    workflow.set_entry_point("analyze_message")
    
    # Add conditional routing
    workflow.add_conditional_edges(
        "analyze_message",
        determine_next_node,
        {
            "reservation": "reservation",
            "knowledge": "knowledge",
            "gibberish": "gibberish", 
            "contradiction": "contradiction",
            "profanity": "profanity",
            END: END
        }
    )
    
    # All tools end the current turn
    for node in ["reservation", "knowledge", "gibberish", "contradiction", "profanity"]:
        workflow.add_edge(node, END)
    
    return workflow.compile()

# Initialize workflow
chatbot_workflow = create_multi_tool_workflow()

def get_or_create_session(session_id: str) -> ChatbotState:
    """Get existing session or create new one"""
    if session_id not in sessions:
        sessions[session_id] = {
            "messages": [AIMessage(content="Hello! I'm your restaurant assistant. I can help you with table reservations, answer questions about our restaurant, or assist with other inquiries. How can I help you today?")],
            "user_message": "",
            "intent": "",
            "date": None,
            "time": None,
            "persons": None,
            "reservation_complete": False,
            "conversation_history": []
        }
    return sessions[session_id]

@app.get("/knowledge")
async def get_knowledge_base():
    """Get all knowledge base entries"""
    return {
        "total_entries": len(KNOWLEDGE_BASE),
        "knowledge_base": KNOWLEDGE_BASE,
        "note": "This contains general world knowledge questions and is considered off-topic from restaurant reservations"
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """Main chat endpoint with multi-tool routing"""
    try:
        # Get or create session
        state = get_or_create_session(message.session_id)
        
        # Add to conversation history
        state["conversation_history"].append(message.message)
        
        # Add user message to state
        state["user_message"] = message.message
        state["messages"].append(HumanMessage(content=message.message))
        
        # Process through workflow
        result = chatbot_workflow.invoke(state)
        
        # Update session
        sessions[message.session_id] = result
        
        # Add assistant response to history
        ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
        if ai_messages:
            latest_response = ai_messages[-1].content
            result["conversation_history"].append(latest_response)
        else:
            latest_response = "I'm here to help!"
        
        # Prepare response
        reservation_status = {
            "date": result.get("date"),
            "time": result.get("time"),
            "persons": result.get("persons"),
            "complete": result.get("reservation_complete", False)
        }
        
        return ChatResponse(
            response=latest_response,
            intent=result.get("intent", "unknown"),
            reservation_status=reservation_status
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/")
async def root():
    """API information"""
    return {
        "message": "Multi-Tool Restaurant Chatbot API",
        "tools": [
            "reservation - Table booking assistance",
            "knowledge - General world knowledge Q&A (off-topic)", 
            "gibberish - Handle unclear messages",
            "contradiction - Correct false statements",
            "profanity - Handle inappropriate content"
        ],
        "endpoints": {
            "POST /chat": "Send message to chatbot",
            "GET /session/{session_id}": "Get session status",
            "DELETE /session/{session_id}": "Clear session",
            "GET /knowledge": "View knowledge base"
        }
    }

@app.get("/knowledge")
async def get_knowledge_base():
    """View the general world knowledge base"""
    return {"knowledge_base": KNOWLEDGE_BASE, "note": "This contains general world knowledge questions and is considered off-topic from restaurant reservations"}

@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get session status"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    state = sessions[session_id]
    return {
        "session_id": session_id,
        "current_intent": state.get("intent", "none"),
        "reservation_status": {
            "date": state.get("date"),
            "time": state.get("time"), 
            "persons": state.get("persons"),
            "complete": state.get("reservation_complete", False)
        },
        "message_count": len(state["messages"]),
        "conversation_length": len(state.get("conversation_history", []))
    }

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear session data"""
    if session_id in sessions:
        del sessions[session_id]
        return {"message": f"Session {session_id} cleared"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)