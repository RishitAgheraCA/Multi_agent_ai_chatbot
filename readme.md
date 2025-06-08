# Agentic AI Chatbot with LangGraph

A sophisticated conversational AI chatbot built with LangGraph that handles multi-turn conversations, manages session state, and includes ethical guardrails. The bot specializes in restaurant reservations while gracefully handling topic switches and adversarial inputs.

## Features

* **Multi-turn Conversations**: Maintains context across conversation turns
* **Fuzzy Time Parsing**: Handles ambiguous time expressions like "this weekend or maybe Monday morning"
* **Clarification Handling**: Asks follow-up questions when input is ambiguous
* **Topic Switch & Resume**: Seamlessly handles off-topic queries and returns to original conversation
* **Ethical Guardrails**: Detects and handles gibberish, profanity, and contradictory statements
* **Session Management**: Isolated sessions for multiple users with persistent state

## Architecture

The chatbot uses **LangGraph** to create a state machine with the following nodes:

1. **Input Validation Node**: Checks for gibberish, profanity, and contradictions
2. **Intent Classification Node**: Determines user intent and handles factual queries
3. **Booking Flow Node**: Manages the reservation process with slot filling
4. **Confirmation Node**: Handles final booking confirmation

## Architecture Diagram

![ArchitectureDiagram](./aichatbotgrapharchitecture.png)


## Quick Start

### Prerequisites

* Python 3.10+
* Docker (optional)
* Groq API key

To get a Groq API key:

1. Visit [https://console.groq.com](https://console.groq.com)
2. Sign in and navigate to the API section
3. Generate a new API key and copy it for use in your environment

### Environment Setup

1. Clone the repository:

```bash
git clone git@github.com:RishitAgheraCA/Multi_agent_ai_chatbot.git
cd Multi_agent_ai_chatbot
```

2. Set up environment variables:

```bash
# Create .env file
echo "GROQ_API_KEY=your_groq_api_key_here" > .env
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

### Running Locally

```bash
# Run the application
python main.py

# Or with uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

### Running with Docker Compose

Instead of creating a new Docker Compose file, just pull and run using existing configuration:

```bash
docker compose pull
docker compose up -d
```

## API Endpoints

### POST /chat

Main conversation endpoint.

**Endpoint:** `http://0.0.0.0:8000/chat`

**Request:**

```json
{
  "message": "I want to book a table.",
  "session_id": "<session_id:str>"
}
```

Get All Knowledge Base Entries

**Endpoint:** GET `http://0.0.0.0:8000/knowledge`

Retrieves all entries in the knowledge base.

## Example Conversation Flow

```
User: I want to book a table this weekend
AI: It sounds like you want to book a table this weekend. That's a great start! This weekend could be either Saturday or Sunday. Do you prefer Saturday, Sunday, or maybe you're open to other options?

User: Sunday, please!
AI: I've got the date as Sunday. Now, let's talk about the time. What time would you prefer to dine on Sunday? Would you like lunch, dinner, or a specific time like 1 pm or 6 pm?

User: For 4 pm.
AI: I've got the time as 4 pm. Now, let's talk about the number of people. How many people will be dining with you?

User: By the way, what’s the capital of Australia?
AI: The capital of Australia is Canberra. Now, shall we get back to your reservation?

User: asdflkj asdfasdfsadf
AI: I’m sorry, I didn’t catch that—could you rephrase?

User: Is 30 °C freezing?
AI: 30°C is actually a warm temperature, not freezing. Please proceed with your reservation inquiry.

User: Party of 20 people
AI: Reservation details
 Date: Sunday,
 Time: 4 pm,
 Persons: 20
 Please respond with "yes please confirm" or let me know for any change.

User: what the f..
AI: I'd appreciate if we could maintain a polite conversation. How can I assist you with our restaurant services?

User: Yes please confirm reservation
AI: Thanks for confirmation, The table will be reserved for you. See you soon.
```

## Key Components

### 1. NLU Processor

* Extracts booking intents from natural language
* Parses fuzzy time expressions
* Identifies party sizes and other booking parameters

### 2. Ethical Filter

* **Gibberish Detection**: Identifies nonsensical input patterns
* **Profanity Filter**: Catches inappropriate language
* **Contradiction Detection**: Identifies logical inconsistencies

### 3. Knowledge Base

* Stores factual information for quick retrieval
* Handles common questions about geography, science, etc.
* Easily extensible with new facts

### 4. Session Management

* Maintains isolated state for each user
* Persists conversation context across turns
* Handles slot filling for booking process

## Configuration

### Environment Variables

* `GROQ_API_KEY`: Required for LLM functionality

### Customization

The chatbot can be easily extended:

1. **Add new intents**: Modify the `NLUProcessor` class
2. **Expand knowledge base**: Update the `KnowledgeBase.facts` dictionary
3. **Add new filters**: Extend the `EthicalFilter` class
4. **Modify conversation flow**: Update the LangGraph state machine

## Monitoring & Logging

* **Structured Logging**: All events are logged with context
* **Error Handling**: Graceful error recovery with user-friendly messages
* **Session Tracking**: Full conversation history for debugging

## Deployment Considerations

### Security

* API keys are handled securely through environment variables
* Input validation prevents common attack vectors

## Troubleshooting

### Common Issues

1. **GROQ\_API\_KEY not set**: Ensure the environment variable is properly configured
2. **Import errors**: Check that all dependencies are installed with correct versions
3. **Port conflicts**: Ensure port 8000 is available or change the port configuration
4. **Docker build fails**: Verify Docker is installed and running

## Future Enhancements

Here are potential enhancements to elevate the chatbot’s capability:

* **Database Integration**: Replace in-memory storage with PostgreSQL or MongoDB
* **Model Upgrade**: Use a larger language model for improved conversational quality
* **Temporal Context Awareness**: Inject current date/time to handle relative date references like “tomorrow” or “this weekend” more accurately
* **RAG-Based Knowledge Base**: Replace the mock knowledge base with Retrieval-Augmented Generation (RAG) embeddings
* **Edit/Cancel Reservations**: Add functionality for modifying or cancelling bookings
* **WebSocket Support**: Implement real-time conversation using WebSockets to allow bot-generated follow-ups even without new user input
