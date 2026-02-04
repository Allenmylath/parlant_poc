import streamlit as st
from sarvamai import SarvamAI
from parlant.client import ParlantClient
import time

# Page configuration
st.set_page_config(
    page_title="Police Website Assistant",
    page_icon="🚔",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for WhatsApp-like chat interface
st.markdown("""
<style>
    @media (max-width: 768px) {
        .main { padding: 0.5rem; }
    }
    
    .user-message {
        background-color: #DCF8C6;
        padding: 10px 15px;
        border-radius: 15px;
        margin: 10px 0;
        max-width: 70%;
        margin-left: auto;
        text-align: left;
        word-wrap: break-word;
    }
    
    .assistant-message {
        background-color: #E8E8E8;
        padding: 10px 15px;
        border-radius: 15px;
        margin: 10px 0;
        max-width: 70%;
        margin-right: auto;
        word-wrap: break-word;
    }
    
    .typing-indicator {
        background-color: #F0F0F0;
        padding: 8px 12px;
        border-radius: 10px;
        margin: 5px 0;
        max-width: 30%;
        text-align: center;
        font-size: 0.85rem;
        color: #666;
        font-style: italic;
    }
    
    .stButton>button {
        border-radius: 20px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
SARVAM_API_KEY = "sk_0aid8hoc_4BSxXwlPGz7wYBMhq9vT75WR"
PARLANT_BASE_URL = "https://parlant-poc.onrender.com"

# Initialize clients
@st.cache_resource
def init_clients():
    sarvam_client = SarvamAI(api_subscription_key=SARVAM_API_KEY)
    parlant_client = ParlantClient(base_url=PARLANT_BASE_URL)
    return sarvam_client, parlant_client

sarvam_client, parlant_client = init_clients()

# Initialize Parlant session
def init_parlant_session():
    try:
        agent = parlant_client.agents.create(
            name="Police Website Assistant",
            description="Helpful assistant for police website queries in any language"
        )
        
        session = parlant_client.sessions.create(
            agent_id=agent.id,
            title=f"Chat {time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        return agent.id, session.id
    except Exception as e:
        st.error(f"Failed to initialize: {str(e)}")
        return None, None

# Initialize session state
if 'agent_id' not in st.session_state or 'session_id' not in st.session_state:
    agent_id, session_id = init_parlant_session()
    st.session_state.agent_id = agent_id
    st.session_state.session_id = session_id

if 'question_text' not in st.session_state:
    st.session_state.question_text = ""
if 'last_audio' not in st.session_state:
    st.session_state.last_audio = None
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'last_event_count' not in st.session_state:
    st.session_state.last_event_count = 0

# Helper functions
def transcribe_audio(audio_file) -> str:
    try:
        response = sarvam_client.speech_to_text.transcribe(
            file=audio_file,
            model="saarika:v2.5",
            language_code="unknown"
        )
        if hasattr(response, 'transcript'):
            return response.transcript
        return str(response)
    except Exception as e:
        st.error(f"Transcription error: {str(e)}")
        return ""

def get_conversation_events():
    """Retrieve and parse all conversation events"""
    events = parlant_client.sessions.list_events(
        session_id=st.session_state.session_id,
        min_offset=0
    )
    
    conversation = []
    for event in events:
        if event.kind == 'message':
            message_text = event.data.get('message', '')
            if message_text:
                conversation.append({
                    'source': event.source,
                    'message': message_text,
                    'type': 'message'
                })
    
    return conversation, len(events)

def send_message_to_parlant(message: str) -> bool:
    try:
        # Send customer message
        parlant_client.sessions.create_event(
            session_id=st.session_state.session_id,
            kind="message",
            source="customer",
            message=message
        )
        
        # Poll for agent response with timeout
        max_wait = 15  # seconds
        poll_interval = 2  # seconds
        elapsed = 0
        
        initial_count = st.session_state.last_event_count
        
        progress_placeholder = st.empty()
        
        while elapsed < max_wait:
            progress_placeholder.info(f"⏳ Waiting for response... ({elapsed}s)")
            
            conversation, event_count = get_conversation_events()
            
            # Check if we have new agent messages
            agent_messages = [m for m in conversation if m['source'] == 'ai_agent']
            
            # If event count increased and we have agent messages, response is ready
            if event_count > initial_count and len(agent_messages) > len([m for m in st.session_state.conversation_history if m.get('source') == 'ai_agent']):
                st.session_state.conversation_history = conversation
                st.session_state.last_event_count = event_count
                progress_placeholder.empty()
                return True
            
            time.sleep(poll_interval)
            elapsed += poll_interval
        
        # Timeout - update with whatever we have
        conversation, event_count = get_conversation_events()
        st.session_state.conversation_history = conversation
        st.session_state.last_event_count = event_count
        progress_placeholder.warning("⚠️ Response timeout - showing current state")
        
        return True
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return False

# UI
st.title("🚔 Police Website Assistant")
st.markdown("Ask your question using voice or text")

# Input Section
st.markdown("### 🎤 Ask Your Question")

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("**Voice Input:**")
    audio_value = st.audio_input("Record", label_visibility="collapsed")
    
    if audio_value:
        audio_id = audio_value.name if hasattr(audio_value, 'name') else str(audio_value.size)
        
        if audio_id != st.session_state.last_audio:
            st.session_state.last_audio = audio_id
            with st.spinner("🎯 Transcribing..."):
                transcription = transcribe_audio(audio_value)
                if transcription:
                    st.session_state.question_text = transcription
                    st.rerun()

with col2:
    st.markdown("**Or Type:**")
    question = st.text_area(
        "Type your question",
        value=st.session_state.question_text,
        placeholder="Type here or use voice...",
        height=100,
        label_visibility="collapsed"
    )
    
    if question != st.session_state.question_text:
        st.session_state.question_text = question

# Buttons
col1, col2, col3 = st.columns(3)

with col1:
    if st.session_state.question_text and st.button("🗑️ Clear", use_container_width=True):
        st.session_state.question_text = ""
        st.session_state.last_audio = None
        st.rerun()

with col2:
    send_button = st.button("🔍 Send", use_container_width=True, type="primary")

with col3:
    if st.button("🔄 New Chat", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# Process message
if send_button and st.session_state.question_text.strip():
    success = send_message_to_parlant(st.session_state.question_text.strip())
    if success:
        st.session_state.question_text = ""
        st.session_state.last_audio = None
        st.rerun()
elif send_button:
    st.warning("⚠️ Please enter a question")

# Display conversation
if st.session_state.conversation_history:
    st.markdown("---")
    st.markdown("### 💬 Conversation")
    
    for msg in st.session_state.conversation_history:
        if msg['source'] == 'customer':
            st.markdown(f"""
            <div class="user-message">
                <b>You:</b><br>{msg['message']}
            </div>
            """, unsafe_allow_html=True)
        elif msg['source'] == 'ai_agent':
            st.markdown(f"""
            <div class="assistant-message">
                <b>🚔 Assistant:</b><br>{msg['message']}
            </div>
            """, unsafe_allow_html=True)
else:
    st.info("👋 Start chatting by typing or using voice!")

# Footer
st.markdown("---")
st.caption("Powered by SHIS.ai | Parlant & Sarvam AI")
