import streamlit as st
import boto3
import os
import hashlib
import numpy as np
import uuid
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup
st.set_page_config(page_title="Lucky the Lumberjack Chatbot", page_icon="🪓")
st.title("Lucky the Lumberjack Chatbot")

# AWS Setups
@st.cache_resource
def setup_bedrock():
    return boto3.client(
        'bedrock-agent-runtime',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        aws_session_token=os.getenv('AWS_SESSION_TOKEN'),
        region_name=os.getenv('AWS_DEFAULT_REGION')
    )

@st.cache_resource
def setup_dynamodb():
    return boto3.resource(
        'dynamodb',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        aws_session_token=os.getenv('AWS_SESSION_TOKEN'),
        region_name=os.getenv('AWS_DEFAULT_REGION')
    )

# Embedding simulation (for real use, call an embedding model)
def embed(text):
    hash_val = hashlib.sha256(text.encode()).digest()
    return np.frombuffer(hash_val[:128], dtype=np.uint8).astype(np.float32)

# Add embedding to vector store
def store_embedding(text, metadata):
    vec = embed(text)
    st.session_state.clarification_embeddings.append((vec, metadata))

# Function to get emoji for different message types
def get_emoji_for_message(message_type):
    emoji_map = {
        "user": "🙋‍♀️",  # User talking
        "greeting": "🌲",  # Lucky intro energy
        "bot": "🧠",  # Smart/helpful answer
        "thinking": "🤔",  # Suspense & charm
        "error": "😵",  # Bot brain fart moment
        "fun_extra": "🦫"  # Lucky's woodland cousin
    }
    return emoji_map.get(message_type, "💬")

# Function to get time-based greeting
def get_greeting():
    from datetime import datetime
    hour = datetime.now().hour
    if hour < 12:
        return "Good morning 🌲"
    elif hour < 18:
        return "Good afternoon 🌤️"
    else:
        return "Good evening 🌙"

# Initialize
if "messages" not in st.session_state:
    greeting = get_greeting()
    st.session_state.messages = [
        {"role": "assistant", "content": f"🪓 {greeting}! I'm Lucky, your Lumberjack assistant! How can I help you today?"}
    ]
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "suggested_questions" not in st.session_state:
    st.session_state.suggested_questions = []

bedrock = setup_bedrock()
dynamodb = setup_dynamodb()
table = dynamodb.Table('chatbot_history')
kb_id = os.getenv("KNOWLEDGE_BASE_ID")

def save_to_dynamodb(session_id, query, response, query_type="general"):
    try:
        table.put_item(
            Item={
                'session_id': session_id,
                'timestamp': datetime.now().isoformat(),
                'query': query,
                'response': response,
                'query_type': query_type
            }
        )
    except Exception as e:
        st.error(f"Database save failed: {e}")

# Display chat history
for message in st.session_state.messages:
    # Determine avatar emoji based on message content
    if message["role"] == "user":
        avatar_emoji = get_emoji_for_message("user")
    else:
        # Check message content to determine bot emoji type
        content = message["content"]
        if "Good morning" in content or "Good afternoon" in content or "Good evening" in content:
            avatar_emoji = get_emoji_for_message("greeting")
        elif "Error:" in content:
            avatar_emoji = get_emoji_for_message("error")
        else:
            avatar_emoji = get_emoji_for_message("bot")
    
    with st.chat_message(message["role"], avatar=avatar_emoji):
        st.write(message["content"])

chat_history_embeddings = []

for message in st.session_state.messages[-6:]:  # or more/less
    content = message["content"]
    role = message["role"]
    metadata = {"role": role, "content": content}
    vec = embed(content)
    chat_history_embeddings.append((vec, metadata))

chat_memory_snippets = "\n".join([
    f"{entry[1]['role'].capitalize()}: {entry[1]['content']}"
    for entry in chat_history_embeddings
])

# Check if we need to process the last message (for button clicks)
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    last_message = st.session_state.messages[-1]["content"]
    
    # Check if this user message needs a response
    needs_response = True
    if len(st.session_state.messages) >= 2:
        for i in range(len(st.session_state.messages) - 1):
            if (st.session_state.messages[i]["role"] == "user" and 
                st.session_state.messages[i]["content"] == last_message and
                i + 1 < len(st.session_state.messages) and
                st.session_state.messages[i + 1]["role"] == "assistant"):
                needs_response = False
                break
    
    if needs_response:
        with st.chat_message("assistant", avatar=get_emoji_for_message("bot")):
            with st.spinner(f"{get_emoji_for_message('thinking')} Thinking..."):
                try:
                    # Call Bedrock Knowledge Base
                    response = bedrock.retrieve_and_generate(
                        input={
                            'text': (
                                f"CHAT HISTORY: {chat_history_embeddings}\n"
                                "You are a helpful assistant. Be conversational and overly friendly. "
                                "Always respond in clear, concise sentences. "
                                "Acknowledge the question, and reiterate the user's question in your response. "
                                "In your response, break down the steps to solve the user's problem in a structured step by step workflow"
                                "If the user asks more than one question, please ask the user to prioritize the most important question and to answer that one first"
                                "If the user asks a question that does not produce a precise result which can be broken down into steps to accomplish, provide a prompt that would provide a more useful result. "
                                "Never give broad summaries of topics. Instead, route users to specific destinations."
                                "When you use information from the knowledge base, cite it at the end.\n\n"
                                "IMPORTANT: At the end of your response, suggest 2-3 specific follow-up questions that are directly related to the topic you just discussed. "
                                "Format these as natural questions that start with phrases like 'What about...?', 'How do I...?', 'Where can I find...?', 'When is...?', etc. "
                                "Make sure these questions are specific to the content you just provided, not generic questions.\n\n"
                                "Suggest alternate contact details if applicable."
                                f"User question: {last_message}"
                            )
                        },
                        retrieveAndGenerateConfiguration={
                            'type': 'KNOWLEDGE_BASE',
                            'knowledgeBaseConfiguration': {
                                'knowledgeBaseId': kb_id,
                                'modelArn': f'arn:aws:bedrock:us-west-2::foundation-model/{os.getenv("BEDROCK_MODEL_ID")}'
                            }
                        }
                    )

                    answer = response['output']['text']
                    st.write(answer)

                    # Sources
                    if 'citations' in response and response['citations']:
                        st.markdown("**📚 Sources:**")
                        for i, citation in enumerate(response['citations'], 1):
                            for ref in citation.get('retrievedReferences', []):
                                location = ref.get('location', {})
                                if 'webLocation' in location:
                                    url = location['webLocation']['url']
                                    st.markdown(f"• [{url}]({url})")
                                elif 's3Location' in location:
                                    s3_uri = location['s3Location']['uri']
                                    st.markdown(f"• {s3_uri}")
                        
                    # Save to DynamoDB
                    citations_text = ""
                    if 'citations' in response and response['citations']:
                        citations_list = []
                        for citation in response['citations']:
                            for ref in citation.get('retrievedReferences', []):
                                location = ref.get('location', {})
                                if 'webLocation' in location:
                                    citations_list.append(location['webLocation']['url'])
                                elif 's3Location' in location:
                                    citations_list.append(location['s3Location']['uri'])
                        citations_text = " | ".join(citations_list)
                    
                    full_response = answer + (f" [Sources: {citations_text}]" if citations_text else "")
                    save_to_dynamodb(st.session_state.session_id, last_message, full_response, "knowledge_base")
                    
                    # Extract suggested questions from the response
                    import re
                    suggested_questions = []
                    lines = answer.split('\n')
                    
                    # Look for questions that start with common question words
                    question_starters = ['What about', 'How do', 'Where can', 'When is', 'Who should', 'Why is', 'Which', 'What are', 'How can', 'What if']
                    
                    for line in lines:
                        line = line.strip()
                        if line.endswith('?') and len(line) > 15:
                            # Clean up the question
                            clean_question = re.sub(r'^[\-\*\d\.\s]+', '', line).strip()
                            
                            # Check if it starts with intelligent question words
                            if any(clean_question.startswith(starter) for starter in question_starters):
                                suggested_questions.append(clean_question)
                            # Also include questions that don't start with generic phrases
                            elif (not clean_question.startswith('Do you') and 
                                  not clean_question.startswith('Would you') and
                                  not clean_question.startswith('Can you') and
                                  not clean_question.startswith('Are you') and
                                  len(clean_question) > 20):
                                suggested_questions.append(clean_question)
                    
                    # If no intelligent questions found, generate topic-specific ones based on the user's question
                    if not suggested_questions:
                        user_question_lower = last_message.lower()
                        if 'research' in user_question_lower:
                            suggested_questions = [
                                "What funding opportunities are available for research?",
                                "How do I apply for research grants?",
                                "Where can I find research collaboration opportunities?"
                            ]
                        elif 'employment' in user_question_lower or 'job' in user_question_lower:
                            suggested_questions = [
                                "What are the application requirements for faculty positions?",
                                "How do I submit my application materials?",
                                "When are the application deadlines?"
                            ]
                        elif 'compliance' in user_question_lower:
                            suggested_questions = [
                                "What are the reporting requirements?",
                                "How do I ensure I'm meeting all compliance standards?",
                                "Where can I find the compliance checklist?"
                            ]
                        elif 'forms' in user_question_lower or 'documents' in user_question_lower:
                            suggested_questions = [
                                "How do I submit completed forms?",
                                "What documents do I need to provide?",
                                "Where can I get help filling out forms?"
                            ]
                        else:
                            suggested_questions = [
                                "What are the next steps I should take?",
                                "How do I get more specific information about this topic?",
                                "Where can I find additional resources?"
                            ]
                    
                    st.session_state.suggested_questions = suggested_questions[:3]  # Limit to 3
                    
                    # Build sources for storage (avoid duplicates)
                    sources_for_storage = ""
                    if 'citations' in response and response['citations']:
                        unique_sources = set()
                        for citation in response['citations']:
                            for ref in citation.get('retrievedReferences', []):
                                location = ref.get('location', {})
                                if 'webLocation' in location:
                                    unique_sources.add(location['webLocation']['url'])
                                elif 's3Location' in location:
                                    unique_sources.add(location['s3Location']['uri'])
                        
                        if unique_sources:
                            sources_for_storage = "\n\n**📚 Sources:**\n\n"
                            for source in sorted(unique_sources):
                                if source.startswith('http'):
                                    sources_for_storage += f"[{source}]({source})\n\n"
                                else:
                                    sources_for_storage += f"{source}\n\n"
                    
                    # Add to chat history with sources
                    st.session_state.messages.append({"role": "assistant", "content": answer + sources_for_storage})
                    st.rerun()
      
                except Exception as e:
                    error_msg = f"Error: {e}"
                    st.error(error_msg)
                    save_to_dynamodb(st.session_state.session_id, last_message, error_msg, "error")
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    st.rerun()

# Chat input
if prompt := st.chat_input("How can I help you today?"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=get_emoji_for_message("user")):
        st.write(prompt)

    # Get AI response
    with st.chat_message("assistant", avatar=get_emoji_for_message("bot")):
        with st.spinner(f"{get_emoji_for_message('thinking')} Thinking..."):
            try:
                # Call Bedrock Knowledge Base
                response = bedrock.retrieve_and_generate(
                    input={
                        'text': (
                            f"CHAT HISTORY: {chat_history_embeddings}\n"
                            "You are a helpful assistant. Be conversational and overly friendly. "
                            #"At the end of the response, please ask the user if they have any other questions."
                            "Always respond in clear, concise sentences. "
                            #"At the end of the response, ask if the answers provided solved the user's query. "
                            "Acknowledge the question, and reiterate the user's question in your response. "
                            "In your response, break down the steps to solve the user's problem in a structured step by step workflow"
                            "If the user asks more than one question, please ask the user to prioritize the most important question and to answer that one first"
                            "If the user asks a question that does not produce a precise result which can be broken down into steps to accomplish, provide a prompt that would provide a more useful result. "
                            "Never give broad summaries of topics. Instead, route users to specific destinations."
                            "When you use information from the knowledge base, cite it at the end.\n\n"
                            "IMPORTANT: At the end of your response, suggest 2-3 specific follow-up questions that are directly related to the topic you just discussed. "
                            "Format these as natural questions that start with phrases like 'What about...?', 'How do I...?', 'Where can I find...?', 'When is...?', etc. "
                            "Make sure these questions are specific to the content you just provided, not generic questions.\n\n"
                            "Suggest alternate contact details if applicable."
                            f"User question: {prompt}"
                        )
                    },
                    retrieveAndGenerateConfiguration={
                        'type': 'KNOWLEDGE_BASE',
                        'knowledgeBaseConfiguration': {
                            'knowledgeBaseId': kb_id,
                            'modelArn': f'arn:aws:bedrock:us-west-2::foundation-model/{os.getenv("BEDROCK_MODEL_ID")}'
                        }
                    }
                )

                answer = response['output']['text']
                # st.write(response['citations'][0]['retrievedReferences'][0]['location'])
                st.write(answer)
                # st.write(response['citations'][0]['retrievedReferences'][0]['location']['webLocation']['url'])

                # Sources
                if 'citations' in response and response['citations']:
                    st.markdown("**📚 Sources:**")
                    for i, citation in enumerate(response['citations'], 1):
                        for ref in citation.get('retrievedReferences', []):
                            location = ref.get('location', {})
                            if 'webLocation' in location:
                                url = location['webLocation']['url']
                                st.markdown(f"• [{url}]({url})")
                            elif 's3Location' in location:
                                s3_uri = location['s3Location']['uri']
                                st.markdown(f"• {s3_uri}")
                    
                # Save to DynamoDB
                citations_text = ""
                if 'citations' in response and response['citations']:
                    citations_list = []
                    for citation in response['citations']:
                        for ref in citation.get('retrievedReferences', []):
                            location = ref.get('location', {})
                            if 'webLocation' in location:
                                citations_list.append(location['webLocation']['url'])
                            elif 's3Location' in location:
                                citations_list.append(location['s3Location']['uri'])
                    citations_text = " | ".join(citations_list)
                
                full_response = answer + (f" [Sources: {citations_text}]" if citations_text else "")
                save_to_dynamodb(st.session_state.session_id, prompt, full_response, "knowledge_base")
                
                # Extract suggested questions from the response
                import re
                suggested_questions = []
                lines = answer.split('\n')
                
                # Look for questions that start with common question words
                question_starters = ['What about', 'How do', 'Where can', 'When is', 'Who should', 'Why is', 'Which', 'What are', 'How can', 'What if']
                
                for line in lines:
                    line = line.strip()
                    if line.endswith('?') and len(line) > 15:
                        # Clean up the question
                        clean_question = re.sub(r'^[\-\*\d\.\s]+', '', line).strip()
                        
                        # Check if it starts with intelligent question words
                        if any(clean_question.startswith(starter) for starter in question_starters):
                            suggested_questions.append(clean_question)
                        # Also include questions that don't start with generic phrases
                        elif (not clean_question.startswith('Do you') and 
                              not clean_question.startswith('Would you') and
                              not clean_question.startswith('Can you') and
                              not clean_question.startswith('Are you') and
                              len(clean_question) > 20):
                            suggested_questions.append(clean_question)
                
                # If no intelligent questions found, generate topic-specific ones based on the user's question
                if not suggested_questions:
                    user_question_lower = prompt.lower()
                    if 'research' in user_question_lower:
                        suggested_questions = [
                            "What funding opportunities are available for research?",
                            "How do I apply for research grants?",
                            "Where can I find research collaboration opportunities?"
                        ]
                    elif 'employment' in user_question_lower or 'job' in user_question_lower:
                        suggested_questions = [
                            "What are the application requirements for faculty positions?",
                            "How do I submit my application materials?",
                            "When are the application deadlines?"
                        ]
                    elif 'compliance' in user_question_lower:
                        suggested_questions = [
                            "What are the reporting requirements?",
                            "How do I ensure I'm meeting all compliance standards?",
                            "Where can I find the compliance checklist?"
                        ]
                    elif 'forms' in user_question_lower or 'documents' in user_question_lower:
                        suggested_questions = [
                            "How do I submit completed forms?",
                            "What documents do I need to provide?",
                            "Where can I get help filling out forms?"
                        ]
                    else:
                        suggested_questions = [
                            "What are the next steps I should take?",
                            "How do I get more specific information about this topic?",
                            "Where can I find additional resources?"
                        ]
                
                st.session_state.suggested_questions = suggested_questions[:3]  # Limit to 3
                
                # Build sources for storage (avoid duplicates)
                sources_for_storage = ""
                if 'citations' in response and response['citations']:
                    unique_sources = set()
                    for citation in response['citations']:
                        for ref in citation.get('retrievedReferences', []):
                            location = ref.get('location', {})
                            if 'webLocation' in location:
                                unique_sources.add(location['webLocation']['url'])
                            elif 's3Location' in location:
                                unique_sources.add(location['s3Location']['uri'])
                    
                    if unique_sources:
                        sources_for_storage = "\n\n**📚 Sources:**\n\n"
                        for source in sorted(unique_sources):
                            if source.startswith('http'):
                                sources_for_storage += f"[{source}]({source})\n\n"
                            else:
                                sources_for_storage += f"{source}\n\n"
                
                # Add to chat history with sources
                st.session_state.messages.append({"role": "assistant", "content": answer + sources_for_storage})
  
            except Exception as e:
                error_msg = f"Error: {e}"
                st.error(error_msg)
                save_to_dynamodb(st.session_state.session_id, prompt, error_msg, "error")

# Show AI-suggested questions as clickable buttons
if st.session_state.suggested_questions:
    st.write("**🤔 You might also want to ask:**")
    for i, question in enumerate(st.session_state.suggested_questions):
        if st.button(f"❓ {question}", key=f"ai_suggest_{i}"):
            st.session_state.messages.append({"role": "user", "content": question})
            st.rerun()

# Show suggestion buttons (only show if just greeting message exists)
if len(st.session_state.messages) == 1:
    st.write("**🏛️ Explore Topics:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔬 Research", key="suggest1"):
            st.session_state.messages.append({"role": "user", "content": "What research opportunities are available?"})
            st.rerun()

    with col2:
        if st.button("📰 News and Events", key="suggest2"):
            st.session_state.messages.append({"role": "user", "content": "What are the latest news and events?"})
            st.rerun()

    with col3:
        if st.button("📁 Forms Library", key="suggest3"):
            st.session_state.messages.append({"role": "user", "content": "Where can I find forms and documents?"})
            st.rerun()
            
    # Additional row of suggestions
    col4, col5, col6 = st.columns(3)

    with col4:
        if st.button("⚖️ Compliance", key="suggest4"):
            st.session_state.messages.append({"role": "user", "content": "What are the compliance requirements?"})
            st.rerun()

    with col5:
        if st.button("💼 Employment", key="suggest5"):
            st.session_state.messages.append({"role": "user", "content": "What employment information is available?"})
            st.rerun()

    with col6:
        if st.button("🏢 Board", key="suggest6"):
            st.session_state.messages.append({"role": "user", "content": "What board information is available?"})
            st.rerun()