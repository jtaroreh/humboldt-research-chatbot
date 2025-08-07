import streamlit as st
import boto3
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup
st.set_page_config(page_title="Simple RAG Chatbot", page_icon="🤖")
st.title("Simple RAG Chatbot")

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

# Initialize
if "messages" not in st.session_state:
    st.session_state.messages = []

bedrock = setup_bedrock()
kb_id = os.getenv("KNOWLEDGE_BASE_ID")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Call Bedrock Knowledge Base
                response = bedrock.retrieve_and_generate(
                    input={
                        'text': (
                            "You are a helpful assistant. Be conversational and overly friendly. "
                            "At the end of the response, please ask the user if they have any other questions. "
                            "Always respond in clear, concise sentences. "
                            "At the end of the response, ask if the answers provided solved the user's query. "
                            "Acknowledge the question, and reiterate the user's question in your response. "
                            "When you use information from the knowledge base, cite it at the end.\n\n"
                            "You are a task-focused chatbot for a university’s research website. "
                            "Your primary goal is to help users quickly find the exact resource, "
                            "service, or page they need related to research at the university. "
                            "Behavior Guidelines: Always recognize the user's intent from natural language, "
                            "even if phrased informally or vaguely (e.g., 'funding stuff,' 'join a lab,' 'IRB form,'"
                            "'I need help with a grant'). If the intent is clear, respond with a concise, actionable reply "
                            "that includes a direct button, link, or next step (e.g., “Visit this page to apply: "
                            "Apply for Undergraduate Research”). If the intent is ambiguous, offer clear multiple-choice "
                            "options or ask a brief clarifying question. Avoid unnecessary filler text. No greetings or small "
                            "talk unless the user initiates it. Never give broad summaries of topics. Instead, route users to "
                            "specific destinations or ask questions that reduce ambiguity. When possible, match vague queries to "
                            "known university research services, documents, or programs. Tone: Helpful, efficient, and professional. "
                            "Like a university help desk that knows exactly where everything is."
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

                #test
                
                answer = response['output']['text']
                # st.write(response['citations'][0]['retrievedReferences'][0]['location'])
                st.write(answer)
                # st.write(response['citations'][0]['retrievedReferences'][0]['location']['webLocation']['url'])

                # Sources
                if 'citations' in response and response['citations']:
                    st.markdown("Sources:")
                    for i, citation in enumerate(response['citations'], 1):
                        for ref in citation.get('retrievedReferences', []):
                            location = ref.get('location', {})
                            if 'webLocation' in location:
                                url = location['webLocation']['url']
                                st.markdown(f"[{url}]({url})")
                            elif 's3Location' in location:
                                s3_uri = location['s3Location']['uri']
                                st.markdown(f"{i}. {s3_uri}")
                    
                # Add to chat history
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"Error: {e}")
