from flask import Flask, request, render_template, jsonify, session
import boto3
import os
import hashlib
import numpy as np
import uuid
from datetime import datetime
from dotenv import load_dotenv
from flask import Response, stream_with_context
import time
import re
import json

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecretkey")

# AWS Clients
def setup_bedrock():
    return boto3.client(
        'bedrock-agent-runtime',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        aws_session_token=os.getenv('AWS_SESSION_TOKEN'),
        region_name=os.getenv('AWS_DEFAULT_REGION')
    )

def setup_dynamodb():
    return boto3.resource(
        'dynamodb',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        aws_session_token=os.getenv('AWS_SESSION_TOKEN'),
        region_name=os.getenv('AWS_DEFAULT_REGION')
    )

bedrock = setup_bedrock()
dynamodb = setup_dynamodb()
table = dynamodb.Table('chatbot_history')
kb_id = os.getenv("KNOWLEDGE_BASE_ID")

# Embed function
def embed(text):
    hash_val = hashlib.sha256(text.encode()).digest()
    return np.frombuffer(hash_val[:128], dtype=np.uint8).astype(np.float32)

# Save to DynamoDB
def save_to_dynamodb(session_id, query, response, query_type="general"):
    table.put_item(
        Item={
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'response': response,
            'query_type': query_type
        }
    )

@app.route("/", methods=["GET"])
def index():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    # The initial suggestions for the homepage are now handled by the frontend
    return render_template("index.html")

# This endpoint is no longer needed as the logic is merged into chat_stream
# @app.route("/suggestions", methods=["POST"])

@app.route("/chat/stream", methods=["POST"])
def chat_stream():
    data = request.get_json()
    user_input = data.get("message")
    chat_history = data.get("history", [])

    # Handle initial page load to get topic suggestions
    if not user_input and not chat_history:
        def initial_suggestions():
            suggestions = [
                "What research programs does the Sponsored Programs Foundation support?",
                "What are the latest news and events?",
                "Where can I find forms and documents?",
                "What are the compliance requirements?",
                "What employment information is available?",
                "What board information is available?"
            ]
            payload = {"type": "suggestions", "content": suggestions}
            yield f"data: {json.dumps(payload)}\n\n"
            yield "data: [DONE]\n\n"
        return Response(stream_with_context(initial_suggestions()), mimetype='text/event-stream')


    # Embed chat history
    chat_history_embeddings = []
    for msg in chat_history[-6:]:
        content = msg.get("content", "")
        role = msg.get("role", "")
        metadata = {"role": role, "content": content}
        vec = embed(content)
        chat_history_embeddings.append((vec, metadata))

    chat_memory_snippets = "\n".join([
        f"{entry[1]['role'].capitalize()}: {entry[1]['content']}"
        for entry in chat_history_embeddings
    ])

    def generate():
        try:
            response = bedrock.retrieve_and_generate(
                input={
                    'text': (
                        f"CHAT HISTORY: {chat_history_embeddings}\n\n"
                        "If the user asks more than one question, please ask the user to prioritize the most important question and to answer that one first"
                        "If the user asks a question that does not produce a precise result which can be broken down into steps to accomplish, provide 2-3 suggested prompts the user could use that would provide more useful results at the end. "
                        "Enclose these suggested prompts within <SUGGESTIONS> and </SUGGESTIONS> tags. Do not include any other text in the suggestion block.\n"
                        "For example: <SUGGESTIONS>What about X?\nHow do I do Y?</SUGGESTIONS>\n\n"
                        "You are a helpful assistant. Be conversational and friendly. "
                        "Always respond in clear, concise sentences."
                        "Your primary goal is to help users quickly find the exact resource, service, or page they need related to research at the university."
                        "Acknowledge the question, and reiterate the user's question in your response. "
                        "In your response, break down the steps to solve the user's problem in a structured step by step workflow."
                        "Never give broad summaries of topics. Instead, route users to specific destinations."
                        "Suggest alternate contact details if applicable.\n\n"
                        f"User question: {user_input}"
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

            full_answer = response['output']['text']
            sources = []
            suggestions = []
            cleaned_answer = full_answer

            # --- Extract and Clean Suggestions ---
            suggestion_match = re.search(r'<SUGGESTIONS>(.*?)</SUGGESTIONS>', full_answer, re.DOTALL)
            if suggestion_match:
                suggestion_text = suggestion_match.group(1).strip()
                suggestions = [q.strip() for q in suggestion_text.split('\n') if q.strip()]
                cleaned_answer = full_answer[:suggestion_match.start()].strip()
            else:
                # Fallback logic if the model fails to use the tags
                if 'research' in user_input.lower() or 'grant' in user_input.lower():
                    suggestions = ["What is the Sponsored Programs Foundation?", "How do I contact the research office?", "What are the indirect cost rates?"]
                else:
                    suggestions = ["What services does the foundation provide?", "How can I contact the foundation?", "What are the foundation's policies?"]

            # --- Extract Sources ---
            if 'citations' in response and response['citations']:
                for citation in response['citations']:
                    for ref in citation.get('retrievedReferences', []):
                        location = ref.get('location', {})
                        if 'webLocation' in location:
                            sources.append(location['webLocation']['url'])
                        elif 's3Location' in location:
                            sources.append(location['s3Location']['uri'])

            # --- Yield Payloads to Frontend ---
            # 1. Suggestions
            if suggestions:
                payload = {"type": "suggestions", "content": suggestions[:3]} # Limit to 3
                yield f"data: {json.dumps(payload)}\n\n"

            # 2. Cleaned Answer
            payload = {"type": "answer", "content": cleaned_answer}
            yield f"data: {json.dumps(payload)}\n\n"

            # 3. Sources
            if sources:
                payload = {"type": "sources", "content": sources}
                yield f"data: {json.dumps(payload)}\n\n"

            # 4. Signal Completion
            yield "data: [DONE]\n\n"

            # Save to DB after sending response to user
            db_response = cleaned_answer + (f" [Sources: {' | '.join(sources)}]" if sources else "")
            save_to_dynamodb(session["session_id"], user_input, db_response, "knowledge_base")

        except Exception as e:
            payload = {"type": "error", "content": f"Error: {str(e)}"}
            yield f"data: {json.dumps(payload)}\n\n"
            yield "data: [DONE]\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream')

if __name__ == "__main__":
    app.run(debug=True)