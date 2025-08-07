import streamlit as st
import boto3
import pandas as pd
from datetime import datetime
import os
import json
from dotenv import load_dotenv
from collections import Counter

#v1
load_dotenv()

st.set_page_config(page_title="Chat Dashboard", layout="wide")
st.markdown("<h1 style='text-align: center;'>📊 CHAT METRICS</h1>", unsafe_allow_html=True)

@st.cache_resource
def setup_dynamodb():
    return boto3.resource(
        'dynamodb',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        aws_session_token=os.getenv('AWS_SESSION_TOKEN'),
        region_name=os.getenv('AWS_DEFAULT_REGION')
    )

@st.cache_resource
def setup_bedrock():
    return boto3.client(
        'bedrock-runtime',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        aws_session_token=os.getenv('AWS_SESSION_TOKEN'),
        region_name=os.getenv('AWS_DEFAULT_REGION')
    )

dynamodb = setup_dynamodb()
bedrock = setup_bedrock()
table = dynamodb.Table('chatbot_history')

@st.cache_data(ttl=300)
def generate_common_questions(queries):
    try:
        all_queries = "\n".join(queries[:50])  # Limit to 50 queries
        prompt = f"""From all user queries that are submitted, group the queries that are similar
        and generate ordered categories (from most searched to least) which would summarize them well. 
        Create a maximum of 5 categories. List the keywords (separated by commas) used to create those categories.
        Also, create a question on a next line which best summarizes whe user queries within each category.
        Don't create an example question summary\n\n{all_queries}"""
        
        response = bedrock.invoke_model(
            modelId='us.amazon.nova-lite-v1:0',
            body=json.dumps({
                'messages': [{'role': 'user', 'content': [{'text': prompt}]}],
                'inferenceConfig': {'max_new_tokens': 500}
            })
        )
        
        result = json.loads(response['body'].read())
        return result['output']['message']['content'][0]['text']
    except Exception as e:
        return f"Error generating questions: {e}"

@st.cache_data(ttl=1)
def fetch_data():
    response = table.scan()
    df = pd.DataFrame(response['Items'])
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp', ascending=False)
    return df

df = fetch_data()

if not df.empty:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Queries", len(df))
    with col2:
        st.metric("Unique Sessions", df['session_id'].nunique())
    with col3:
        avg_queries = len(df) / df['session_id'].nunique()
        st.metric("Avg Queries/Session", f"{avg_queries:.1f}")
    
    # Query times bar chart
    st.subheader("Query Times")
    _, chart_col, _ = st.columns([1, 2, 1])
    with chart_col:
        df['hour'] = df['timestamp'].dt.hour
        hourly_counts = df['hour'].value_counts().reindex(range(24), fill_value=0)
        chart_data = pd.DataFrame({
            'Hour': [f"{h}:00" for h in range(24)],
            'Number of Queries': hourly_counts.values
        }).set_index('Hour')
        st.bar_chart(chart_data)
    
    # Popular Topics
    with st.expander("Popular Topics - Dynamic Analysis"):
        if len(df) > 0:
            with st.spinner("Generating common questions..."):
                common_questions = generate_common_questions(df['query'].tolist())
                st.write(common_questions)
        else:
            st.write("No queries available for analysis")
    
    # Popular topics
    # st.subheader("Popular Topics")
    # topics = {
    #     'Policy': ['policy', 'policies', 'rule', 'rules', 'guideline'],
    #     'Contact': ['contact', 'email', 'phone', 'call', 'reach'],
    #     'Schedule': ['schedule', 'time', 'date', 'when', 'deadline'],
    #     'General': []
    # }
    
    # topic_counts = {topic: 0 for topic in topics}
    # for query in df['query'].tolist():
    #     query_lower = query.lower()
    #     matched = False
    #     for topic, keywords in topics.items():
    #         if topic == 'General':
    #             continue
    #         if any(keyword in query_lower for keyword in keywords):
    #             topic_counts[topic] += 1
    #             matched = True
    #             break
    #     if not matched:
    #         topic_counts['General'] += 1
    
    # sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    # for i, (topic, count) in enumerate(sorted_topics, 1):
    #     with st.expander(f"**{i}.** {topic} ({count} queries)"):
    #         if topic == 'General':
    #             st.write("Queries that don't match specific keywords")
    #         else:
    #             st.write(f"Keywords: {', '.join(topics[topic])}")
    

    
    # # Remaining queries dropdown
    # remaining_queries = [q for q, _ in query_counts.most_common()[3:]]
    # if remaining_queries:
    #     selected_query = st.selectbox("Other Queries", ["Select a query..."] + remaining_queries)
    #     if selected_query != "Select a query...":
    #         count = query_counts[selected_query]
    #         st.write(f"**Query:** {selected_query} ({count} times)")
    
    st.subheader("Recent Chats (10)")
    col1, col2, _ , _ = st.columns([1, 1, 1, 1])
    with col1:
        if st.button("Collapse All"):
            st.session_state.expand_all = False
            st.rerun()
    with col1:
        if st.button("Expand All"):
            st.session_state.expand_all = True
            st.rerun()
    
    for session_id in df['session_id'].unique()[:10]:
        session_data = df[df['session_id'] == session_id]
        total_queries = len(session_data)
        display_data = session_data.head(100)
        session_start = session_data['timestamp'].max().strftime('%Y-%m-%d %H:%M')
        expanded = st.session_state.get('expand_all', False)
        with st.expander(f"Session {session_id[:8]}... - {session_start} ({total_queries} queries)", expanded=expanded):
            for _, row in display_data.iterrows():
                st.write(f"**{row['timestamp'].strftime('%H:%M')}**")
                with st.expander(f"Query: {row['query'][:50]}{'...' if len(row['query']) > 50 else ''}"):
                    st.write(row['query'])
                with st.expander(f"Response: {row['response'][:50]}{'...' if len(row['response']) > 50 else ''}"):
                    st.write(row['response'])
                st.write("---")
else:
    st.info("No data found")