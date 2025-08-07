import streamlit as st
import boto3
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv
from collections import Counter

load_dotenv()

st.set_page_config(page_title="Chat Dashboard", layout="wide")
st.title("📊 Chat History Dashboard")

@st.cache_resource
def setup_dynamodb():
    return boto3.resource(
        'dynamodb',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        aws_session_token=os.getenv('AWS_SESSION_TOKEN'),
        region_name=os.getenv('AWS_DEFAULT_REGION')
    )

dynamodb = setup_dynamodb()
table = dynamodb.Table('chatbot_history')

@st.cache_data(ttl=60)
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
        st.metric("Error Rate", f"{(df['query_type'] == 'error').sum() / len(df) * 100:.1f}%")
    
    # Query times bar chart
    st.subheader("Query Times")
    df['hour'] = df['timestamp'].dt.hour
    hourly_counts = df['hour'].value_counts().reindex(range(24), fill_value=0)
    chart_data = pd.DataFrame({
        'Hour': [f"{h}:00" for h in range(24)],
        'Queries': hourly_counts.values
    }).set_index('Hour')
    st.bar_chart(chart_data)
    
    # Popular queries
    st.subheader("Popular Queries")
    query_counts = Counter(df['query'].tolist())
    top_3 = query_counts.most_common(3)
    
    for i, (query, count) in enumerate(top_3, 1):
        st.write(f"**{i}.** {query} ({count} times)")
    
    # Remaining queries dropdown
    remaining_queries = [q for q, _ in query_counts.most_common()[3:]]
    if remaining_queries:
        selected_query = st.selectbox("Other Queries", ["Select a query..."] + remaining_queries)
        if selected_query != "Select a query...":
            count = query_counts[selected_query]
            st.write(f"**Query:** {selected_query} ({count} times)")
    
    st.subheader("Recent Chats")
    for _, row in df.head(10).iterrows():
        with st.expander(f"{row['timestamp'].strftime('%Y-%m-%d %H:%M')} - {row['query_type']}"):
            st.write("**Query:**", row['query'])
            st.write("**Response:**", row['response'])
            st.write("**Session:**", row['session_id'])
else:
    st.info("No data found")