import streamlit as st
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI  
import os
os.environ['GOOGLE_API_KEY']='AIzaSyAAnhyMX1Xt7QVYyV6nXfSmjCB2RM2WMDo'

# Page title
st.set_page_config(page_title="📊 Pandas Agent with CSV Upload", layout="wide")
st.title("📊 Pandas Agent with Gemini")
st.write("Upload your CSV file and ask questions in natural language.")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read CSV
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of your data:")
    st.dataframe(df.head())

    # Create Gemini LLM
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

    # Create Pandas Agent
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        allow_dangerous_code=True
    )

    # User input for queries
    user_query = st.text_input("💬 Ask a question about your data:")

    if user_query:
        with st.spinner("🤔 Thinking..."):
            response = agent.run(user_query)
        st.write("### 📌 Answer:")
        st.write(response)

else:
    st.info("Please upload a CSV file to begin.")
