# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 11:50:36 2025

@author: 丫丫
"""


import streamlit as st
import pandas as pd
import json
import aiohttp
import asyncio
import index

def read_token_json(file_path="token.json"):
    try:
        with open(file_path, "r") as f:
            config = json.load(f)
        return config["siliconflow_api_token"]
    except FileNotFoundError:
        st.error(f"Key file '{file_path}' not found, please ensure the file exists.")
        return None
    except KeyError:
        st.error("The 'API_TOKEN' key is missing in secrets.json.")
        return None

# Asynchronous API Request
async def make_request(messages, base_url="https://api.siliconflow.cn/v1/chat/completions", api_token=None, max_retries=3):
    if not api_token:
        raise ValueError("API key is required.")
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-ai/DeepSeek-V3",
        "messages": messages,
        "stream": False,
        "max_tokens": 1000,
        "temperature": 0.1,
        "top_p": 0.7,
        "response_format": {"type": "text"}
    }
    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(base_url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("choices", [{}])[0].get("message", {}).get("content", "")
                    else:
                        st.error(f"Attempt {attempt + 1} failed with status code {response.status}.")
        except Exception as e:
            st.error(f"Attempt {attempt + 1} failed with error: {e}")
        if attempt < max_retries - 1:
            await asyncio.sleep(1)
    return None

# AI Interaction Function
async def chat_with_AI(analysis_results, user_prompt, api_token):
    if isinstance(analysis_results, pd.DataFrame):
        context = "Data:\n" + str(analysis_results)
    else:
        context = str(analysis_results)
    
    SYSTEM_PROMPT = (
        "You are a financial assistant skilled in analyzing portfolio and benchmark performance metrics. "
        "Based on the provided data, follow this format:\n"
        "Title: Return\n"
        "A concise and professional paragraph describing the portfolio's return performance, e.g., 'the portfolio generated a return of X% per year.'\n\n"
        "Title: Risk\n"
        "A concise and professional paragraph describing the portfolio's risk performance, e.g., 'the maximum drawdown of the portfolio was X%.'\n"
        "Do not use bullet points, and keep the two paragraphs of similar length."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{context}\nUser question: {user_prompt}"}
    ]
    response = await make_request(messages, api_token=api_token)
    return response if response else "Sorry, unable to retrieve AI response, please try again later."

# Streamlit Application
async def main():
    # Load API key
    API_TOKEN = read_token_json(r"C:\Streamlit\token.json")
    st.write(API_TOKEN)
    # AI Summary Section (only runs if API_TOKEN is available)
    if API_TOKEN:
        analysis_results_matrics = index.performance_metrics()
        st.subheader("AI Portfolio Performance Summary")
        with st.spinner("Generating AI summary..."):
            summary_prompt = "Please analyze the performance metrics of the portfolio and benchmark, summarize their performance, and highlight key differences and strengths/weaknesses."
            summary = await chat_with_AI(analysis_results_matrics, summary_prompt, API_TOKEN)
            st.write(summary)
    else:
        st.info("AI functionality skipped due to missing API key.")

# Investment Chatbot Function
async def investment_chatbot(user_input, api_token, conversation_history=[]):
    SYSTEM_PROMPT = (
        "You are a financial assistant skilled in answering investment-related questions. "
        "You can explain charts and data, provide investment advice, and respond to user inquiries. "
        "Please respond in concise and professional language to ensure user understanding."
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history + [{"role": "user", "content": user_input}]
    response = await make_request(messages, api_token=api_token)
    if response:
        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": response})
    return response

# Display AI Portfolio Performance Summary
async def display_ai_summary(portfolio_returns, benchmark_returns, API_TOKEN):
    if API_TOKEN:
        analysis_results_matrics = index.performance_metrics(portfolio_returns, benchmark_returns)
        st.subheader("AI Portfolio Performance Summary")
        with st.spinner("Generating AI summary..."):
            summary_prompt = "Please analyze the performance metrics of the portfolio and benchmark, summarize their performance, and highlight key differences and strengths/weaknesses."
            summary = await chat_with_AI(analysis_results_matrics, summary_prompt, API_TOKEN)
            st.write(summary)
    else:
        st.info("AI functionality skipped due to missing API key.")

# Display Chatbot
async def display_chatbot(API_TOKEN):
    st.title("Investment Chatbot")

    if not API_TOKEN:
        st.warning("Missing API key, unable to run the chatbot functionality.")
        return

    # Initialize conversation history
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    # Display conversation history
    for message in st.session_state.conversation_history:
        if message["role"] == "user":
            st.write(f"**User**: {message['content']}")
        else:
            st.write(f"**AI**: {message['content']}")

    # User input box
    user_input = st.text_input("Please enter your investment question:", key="user_input")

    # Send button
    if st.button("Send"):
        if user_input:
            with st.spinner("Generating response..."):
                response = await investment_chatbot(user_input, API_TOKEN, st.session_state.conversation_history)
                if response:
                    st.write(f"**AI**: {response}")
                else:
                    st.error("Unable to retrieve AI response, please try again later.")
        else:
            st.warning("Please enter a question.")