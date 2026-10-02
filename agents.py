from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from tools import web_search, web_scrape

import streamlit as st


# =========================================================
# GROQ API KEY
# =========================================================

try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    st.error(
        "GROQ_API_KEY is missing. "
        "Please add it in Streamlit Cloud → Settings → Secrets."
    )
    st.stop()


# =========================================================
# MODEL SETUP
# =========================================================

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=GROQ_API_KEY
)


# =========================================================
# 1st AGENT: SEARCH AGENT
# =========================================================

def build_search_agent():

    return create_agent(
        model=llm,
        tools=[web_search]
    )


# =========================================================
# 2nd AGENT: READER AGENT
# =========================================================

def build_reader_agent():

    return create_agent(
        model=llm,
        tools=[web_scrape]
    )


# =========================================================
# WRITER CHAIN
# =========================================================

writer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert research writer. "
        "Write clear, structured, insightful, factual "
        "and professional reports."
    ),

    (
        "human",
        """
Write a detailed research report on the topic below.

Topic:
{topic}

Research Gathered:
{research}

Structure the report as:

# Introduction

Explain the topic briefly.

# Key Findings

Give at least 3 important and well-explained findings.

# Conclusion

Give a concise conclusion based only on the research.

# Sources

List the URLs found in the research.

Rules:

- Be factual.
- Do not invent information.
- Do not repeat the same point.
- Use clear headings.
- Keep the report professional.
"""
    ),
])


writer_chain = (
    writer_prompt
    | llm
    | StrOutputParser()
)


# =========================================================
# CRITIC CHAIN
# =========================================================

critic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a sharp and constructive research critic. "
        "Be honest, specific and helpful."
    ),

    (
        "human",
        """
Review the research report below.

Report:
{report}

Respond in exactly this format:

Score: X/10

Strengths:

- ...
- ...

Areas to Improve:

- ...
- ...

One line verdict:

...
"""
    ),
])


critic_chain = (
    critic_prompt
    | llm
    | StrOutputParser()
)