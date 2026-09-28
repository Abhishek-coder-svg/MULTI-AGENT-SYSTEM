import time
import traceback

import streamlit as st
from dotenv import load_dotenv

from agents import (
    build_search_agent,
    build_reader_agent,
    writer_chain,
    critic_chain,
)

load_dotenv()

st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    page_icon="🔎",
    layout="wide",
)


# -----------------------------------
# Helper function
# -----------------------------------

def to_text(content):
    if hasattr(content, "content"):
        content = content.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "\n".join(
            block.get("text", str(block))
            if isinstance(block, dict)
            else str(block)
            for block in content
        )

    return str(content)


# -----------------------------------
# Retry function for Groq 429 errors
# -----------------------------------

def retry_call(function, max_retries=3):
    for attempt in range(max_retries):

        try:
            return function()

        except Exception as e:

            error_text = str(e)

            if "429" in error_text or "rate_limit" in error_text:
                wait_time = 5 * (attempt + 1)

                st.warning(
                    f"Groq rate limit reached. "
                    f"Waiting {wait_time} seconds..."
                )

                time.sleep(wait_time)

            else:
                raise e

    raise Exception(
        "Groq rate limit continued after multiple retries. "
        "Please wait for a minute and try again."
    )


# -----------------------------------
# Load agents only once
# -----------------------------------

@st.cache_resource
def get_agents():

    search_agent = build_search_agent()
    reader_agent = build_reader_agent()

    return search_agent, reader_agent


# -----------------------------------
# Main pipeline
# -----------------------------------

def run_pipeline(topic, status):

    search_agent, reader_agent = get_agents()

    state = {}

    # =================================
    # STEP 1 - SEARCH
    # =================================

    status.update(
        label="🔎 Step 1/4 - Searching...",
        state="running"
    )

    search_result = retry_call(
        lambda: search_agent.invoke({
            "messages": [
                (
                    "user",
                    f"""
Find recent and reliable information about:

{topic}

Use the web search tool.
Return only the most important information.
"""
                )
            ]
        })
    )

    state["search_results"] = to_text(
        search_result["messages"][-1].content
    )

    # Keep only limited text
    state["search_results"] = state["search_results"][:1000]


    # =================================
    # STEP 2 - READER
    # =================================

    status.update(
        label="📄 Step 2/4 - Reading...",
        state="running"
    )

    reader_result = retry_call(
        lambda: reader_agent.invoke({
            "messages": [
                (
                    "user",
                    f"""
From the search results below, select ONE
direct and reliable URL.

Then use the web scraping tool to read it.

Search results:

{state["search_results"]}

Return only the important extracted information.
"""
                )
            ]
        })
    )

    state["scraped_content"] = to_text(
        reader_result["messages"][-1].content
    )

    # Keep scraped information small
    state["scraped_content"] = (
        state["scraped_content"][:1800]
    )


    # =================================
    # STEP 3 - WRITER
    # =================================

    status.update(
        label="✍️ Step 3/4 - Writing report...",
        state="running"
    )

    research = (
        f"SEARCH RESULTS:\n"
        f"{state['search_results']}\n\n"
        f"SCRAPED CONTENT:\n"
        f"{state['scraped_content']}"
    )

    state["report"] = retry_call(
        lambda: writer_chain.invoke({
            "topic": topic,
            "research": research
        })
    )

    state["report"] = to_text(
        state["report"]
    )


    # =================================
    # STEP 4 - CRITIC
    # =================================

    status.update(
        label="🧐 Step 4/4 - Reviewing...",
        state="running"
    )

    state["feedback"] = retry_call(
        lambda: critic_chain.invoke({
            "report": state["report"][:3000]
        })
    )

    state["feedback"] = to_text(
        state["feedback"]
    )


    return state


# =====================================
# STREAMLIT UI
# =====================================

st.title("🔎 Multi-Agent Research Assistant")

st.caption(
    "Search → Read → Write → Critic"
)


with st.form("research_form"):

    topic = st.text_input(
        "Research Topic",
        placeholder="e.g. Latest advances in AI"
    )

    submit = st.form_submit_button(
        "🚀 Run Research",
        type="primary"
    )


# =====================================
# RUN PIPELINE
# =====================================

if submit:

    topic = topic.strip()

    if not topic:

        st.warning(
            "Please enter a research topic."
        )

    else:

        with st.status(
            "Starting research...",
            expanded=True
        ) as status:

            try:

                result = run_pipeline(
                    topic,
                    status
                )

                status.update(
                    label="✅ Research completed",
                    state="complete"
                )

                st.session_state.result = result
                st.session_state.topic = topic

            except Exception as e:

                status.update(
                    label="❌ Pipeline failed",
                    state="error"
                )

                st.error(str(e))

                with st.expander(
                    "Error details"
                ):
                    st.code(
                        traceback.format_exc()
                    )


# =====================================
# DISPLAY RESULTS
# =====================================

if "result" in st.session_state:

    result = st.session_state.result

    st.divider()

    st.subheader(
        f"Results: {st.session_state.topic}"
    )

    report, critic, search, scrape = st.tabs([
        "📝 Report",
        "🧐 Critic",
        "🔎 Search",
        "📄 Scraped Data"
    ])


    with report:

        st.markdown(
            result["report"]
        )


    with critic:

        st.markdown(
            result["feedback"]
        )


    with search:

        st.markdown(
            result["search_results"]
        )


    with scrape:

        st.markdown(
            result["scraped_content"]
        )


    st.download_button(
        "⬇️ Download Report",
        data=result["report"],
        file_name="research_report.md",
        mime="text/markdown",
    )
