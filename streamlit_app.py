
import streamlit as st
import json
import traceback

from graph_rag import (
    driver,
    DATABASE,
    generate_answer
)

from retriever import retrieve_context


# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="GraphRAG | Startup Intelligence",
    page_icon="🧠",
    layout="wide"
)

# -----------------------------
# Custom styling
# -----------------------------

st.markdown("""
<style>
    .main {
        background-color: #0b0f1a;
    }

    .block-container {
        padding-top: 2rem;
        max-width: 1100px;
    }

    .project-subtitle {
        color: #9ca3af;
        font-size: 16px;
    }

    .project-title {
        color: #4fd1c5;
        font-size: 36px;
        font-weight: 700;
    }

    div[data-testid="stChatMessage"] {
        border-radius: 12px;
        padding: 12px;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:

    st.markdown("## 🧠 GraphRAG")

    st.caption("Startup Intelligence Platform")

    st.divider()

    st.markdown("### Architecture")

    st.markdown("""
    - Python
    - Neo4j Graph Database
    - Sentence Transformers
    - MiniLM Embeddings
    - Vector Search
    - Gemini API
    - Streamlit
    """)

    st.divider()

    st.markdown("### Example Questions")

    examples = [
        "Who are all the founders?",
        "Which investors funded NovaPay?",
        "Which companies operate in Fintech?",
        "Which companies were acquired?",
        "Tell me about AI healthcare companies.",
        "Who founded more than one company?"
    ]

    selected_example = st.selectbox(
        "Choose a question",
        ["Select a question..."] + examples
    )

    if st.button("Use selected question", use_container_width=True):
        if selected_example != "Select a question...":
            st.session_state.pending_question = selected_example

    st.divider()

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# -----------------------------
# Main page
# -----------------------------

st.markdown(
    '<div class="project-title">GraphRAG Startup Intelligence</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="project-subtitle">'
    'Explore startup companies, founders, investors, products, '
    'and acquisitions using Graph Retrieval-Augmented Generation.'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# -----------------------------
# Database statistics
# -----------------------------

col1, col2 = st.columns(2)

try:
    with driver.session(database=DATABASE) as session:

        node_count = session.run(
            "MATCH (n) RETURN count(n) AS total"
        ).single()["total"]

        relationship_count = session.run(
            "MATCH ()-[r]->() RETURN count(r) AS total"
        ).single()["total"]

    col1.metric("Total Graph Nodes", node_count)
    col2.metric("Total Relationships", relationship_count)

except Exception:
    col1.metric("Graph Database", "Unavailable")
    col2.metric("Connection", "Check Neo4j")


st.divider()


# -----------------------------
# Chat history
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("results") is not None:
            with st.expander("View retrieved graph information"):
                st.json(message["results"])


# -----------------------------
# User input
# -----------------------------

prompt = st.chat_input(
    "Ask something about companies, founders, or investors..."
)

if st.session_state.pending_question:
    prompt = st.session_state.pending_question
    st.session_state.pending_question = None


# -----------------------------
# Process question
# -----------------------------

if prompt:

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        try:
            with st.spinner("Retrieving graph information..."):

                retrieval_results = retrieve_context(
                    question=prompt,
                    driver=driver,
                    database=DATABASE,
                    top_k=3
                )

            with st.spinner("Gemini is generating your answer..."):

                answer = generate_answer(
                    prompt,
                    retrieval_results
                )

            st.markdown(answer)

            with st.expander("View retrieved graph information"):
                st.json(retrieval_results["results"])

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "results": retrieval_results["results"]
            })

        except Exception as e:

            st.error(f"An error occurred: {e}")
            st.code(traceback.format_exc())
