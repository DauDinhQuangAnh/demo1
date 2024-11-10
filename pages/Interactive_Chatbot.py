import streamlit as st
from sentence_transformers import SentenceTransformer
import pandas as pd
from search import vector_search, keywords_search, hyde_search
from constant import USER, ASSISTANT, ONLINE_LLM
import chromadb
import json

st.markdown(
    """
    <h1 style='display: flex; align-items: center;'>
        <img src="https://tuyensinh.uit.edu.vn/sites/default/files/uploads/images/uit_footer.png" width="50" style='margin-right: 10px'>
        UIT Admissions Chatbot 🎓
    </h1>
    """,
    unsafe_allow_html=True
)
st.markdown("Welcome to the UIT Admissions Chatbot!❓❓❓ Discover all the information you need about admissions, 📚programs, 💸scholarships, 🌟Student Life at UIT and more with us.")


def load_session_state(file_path="session_state.json"):
    try:
        with open(file_path, "r") as file:
            session_data = json.load(file)
        for key, value in session_data.items():
            st.session_state[key] = value
        # st.success("Session state loaded successfully!")
    except FileNotFoundError:
        st.error("Session state file not found.")
    except json.JSONDecodeError:
        st.error("Error decoding session state file.")
if "search_option" not in st.session_state:
    st.session_state.search_option = "Vector Search"
# Display the collection name
if "collection" not in st.session_state:
    load_session_state(file_path="pages/session_state.json")

if "client" not in st.session_state:
    st.session_state.client = chromadb.PersistentClient("db")
    if "random_collection_name" in st.session_state:
        st.session_state.collection = st.session_state.client.get_collection(
            st.session_state.random_collection_name
        )

if "embedding_model_name" in st.session_state and "embedding_model" not in st.session_state:
    st.session_state.embedding_model = SentenceTransformer(
        st.session_state.embedding_model_name
    )
if "columns_to_answer" not in st.session_state:
    st.session_state.columns_to_answer = ["name"]

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display the chat history using chat UI
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.chat_history.append({"role": USER, "content": prompt})

    # Display user message in chat message container
    with st.chat_message(USER):
        st.markdown(prompt)

    # Process and enhance the prompt
    if st.session_state.collection is not None:
        metadatas, retrieved_data = [], ""
        if "columns_to_answer" in st.session_state:
            columns_to_answer = st.session_state.columns_to_answer
            search_option = st.session_state.search_option

            if search_option == "Vector Search":
                metadatas, retrieved_data = vector_search(
                    st.session_state.embedding_model,
                    prompt,
                    st.session_state.collection,
                    columns_to_answer,
                    st.session_state.number_docs_retrieval
                )

            elif search_option == "Keywords Search":
                metadatas, retrieved_data = keywords_search(
                    prompt,
                    st.session_state.collection,
                    columns_to_answer,
                    st.session_state.number_docs_retrieval
                )

            elif search_option == "Hyde Search":
                model = st.session_state.llm_model
                metadatas, retrieved_data = hyde_search(
                    model,
                    st.session_state.embedding_model,
                    prompt,
                    st.session_state.collection,
                    columns_to_answer,
                    st.session_state.number_docs_retrieval,
                    num_samples=1
                )

            # Enhance the prompt and generate response
            if metadatas:
                enhanced_prompt = f'The prompt of the user is: "{prompt}". Answer it based on the following retrieved data:\n{retrieved_data}'
                response = st.session_state.llm_model.generate_content(enhanced_prompt)

                # Update chat history
                st.markdown(response)
                st.session_state.chat_history.append({"role": ASSISTANT, "content": response})
            else:
                st.warning("No data to enhance the prompt.")
        else:
            st.warning("Select columns to answer from.")
    else:
        st.error("No collection found. Upload data and save it first.")
