import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

DB_FAISS_PATH = "vectorstore/db_faiss"
GROQ_MODEL_NAME = "llama-3.1-8b-instant"


@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


@st.cache_resource
def get_llm():
    return ChatGroq(
        model=GROQ_MODEL_NAME,
        temperature=0.5,
        max_tokens=512,
        api_key=os.environ.get("GROQ_API_KEY"),
    )


PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful medical assistant. Answer the user's question "
     "based only on the following context:\n\n{context}"),
    ("human", "{input}"),
])


def ask(query):
    vectorstore = get_vectorstore()
    llm = get_llm()
    docs = vectorstore.as_retriever(search_kwargs={'k': 3}).invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)
    response = llm.invoke(PROMPT.format_messages(context=context, input=query))
    return response.content, docs


def main():
    st.title("Ask MedicAI!")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Ask a medical question...")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        try:
            result, source_documents = ask(prompt)

            st.chat_message('assistant').markdown(result)
            st.session_state.messages.append({'role': 'assistant', 'content': result})

            with st.sidebar:
                st.subheader("Sources")
                for i, doc in enumerate(source_documents, 1):
                    with st.expander(f"Source {i} — page {doc.metadata.get('page', '?')}"):
                        st.caption(doc.page_content[:300] + "...")

        except Exception as e:
            st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
