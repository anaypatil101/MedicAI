import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.hub import pull as hub_pull
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

DB_FAISS_PATH = "vectorstore/db_faiss"
GROQ_MODEL_NAME = "llama-3.1-8b-instant"


@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


@st.cache_resource
def get_rag_chain():
    vectorstore = get_vectorstore()
    if vectorstore is None:
        return None

    llm = ChatGroq(
        model=GROQ_MODEL_NAME,
        temperature=0.5,
        max_tokens=512,
        api_key=os.environ.get("GROQ_API_KEY"),
    )

    retrieval_qa_chat_prompt = hub_pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    return create_retrieval_chain(vectorstore.as_retriever(search_kwargs={'k': 3}), combine_docs_chain)


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
            rag_chain = get_rag_chain()
            if rag_chain is None:
                st.error("Failed to load the vector store. Run create_memory_for_llm.py first.")
            else:
                response = rag_chain.invoke({'input': prompt})
                result = response["answer"]
                source_documents = response["context"]

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