import os
import dotenv
import streamlit as st
dotenv.load_dotenv()
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
    StorageContext,
    LLMPredictor,
    load_index_from_storage,
)
from langchain.chat_models import ChatOpenAI

index_name = "./saved_index"
documents_folder = "./documents"


@st.cache_resource
def initialize_index(index_name, documents_folder):
    llm_predictor = LLMPredictor(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    )
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
    if os.path.exists(index_name):
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=index_name),
            service_context=service_context,
        )
    else:
        documents = SimpleDirectoryReader(documents_folder).load_data()
        index = VectorStoreIndex.from_documents(
            documents, service_context=service_context
        )
        index.storage_context.persist(persist_dir=index_name)

    return index


@st.cache_data(max_entries=200, persist=True)
def query_index(_index, query_text):
    if _index is None:
        return "Please initialize the index!"
    response = _index.as_query_engine().query(query_text)
    return str(response)


st.title("🦙 Bimo Demo 🦙")
st.subheader("Enter a query about ENST 100")

index = initialize_index(index_name, documents_folder)

text = st.text_input("Query:", value="What is ecosystem and ecosystem ecology?")

if st.button("Run Query") and text is not None:
    response = query_index(index, text)
    st.markdown(response)

    # llm_col, embed_col = st.columns(2)
    # with llm_col:
    #     st.markdown(
    #         f"LLM Tokens Used: {index.service_context.llm_predictor._last_token_usage}"
    #     )

    # with embed_col:
    #     st.markdown(
    #         f"Embedding Tokens Used: {index.service_context.embed_model._last_token_usage}"
    #     )
