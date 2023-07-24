import io, os
import base64
import dotenv
import PyPDF2
import streamlit as st
dotenv.load_dotenv()
from llama_index import (
    ListIndex,
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
    StorageContext,
    LLMPredictor,
    load_index_from_storage,
    ComposableGraph,
)
from langchain.chat_models import ChatOpenAI


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


# @st.cache_data(max_entries=100, persist=True)
# def query_index(_index, query_text):
#     if _index is None:
#         return "Please initialize the index!"
#     response = _index.as_query_engine().query(query_text)
#     return response


def display_sources(response):
    for source in response.source_nodes:
        # Get PDF file name and page number from metadata
        if 'file_name' in source.node.metadata and 'page_label' in source.node.metadata:
            pdf_file = ""
            if "ENST 100" in source.node.metadata['file_name']:
                pdf_file = "documents/lectures/" + source.node.metadata['file_name']
            elif "ENST 100" not in source.node.metadata['file_name']:
                pdf_file = "documents/textbooks/" + source.node.metadata['file_name']
            pdf_page = int(source.node.metadata['page_label']) - 1  # PyPDF2 uses 0-based indexing

            # Open the PDF 
            with open(pdf_file, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                pdf_writer = PyPDF2.PdfWriter()
                pdf_writer.add_page(pdf_reader.pages[pdf_page])
                
                pdf_bytes = io.BytesIO()
                pdf_writer.write(pdf_bytes)
                pdf_bytes = pdf_bytes.getvalue()

            # Encode the PDF bytes to base64 and create HTML link
            base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
            pdf_file_html_link = f'<embed src="data:application/pdf;base64,{base64_pdf}#page={pdf_page}" width="700" height="500" type="application/pdf">'

            # Display the PDF page in Streamlit
            st.markdown(pdf_file_html_link, unsafe_allow_html=True)
            st.markdown("")
            st.markdown(source.node.metadata)
            st.markdown(source.score)
            st.markdown(source.node.text)

    # st.markdown(response.get_formatted_sources())


st.title("🦙 Bimo Demo 🦙")
st.subheader("Ask a question about ENST 100")

lecture_index = initialize_index("indices/lectures", "documents/lectures")
textbook_index = initialize_index("indices/textbooks", "documents/textbooks")
index_summaries = ["Lecture notes about environmental science", "Textbook about environmental science"]

graph = ComposableGraph.from_indices(ListIndex, [lecture_index, textbook_index], index_summaries)
query_engine = graph.as_query_engine()

text = st.text_input("Query:", value="What is ecosystem and ecosystem ecology?")

if st.button("Run Query") and text is not None:
    # response = query_index(index, text)
    response = query_engine.query(text)

    st.divider()
    st.caption("Response:")
    st.markdown(response)

    st.divider()
    st.caption("Sources:")
    display_sources(response)

    # llm_col, embed_col = st.columns(2)
    # with llm_col:
    #     st.markdown(
    #         f"LLM Tokens Used: {index.service_context.llm_predictor._last_token_usage}"
    #     )

    # with embed_col:
    #     st.markdown(
    #         f"Embedding Tokens Used: {index.service_context.embed_model._last_token_usage}"
    #     )
