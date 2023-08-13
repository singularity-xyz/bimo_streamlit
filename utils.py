import os
import sys
import logging
import dotenv
import PyPDF2
import base64
import streamlit as st
from io import BytesIO

dotenv.load_dotenv()

from llama_index import (
    ListIndex,
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
    LLMPredictor,
)
from llama_index.selectors.pydantic_selectors import (
    # PydanticMultiSelector,
    PydanticSingleSelector,
)
from llama_index.indices.postprocessor import (
    SentenceTransformerRerank,
    PrevNextNodePostprocessor,
)
from llama_index.tools.query_engine import QueryEngineTool
from llama_index.query_engine.router_query_engine import RouterQueryEngine
from llama_index.indices.query.query_transform.base import HyDEQueryTransform
from llama_index.query_engine.transform_query_engine import TransformQueryEngine
from langchain.chat_models import ChatOpenAI

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().handlers = [logging.StreamHandler(sys.stdout)]

LLM = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

class QueryEngineFactory:
    def __init__(self, course):
        self.course = course

    def create_index(self, index_type, folder_name):
        index_path = f"courses/{self.course}/indices/{folder_name}/"
        documents_path = f"courses/{self.course}/{folder_name}/"

        llm_predictor = LLMPredictor(llm=LLM)
        service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

        if os.path.exists(index_path):
            return load_index_from_storage(
                StorageContext.from_defaults(persist_dir=index_path),
                service_context=service_context,
            )

        documents = SimpleDirectoryReader(documents_path).load_data()
        for document in documents:
            document.metadata["path"] = documents_path

        if index_type == "list":
            index = ListIndex.from_documents(documents=documents, service_context=service_context)
            index.storage_context.persist(persist_dir=index_path)
        elif index_type == "vector":
            index = VectorStoreIndex.from_documents(documents=documents, service_context=service_context)
            index.storage_context.persist(persist_dir=index_path)
        else:
            raise ValueError("Invalid index type!")
        
        return index

    def create_query_engine(self, name):
        if name == "syllabus":
            index = self.create_index(index_type="list", folder_name=name)
            postprocessor = SentenceTransformerRerank(model="cross-encoder/ms-marco-MiniLM-L-2-v2", top_n=2)
            query_engine = index.as_query_engine(node_postprocessors=[postprocessor])
        elif name == "lectures":
            query_transform = HyDEQueryTransform()
            index = self.create_index(index_type="vector", folder_name=name)
            # postprocessor = PrevNextNodePostprocessor(docstore=index.docstore, num_nodes=1, mode="both")
            query_engine = TransformQueryEngine(
                query_engine=index.as_query_engine(),
                query_transform=query_transform,
            )
        elif name == "router":
            syllabus_tool = QueryEngineTool.from_defaults(
                query_engine=self.create_query_engine("syllabus"),
                description="Queries information from the course syllabus including: course information, learning objectives, required materials, assignments, grading, policies, dates, schedule, etc.",
            )
            lecture_tool = QueryEngineTool.from_defaults(
                query_engine=self.create_query_engine("lectures"),
                description="Queries information from the course lecture slides including: subject knowledge, topics, concepts, definitions, examples, etc.",
            )
            query_engine = RouterQueryEngine(
                selector=PydanticSingleSelector.from_defaults(),
                query_engine_tools=[syllabus_tool, lecture_tool],
            )
        else:
            raise ValueError("Invalid query engine name!")
        
        return query_engine


class PDFViewer:
    @staticmethod
    def display_sources(response):
        for source in response.source_nodes:
            pdf_file_html_link = PDFViewer._get_pdf_file_html_link(source)
            st.markdown(pdf_file_html_link, unsafe_allow_html=True)
            st.json(source.node.metadata)

    @staticmethod
    def _get_pdf_file_html_link(source):
        pdf_file = source.node.metadata["path"] + source.node.metadata["file_name"]
        pdf_page = int(source.node.metadata["page_label"]) - 1
        source.node.metadata["score"] = source.score

        with open(pdf_file, "rb") as file:
            pdf_bytes = BytesIO(file.read())
            base64_pdf = base64.b64encode(pdf_bytes.getvalue()).decode("utf-8")

        return f'<embed src="data:application/pdf;base64,{base64_pdf}#page={pdf_page+1}" width="700" height="500" type="application/pdf">'
