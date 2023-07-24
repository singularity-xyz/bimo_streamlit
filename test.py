import sys
import dotenv
import logging
import pandas as pd

dotenv.load_dotenv()
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index.llms import OpenAI
from llama_index.playground import Playground
from llama_index.evaluation import DatasetGenerator, QueryResponseEvaluator
from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    ListIndex,
    TreeIndex,
    KeywordTableIndex,
    KnowledgeGraphIndex,
    ServiceContext,
    LLMPredictor,
    Response,
)

reader = SimpleDirectoryReader("documents/lectures")
documents = reader.load_data()

playground = Playground.from_docs(documents)
results_df = playground.compare("What does the I=PAT equation describe?")

print(results_df)

# data_generator = DatasetGenerator.from_documents(documents)

# eval_questions = data_generator.generate_questions_from_nodes()
# print(eval_questions)

# gpt4 = OpenAI(temperature=0, model="gpt-4")
# service_context_gpt4 = ServiceContext.from_defaults(llm=gpt4)

# evaluator_gpt4 = QueryResponseEvaluator(service_context=service_context_gpt4)

# vector_index = VectorStoreIndex.from_documents(
#     documents, service_context=service_context_gpt4
# )

# query_engine = vector_index.as_query_engine()
# response_vector = query_engine.query(eval_questions[1])
# eval_result = evaluator_gpt4.evaluate(eval_questions[1], response_vector)
# print(eval_result)