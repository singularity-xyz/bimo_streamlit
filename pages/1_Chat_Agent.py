import streamlit as st

from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="langchain_search_api_key_openai", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/pages/2_Chat_with_search.py)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

st.title("🔎 LangChain - Chat with search")

"""
In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
Try more LangChain 🤝 Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
"""

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="Who won the Women's U.S. Open in 2018?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, streaming=True)
    search = DuckDuckGoSearchRun(name="Search")
    search_agent = initialize_agent([search], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True)
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)

# syllabus_tool_config = IndexToolConfig(
#     query_engine=syllabus_query_engine,
#     name="Syllabus",
#     description="Queries information from the course syllabus, including course description, learning objectives, required materials, assignments, grading, policies, etc.",
#     tool_kwargs={"return_direct": True},
# )
# syllabus_tool = LlamaIndexTool.from_tool_config(syllabus_tool_config)

# lecture_tool_config = IndexToolConfig(
#     query_engine=lecture_query_engine,
#     name="Lectures",
#     description="Queries information from the course lecture slides and materials. Useful for finding information about specific topics, conecepts, examples, terms, etc. covered in the course.",
#     tool_kwargs={"return_direct": True},
# )
# lecture_tool = LlamaIndexTool.from_tool_config(lecture_tool_config)

# tools = [syllabus_tool, lecture_tool]
# # agent = create_conversational_retrieval_agent(LLM, tools, verbose=True)
# agent = initialize_agent(tools, LLM, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)

# if prompt := st.chat_input():
#     st.chat_message("user").write(prompt)
#     with st.chat_message("assistant"):
#         st_callback = StreamlitCallbackHandler(st.container())
#         response = agent.run(prompt, callbacks=[st_callback])
#         st.write(response)