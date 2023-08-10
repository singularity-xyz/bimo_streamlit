import streamlit as st
from utils import QueryEngineFactory, PDFViewer

COURSE = "CSCI-360"

query_engine_factory = QueryEngineFactory(COURSE)
router_query_engine = query_engine_factory.create_query_engine("router")

st.title("Query Engine Demo 🔍")
text = st.text_input(f"Ask me anything about {COURSE}!")

if st.button("Run Query") and text is not None:
    response = router_query_engine.query(text)

    st.divider()
    st.caption("Response:")
    st.markdown(response)

    st.divider()
    st.caption("Sources:")
    PDFViewer.display_sources(response)
