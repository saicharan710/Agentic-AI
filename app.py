import streamlit as st
from rag import vector_store, ingest_data, rag_chain

st.set_page_config(page_title="RAG Dashboard", page_icon="📚")

st.title("📚 Resume RAG Dashboard")

st.subheader("Vector Database")

# Display database count
if st.button("Refresh Database"):
    st.rerun()

try:
    data = vector_store.get(include=["documents", "metadatas"])

    ids = data.get("ids", [])
    documents = data.get("documents", []) or []
    metadatas = data.get("metadatas", []) or []

    st.metric("Total Stored Chunks", len(ids))

    if ids:
        for i, doc in enumerate(documents):
            with st.expander(f"Chunk {i + 1}"):
                st.write("**Chunk ID:**", ids[i])
                st.write("**Content:**")
                st.write(doc)
                st.write("**Metadata:**")
                st.json(metadatas[i] or {})
    else:
        st.info("No documents found. Please ingest your resume.")

except Exception as e:
    st.error(f"Database error: {e}")

st.divider()

# Ingest PDF
st.subheader("Upload / Ingest Resume")

if st.button("Ingest Resume"):
    with st.spinner("Processing resume..."):
        try:
            result = ingest_data()
            st.success(result)
            st.rerun()
        except Exception as e:
            st.error(str(e))

st.divider()

# Ask questions
st.subheader("Ask Your Resume")

query = st.text_input("Enter your question")

if st.button("Get Answer"):
    if query:
        with st.spinner("Generating answer..."):
            try:
                response = rag_chain(query)
                st.success(response["result"])
            except Exception as e:
                st.error(str(e))
    else:
        st.warning("Please enter a question.")