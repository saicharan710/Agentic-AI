import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA

from rag import vector_store, ingest_data, rag_chain

st.set_page_config(page_title="RAG Dashboard", page_icon="📚")

st.title("📚 Resume RAG Dashboard")

st.subheader("Vector Database")

# Display database count
if st.button("Refresh Database"):
    st.rerun()

try:
    data = vector_store.get(include=["documents", "metadatas", "embeddings"])

    ids = data.get("ids", [])
    documents = data.get("documents", []) or []
    metadatas = data.get("metadatas", []) or []
    embeddings = data.get("embeddings")

    st.metric("Total Stored Chunks", len(ids))

    if ids:
        for i, doc in enumerate(documents):
            with st.expander(f"Chunk {i + 1}"):
                st.write("**Chunk ID:**", ids[i])
                st.write("**Content:**")
                st.write(doc)
                st.write("**Metadata:**")
                st.json(metadatas[i] or {})

        st.subheader("3D Embedding Visualization")
        if embeddings is None:
            st.info("No embeddings were returned for this collection.")
        else:
            embedding_matrix = np.asarray(embeddings, dtype=float)
            if (
                embedding_matrix.ndim != 2
                or embedding_matrix.shape[0] != len(ids)
                or embedding_matrix.shape[1] == 0
            ):
                st.error("The database returned embeddings in an unexpected format.")
            else:
                component_count = min(3, *embedding_matrix.shape)
                if len(ids) == 1:
                    coordinates = np.zeros((1, component_count))
                    st.caption("Only one vector is stored, so it is shown at the origin.")
                else:
                    coordinates = PCA(n_components=component_count).fit_transform(
                        embedding_matrix
                    )

                if component_count < 3:
                    coordinates = np.pad(
                        coordinates,
                        ((0, 0), (0, 3 - component_count)),
                    )

                custom_data = [
                    [chunk_id, document or ""]
                    for chunk_id, document in zip(ids, documents)
                ]
                figure = go.Figure(
                    data=[
                        go.Scatter3d(
                            x=coordinates[:, 0],
                            y=coordinates[:, 1],
                            z=coordinates[:, 2],
                            mode="markers",
                            marker={
                                "size": 7,
                                "color": list(range(len(ids))),
                                "colorscale": "Viridis",
                                "colorbar": {"title": "Chunk"},
                            },
                            customdata=custom_data,
                            hovertemplate=(
                                "Chunk ID: %{customdata[0]}<br>"
                                "Content: %{customdata[1]}<extra></extra>"
                            ),
                        )
                    ]
                )
                figure.update_layout(
                    scene={
                        "xaxis_title": "Principal component 1",
                        "yaxis_title": "Principal component 2",
                        "zaxis_title": "Principal component 3",
                    },
                    margin={"l": 0, "r": 0, "t": 20, "b": 0},
                    height=600,
                )
                st.plotly_chart(figure, use_container_width=True)
                st.caption(
                    "Each point is a stored chunk. PCA projects the embeddings "
                    "into 3D; nearby points have similar vector representations."
                )
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