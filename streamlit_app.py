import streamlit as st
from rag_backend import (
    retrieve_multi_modal,
    prepare_context_for_model,
    answer_with_multimodal_rag,
)

st.set_page_config(page_title="Multi-Modal RAG QA", layout="wide")
st.title(" Multi-Modal RAG for Qatar Staff Report")
st.caption("Ask questions grounded in the IMF Article IV Qatar document.")


query = st.text_area("Enter your question", height=120)

col1, col2 = st.columns([1, 2], vertical_alignment="top")

with col1:
    st.header(" Retrieval Settings")
    k_faiss = st.slider("FAISS neighbors", 4, 40, 12)
    k_colpali = st.slider("ColPali neighbors", 0, 10, 3)
    max_text = st.slider("Max text snippets", 1, 12, 6)
    max_imgs = st.slider("Max image snippets", 0, 4, 1)
    max_tokens = st.number_input(
        "Max generation tokens", 50, 2048, 256, step=64
    )

    run = st.button("Run RAG")

with col2:
    answer_box = st.empty()
    ctx_box = st.empty()
    img_box = st.empty()


if run:
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner(" Retrieving context..."):
            retrieved = retrieve_multi_modal(
                query, k_faiss=k_faiss, k_colpali=k_colpali
            )
            text_snips, img_snips, used_meta = prepare_context_for_model(
                retrieved, max_text=max_text, max_img=max_imgs
            )

        ctx_box.subheader(" Retrieved Text Context")
        if text_snips:
            for t in text_snips:
                ctx_box.markdown(f"- {t}")
        else:
            ctx_box.info("No text snippets used.")

        img_box.subheader(" Retrieved Page Images")
        if img_snips:
            for im in img_snips:
                img_box.image(im, use_column_width=True)
        else:
            img_box.info("No images used.")

        with st.spinner(" Generating answer..."):
            answer, meta_used = answer_with_multimodal_rag(
                query,
                top_k_faiss=k_faiss,
                top_k_colpali=k_colpali,
                max_text_snips=max_text,
                max_img_snips=max_imgs,
                max_new_tokens=max_tokens,
            )

        answer_box.subheader(" Answer")
        answer_box.write(answer)
