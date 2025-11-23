import os
import math
import pickle
from typing import List, Dict, Any

import numpy as np
import torch
import faiss
from PIL import Image
from pdf2image import convert_from_path

from transformers import (
    AutoTokenizer,
    AutoModel,
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
)
from qwen_vl_utils import process_vision_info
from byaldi import RAGMultiModalModel


# ============================
# Global config
# ============================
DATA_DIR = "data"
FAISS_INDEX_PATH = "multimodal_index.faiss"
METADATA_PATH = "metadata_store.pkl"
VLM_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
TEXT_EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLPALI_MODEL_ID = "vidore/colpali-v1.2"

device = "cuda" if torch.cuda.is_available() else "cpu"


# ============================
# Load PDFs as page images
# ============================
def load_pdfs_and_images(data_dir: str = DATA_DIR):
    pdf_files = sorted(
        [f for f in os.listdir(data_dir) if f.lower().endswith(".pdf")]
    )
    all_images_local: Dict[int, List[Image.Image]] = {}
    for doc_id, pdf_name in enumerate(pdf_files):
        pdf_path = os.path.join(data_dir, pdf_name)
        pages = convert_from_path(pdf_path, dpi=200, fmt="jpeg")
        all_images_local[doc_id] = pages
    return pdf_files, all_images_local


pdf_files_ordered, all_images = load_pdfs_and_images()


# ============================
# Load FAISS index + metadata
# ============================
def load_faiss_and_meta(
    index_path: str = FAISS_INDEX_PATH, meta_path: str = METADATA_PATH
):
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found at {index_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata not found at {meta_path}")

    idx = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    if idx.ntotal != len(meta):
        print(
            f" index.ntotal ({idx.ntotal}) != len(metadata) ({len(meta)}) – "
            "retrieval still works but alignment might be off."
        )
    return idx, meta, idx.d


index, metadata_store, INDEX_DIM = load_faiss_and_meta()


# ============================
# Text encoder (for query embedding)
# ============================
text_tokenizer = AutoTokenizer.from_pretrained(TEXT_EMB_MODEL)
text_model = AutoModel.from_pretrained(TEXT_EMB_MODEL).to(device).eval()


def embed_texts(texts: List[str], batch_size: int = 32) -> np.ndarray:
    if not texts:
        return np.zeros((0, text_model.config.hidden_size), dtype="float32")
    embs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = text_tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = text_model(**enc)
            token_emb = out.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1)
            token_emb = token_emb * mask
            summed = token_emb.sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            pooled = (summed / counts).cpu()
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            embs.append(pooled)
    return torch.cat(embs, dim=0).numpy().astype("float32")


def pad_or_truncate(arr: np.ndarray, target_dim: int) -> np.ndarray:
    if arr is None or arr.size == 0:
        return np.zeros((0, target_dim), dtype="float32")
    b, d = arr.shape
    if d == target_dim:
        return arr.astype("float32")
    if d < target_dim:
        pad_width = target_dim - d
        return np.pad(arr, ((0, 0), (0, pad_width)), mode="constant").astype(
            "float32"
        )
    return arr[:, :target_dim].astype("float32")


def embed_query_text(query: str) -> np.ndarray:
    q_emb = embed_texts([query], batch_size=1)
    q_emb = pad_or_truncate(q_emb, INDEX_DIM)
    faiss.normalize_L2(q_emb)
    return q_emb


# ============================
# ColPali image retriever
# ============================
def build_colpali_index(image_root="colpali_images"):
    os.makedirs(image_root, exist_ok=True)
    # Save page images as PNG
    png_paths = []
    for doc_id, pages in all_images.items():
        for pnum, pil_img in enumerate(pages, start=1):
            png_path = os.path.join(image_root, f"doc{doc_id}_page{pnum}.png")
            if not os.path.exists(png_path):
                pil_img.save(png_path, "PNG")
            png_paths.append(png_path)
    model = RAGMultiModalModel.from_pretrained(COLPALI_MODEL_ID)
    model.index(
        input_path=image_root,
        index_name="image_index",
        store_collection_with_index=False,
        overwrite=True,
    )
    return model


try:
    docs_retrieval_model = build_colpali_index()
except Exception as e:
    print(" ColPali not available:", e)
    docs_retrieval_model = None


# ============================
# FAISS search
# ============================
def search_faiss(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    q_emb = embed_query_text(query)
    D, I = index.search(q_emb, top_k)
    results = []
    for score, idx_val in zip(D[0], I[0]):
        if int(idx_val) < 0:
            continue
        meta = metadata_store[int(idx_val)]
        results.append(
            {
                "score": float(score),
                "meta": meta,
                "index_id": int(idx_val),
            }
        )
    return results


def normalize_scores(results: List[Dict[str, Any]], key: str = "score"):
    if not results:
        return results
    vals = [r[key] for r in results]
    mn, mx = min(vals), max(vals)
    if math.isclose(mn, mx):
        for r in results:
            r["norm_score"] = 1.0
        return results
    for r in results:
        r["norm_score"] = (r[key] - mn) / (mx - mn)
    return results


# ============================
# Unified retrieval (FAISS + ColPali)
# ============================
def retrieve_multi_modal(
    query: str, k_faiss: int = 8, k_colpali: int = 3
) -> List[Dict[str, Any]]:
    results = []

    # ---- FAISS (text) ----
    faiss_res = search_faiss(query, top_k=k_faiss)
    normalize_scores(faiss_res)
    for r in faiss_res:
        m = r["meta"]
        results.append(
            {
                "modality": m.get("type"),
                "doc_id": m.get("doc_id"),
                "doc_name": m.get("doc_name"),
                "page_num": m.get("page_num"),
                "content": m.get("content"),
                "score": r["score"],
                "norm_score": r.get("norm_score", 0.0),
                "source": "faiss",
            }
        )

    # ---- ColPali (page images) ----
    if docs_retrieval_model is not None and k_colpali > 0:
        try:
            raw = docs_retrieval_model.search(query, k=k_colpali)
            cols = []
            for rr in raw:
                cols.append(
                    {
                        "score": float(rr["score"]),
                        "doc_id": int(rr["doc_id"]),
                        "page_num": int(rr["page_num"]),
                    }
                )
            normalize_scores(cols)
            for c in cols:
                doc_id = c["doc_id"]
                doc_name = (
                    pdf_files_ordered[doc_id]
                    if (0 <= doc_id < len(pdf_files_ordered))
                    else None
                )
                results.append(
                    {
                        "modality": "page_image",
                        "doc_id": c["doc_id"],
                        "doc_name": doc_name,
                        "page_num": c["page_num"],
                        "content": None,
                        "score": c["score"],
                        "norm_score": c.get("norm_score", 0.0),
                        "source": "colpali",
                    }
                )
        except Exception as e:
            print(" ColPali search failed:", e)

    # sort + dedup
    results.sort(key=lambda x: x.get("norm_score", 0.0), reverse=True)

    seen = set()
    unique = []
    for r in results:
        key = (
            r.get("doc_id"),
            r.get("page_num"),
            r.get("modality"),
            (r.get("content") or "")[:120],
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)

    return unique


# ============================
# Context preparation
# ============================
def prepare_context_for_model(
    retrieved: List[Dict[str, Any]], max_text: int = 4, max_img: int = 2
):
    text_snips: List[str] = []
    img_snips: List[Image.Image] = []
    used_meta: List[Dict[str, Any]] = []

    t = im = 0
    for r in retrieved:
        mod = r.get("modality")
        if mod in ("text", "table_row", "table_header", "chart_caption") and t < max_text:
            text_snips.append(f"[p.{r['page_num']}] {r['content']}")
            used_meta.append(
                {
                    "doc_name": r["doc_name"],
                    "page_num": r["page_num"],
                    "modality": mod,
                    "score": r.get("score"),
                    "source": r.get("source", "faiss/colpali"),
                }
            )
            t += 1
        elif mod == "page_image" and im < max_img:
            try:
                img_obj = all_images[r["doc_id"]][r["page_num"] - 1]
                img_snips.append(img_obj)
                used_meta.append(
                    {
                        "doc_name": r["doc_name"],
                        "page_num": r["page_num"],
                        "modality": "page_image",
                        "score": r.get("score"),
                        "source": r.get("source", "colpali"),
                    }
                )
                im += 1
            except Exception:
                continue

        if t >= max_text and im >= max_img:
            break

    return text_snips, img_snips, used_meta


# ============================
# VLM loader + safe generation
# ============================
try:
    vl_model = Qwen2VLForConditionalGeneration.from_pretrained(
        VLM_MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
    )
    vl_processor = Qwen2VLProcessor.from_pretrained(
        VLM_MODEL_ID, min_pixels=224 * 224, max_pixels=1024 * 1024
    )
    vl_available = True
except Exception as e:
    print(" VLM load failed, falling back to text-only:", e)
    vl_model = None
    vl_processor = None
    vl_available = False


def generate_answer_safe(
    system_instruction: str,
    user_question: str,
    text_snippets: List[str] = None,
    image_list: List[Image.Image] = None,
    max_new_tokens: int = 256,
):
    text_snippets = text_snippets or []
    image_list = image_list or []

    if not vl_available:
        # Simple text-only fallback: return context + question for debugging
        context = "\n".join(text_snippets[:8])
        fallback = (
            "VLM unavailable in this runtime. Here is the retrieved context and question:\n\n"
            f"{context}\n\nQuestion: {user_question}"
        )
        return fallback, {"vl_available": False}

    # Build chat input
    content_blocks = [{"type": "text", "text": system_instruction}]
    for t in text_snippets[:10]:
        content_blocks.append({"type": "text", "text": t})
    content_blocks.append({"type": "text", "text": "User question: " + user_question})

    chat = [{"role": "user", "content": content_blocks}]

    try:
        raw_text = vl_processor.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        raw_text = (
            system_instruction
            + "\n\n"
            + "\n".join(text_snippets[:8])
            + "\n\nUser question: "
            + user_question
        )

    limited_images = image_list[:1]

    try:
        if limited_images:
            image_inputs, _ = process_vision_info(chat)
            inputs = vl_processor(
                text=[raw_text],
                images=image_inputs,
                padding=True,
                return_tensors="pt",
            )
        else:
            inputs = vl_processor(
                text=[raw_text], padding=True, return_tensors="pt"
            )
    except Exception:
        inputs = vl_processor(
            text=[raw_text], padding=True, return_tensors="pt"
        )

    inputs = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in inputs.items()
    }

    with torch.no_grad():
        out = vl_model.generate(**inputs, max_new_tokens=max_new_tokens)

    if "input_ids" in inputs:
        inlen = inputs["input_ids"].shape[1]
        out_trim = out[:, inlen:]
    else:
        out_trim = out

    answer = vl_processor.batch_decode(
        out_trim, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return answer, {"vl_available": True}


# ============================
# High-level RAG QA
# ============================
def format_sources(used_metadata: List[Dict[str, Any]]) -> str:
    seen = set()
    lines = []
    for m in used_metadata:
        doc = m.get("doc_name") or "unknown"
        page = m.get("page_num") or "?"
        key = (doc, page)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- {doc} [p.{page}]")
    return "\n".join(lines)


def answer_with_multimodal_rag(
    query: str,
    top_k_faiss: int = 8,
    top_k_colpali: int = 3,
    max_text_snips: int = 6,
    max_img_snips: int = 2,
    max_new_tokens: int = 256,
):
    # 1) Retrieve
    retrieved = retrieve_multi_modal(
        query, k_faiss=top_k_faiss, k_colpali=top_k_colpali
    )

    # 2) Prepare context
    text_snips, img_snips, used_meta = prepare_context_for_model(
        retrieved, max_text=max_text_snips, max_img=max_img_snips
    )

    pages_used = sorted(
        {
            m.get("page_num")
            for m in used_meta
            if m.get("page_num") is not None
        }
    )
    pages_info = ", ".join(map(str, pages_used)) if pages_used else "N/A"

    system_instruction = f"""
You are a careful multi-modal assistant. Use ONLY the provided context to answer.
Each snippet begins with a page tag like [p.5].

Rules:
- Every factual statement MUST be backed by a snippet and include the same [p.X] citation.
- If the context is insufficient, say: "I am unsure based on the provided documents."

Retrieved candidate pages: {pages_info}
""".strip()

    answer, _ = generate_answer_safe(
        system_instruction,
        query,
        text_snippets=text_snips,
        image_list=img_snips,
        max_new_tokens=max_new_tokens,
    )

    final = answer.strip() + "\n\nSources:\n" + (
        format_sources(used_meta) if used_meta else "- (no sources retrieved)"
    )

    return final, used_meta
