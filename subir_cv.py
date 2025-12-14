import os
import argparse
import gc
import numpy as np
import torch
from dotenv import load_dotenv
from pinecone import Pinecone
from transformers import AutoTokenizer, AutoModel


def split_text_into_chunks(text: str, max_chars: int = 700, overlap: int = 80) -> list[str]:
    text = " ".join(text.split())
    n = len(text)
    if n == 0:
        return []

    overlap = min(overlap, max_chars - 1) if max_chars > 1 else 0
    chunks = []
    start = 0

    while start < n:
        end = min(start + max_chars, n)

        if end < n:
            window = text[start:end]
            last_space = window.rfind(" ")
            if last_space != -1 and last_space > 200:
                end = start + last_space

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        new_start = end - overlap
        if new_start <= start:
            new_start = end
        start = new_start

    return chunks


def batched(items, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield i, items[i:i + batch_size]


@torch.no_grad()
def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # last_hidden_state: (B, T, H)
    # attention_mask: (B, T)
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # (B,T,1)
    summed = (last_hidden_state * mask).sum(dim=1)                  # (B,H)
    counts = mask.sum(dim=1).clamp(min=1e-9)                        # (B,1)
    return summed / counts


def main():
    parser = argparse.ArgumentParser(description="Subir CV a Pinecone con jina-embeddings-v2-small-en (TB3).")
    parser.add_argument("--cv_id", required=True)
    parser.add_argument("--file", required=True)
    parser.add_argument("--namespace", default="cv-demo")
    parser.add_argument("--reindex", action="store_true")

    parser.add_argument("--max_chars", type=int, default=700)
    parser.add_argument("--overlap", type=int, default=80)

    parser.add_argument("--embed_batch", type=int, default=16)
    parser.add_argument("--upsert_batch", type=int, default=100)

    parser.add_argument("--model", default="jinaai/jina-embeddings-v2-base-es")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--fp16", action="store_true", help="Usar float16 en GPU")

    args = parser.parse_args()

    print("[1/7] .env", flush=True)
    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    if not api_key or not index_name:
        raise ValueError("Faltan PINECONE_API_KEY o PINECONE_INDEX_NAME en .env")

    if not os.path.exists(args.file):
        raise FileNotFoundError(f"No existe el archivo: {args.file}")

    print("[2/7] Leer CV", flush=True)
    with open(args.file, "r", encoding="utf-8") as f:
        text = f.read()

    print("[3/7] Chunks", flush=True)
    chunks = split_text_into_chunks(text, max_chars=args.max_chars, overlap=args.overlap)
    total = len(chunks)
    print(f"Archivo={os.path.basename(args.file)} | chars={len(text)} | chunks={total}", flush=True)

    print("[4/7] Pinecone", flush=True)
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)

    if args.reindex:
        try:
            index.delete(namespace=args.namespace, filter={"cv_id": {"$eq": args.cv_id}})
            print(" Delete OK", flush=True)
        except Exception as e:
            if "Namespace not found" in str(e):
                print("ℹ Namespace no existe todavía. Se creará con el primer upsert.", flush=True)
            else:
                raise

    print("[5/7] Cargar modelo HF", flush=True)
    # dtype
    use_fp16 = args.fp16 and args.device == "cuda"
    dtype = torch.float16 if use_fp16 else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    )

    if args.device == "cuda":
        model = model.to("cuda")
    model.eval()

    print(f" Modelo listo | device={args.device} | fp16={use_fp16}", flush=True)

    print("[6/7] Embeddings + upsert", flush=True)
    source_file = os.path.basename(args.file)
    vectors_buffer = []
    upserted = 0

    for start_i, batch_texts in batched(chunks, args.embed_batch):
        # Tokenización 
        tok = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        tok = {k: v.to(model.device) for k, v in tok.items()}

        out = model(**tok)
        emb = mean_pooling(out.last_hidden_state, tok["attention_mask"])  # (B,H)

        # normalizar L2 para cosine
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        emb = emb.detach().to("cpu").to(torch.float32).numpy()

        for j, ch in enumerate(batch_texts):
            idx = start_i + j
            vectors_buffer.append({
                "id": f"{args.cv_id}_chunk_{idx}",
                "values": emb[j].tolist(),
                "metadata": {
                    "cv_id": args.cv_id,
                    "chunk_index": idx,
                    "source_file": source_file,
                    "text": ch
                }
            })

        if len(vectors_buffer) >= args.upsert_batch:
            index.upsert(vectors=vectors_buffer, namespace=args.namespace)
            upserted += len(vectors_buffer)
            print(f"Upsert parcial: {upserted}/{total}", flush=True)
            vectors_buffer.clear()
            gc.collect()

    if vectors_buffer:
        index.upsert(vectors=vectors_buffer, namespace=args.namespace)
        upserted += len(vectors_buffer)
        print(f"Upsert final: {upserted}/{total}", flush=True)
        vectors_buffer.clear()

    print("[7/7] OK", flush=True)
    print(f" Listo. Indexado cv_id={args.cv_id} con {total} chunks en namespace={args.namespace}", flush=True)


if __name__ == "__main__":
    main()
