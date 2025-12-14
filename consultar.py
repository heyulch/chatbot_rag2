import os
import argparse
from dotenv import load_dotenv

import torch
import numpy as np
from pinecone import Pinecone
from transformers import AutoTokenizer, AutoModel

from router import route  


# ========= Embeddings  =========
MODEL_NAME = "jinaai/jina-embeddings-v2-base-es"

@torch.no_grad()
def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts

def embed_query(text: str, tokenizer, model) -> np.ndarray:
    tok = tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    tok = {k: v.to(model.device) for k, v in tok.items()}
    out = model(**tok)
    emb = mean_pooling(out.last_hidden_state, tok["attention_mask"])  # (1,H)
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb[0].detach().to("cpu").to(torch.float32).numpy()


#  Pinecone retrieve 
def retrieve_chunks(index, namespace: str, cv_id: str, qvec: np.ndarray, top_k: int = 4) -> list[str]:
    res = index.query(
        vector=qvec.tolist(),
        top_k=top_k,
        namespace=namespace,
        include_metadata=True,
        filter={"cv_id": {"$eq": cv_id}}
    )
    matches = res.get("matches", []) or []
    texts = []
    for m in matches:
        md = m.get("metadata") or {}
        t = md.get("text")
        if t:
            texts.append(t)
    return texts


#  “Agentes”
AGENT_SYSTEM = {
    "heyul": "Eres el Agente de Heyul. Responde solo con información del CV de Heyul. Si falta dato, dilo.",
    "maria": "Eres el Agente de María. Responde solo con información del CV de María. Si falta dato, dilo.",
    "carlos": "Eres el Agente de Carlos. Responde solo con información del CV de Carlos. Si falta dato, dilo.",
}

def format_context(cv_id: str, chunks: list[str]) -> str:
    # contexto simple
    return f"CV_ID={cv_id}\n" + "\n---\n".join(chunks)


#  LLM (Groq) 
def call_groq(system: str, user: str) -> str:
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", default="cv-demo")
    parser.add_argument("--top_k", type=int, default=4)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--no_llm", action="store_true", help="Si se activa, no llama a Groq; imprime contexto recuperado.")
    args = parser.parse_args()

    load_dotenv()

    # Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

    # Embedding model
    dtype = torch.float16 if (args.fp16 and args.device == "cuda") else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    #model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=dtype, low_cpu_mem_usage=True)
    model = AutoModel.from_pretrained(
    MODEL_NAME,
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
    attn_implementation="eager")

    model = model.to(args.device)
    model.eval()

    print(" Listo. Escribe tu pregunta.")
    while True:
        query = input("\n> ").strip()
        if not query:
            break

        decision = route(query)  
        mode = decision["mode"]
        cv_ids = decision["cv_ids"]

        qvec = embed_query(query, tokenizer, model)

        if mode == "single":
            cv_id = cv_ids[0]
            chunks = retrieve_chunks(index, args.namespace, cv_id, qvec, top_k=args.top_k)
            context = format_context(cv_id, chunks)

            if args.no_llm:
                print(f"\n[ROUTE] single -> {cv_id}")
                print(context)
                continue

            system = AGENT_SYSTEM.get(cv_id, "Eres un agente y respondes usando solo el contexto provisto.")
            user_prompt = f"Pregunta:\n{query}\n\nContexto:\n{context}"
            answer = call_groq(system, user_prompt)
            print(f"\n[ROUTE] single -> {cv_id}\n{answer}")

        else:
            # multi
            contexts = {}
            for cv_id in cv_ids:
                chunks = retrieve_chunks(index, args.namespace, cv_id, qvec, top_k=args.top_k)
                contexts[cv_id] = chunks

            if args.no_llm:
                print(f"\n[ROUTE] multi -> {cv_ids}")
                for cv_id in cv_ids:
                    print("\n" + "="*20)
                    print(format_context(cv_id, contexts[cv_id]))
                continue

            # Prompt comparativo por persona
            blocks = []
            for cv_id in cv_ids:
                blocks.append(format_context(cv_id, contexts[cv_id]))
            joined = "\n\n====================\n\n".join(blocks)

            system = (
                "Eres un asistente que responde comparando múltiples CVs. "
             #   "Responde con secciones por persona (una sección por CV_ID). "
                "No inventes datos fuera del contexto."
            )
            user_prompt = f"Pregunta:\n{query}\n\nContextos:\n{joined}"
            answer = call_groq(system, user_prompt)
            print(f"\n[ROUTE] multi -> {cv_ids}\n{answer}")


if __name__ == "__main__":
    main()
