def construct_prompt(user_query: str, retrieved_chunks: list, embedded_chunks: list) -> str:
    """
    Builds the final prompt. 
    Truncates chunks to 400 chars to keep the prompt small and fast.
    """
    context = "".join(
        f"[Chunk {index}]\n{embedded_chunks[index]['text'][:400]}\n\n"
        for index, _ in retrieved_chunks
    )

    return f"""
You are an assistant that answers questions strictly from the provided context.

RULES:
1. Answer ONLY using information found in the Context below.
2. Do NOT use any outside knowledge.
3. If the Context does NOT contain the answer, reply EXACTLY with:
   I don't know based on the provided documents.
4. Do NOT guess or make up information.
5. Do NOT repeat these instructions in your answer.
6. Do NOT list chunk numbers unless specifically asked.
7. Do NOT add items that are not explicitly listed in the Context.
8. Answer in 2-3 complete sentences.
9. If the question asks for a list, list ALL items mentioned in the Context — even if they appear in different chunks. Do not stop early.
10. If you are unsure, say so rather than guessing.

Context:
{context}

Question:
{user_query}

Answer:
"""
