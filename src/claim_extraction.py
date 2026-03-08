import asyncio
import os
import time
import orjson
from groq import Groq
from tqdm import tqdm
from tenacity import retry, wait_exponential, stop_after_attempt
from dotenv import load_dotenv

load_dotenv()

MODEL = "llama3-70b-8192"
INPUT_FILE = "../data/processed/chunks.json"
OUTPUT_FILE = "extractions.json"

SAVE_INTERVAL = 600  # 10 minutes

client = Groq(api_key=os.environ["GROQ_API_KEY"])


# -----------------------------
# Load / Save
# -----------------------------

def load_chunks():
    with open(INPUT_FILE, "rb") as f:
        return orjson.loads(f.read())


def save_results(results):
    with open(OUTPUT_FILE, "wb") as f:
        f.write(orjson.dumps(results, option=orjson.OPT_INDENT_2))


# -----------------------------
# Prompt
# -----------------------------

def build_prompt(chunk):

    text = chunk["text"][:4000]  # safety truncation

    return f"""
Extract structured knowledge from the GitHub discussion.

Return ONLY JSON:

{{
 "entities": [],
 "claims": [],
 "evidence": []
}}

Issue Title: {chunk['issue_title']}
Issue State: {chunk['issue_state']}
Author: {chunk['author']}
Timestamp: {chunk['timestamp']}

Text:
{text}
"""


# -----------------------------
# Extraction
# -----------------------------

@retry(wait=wait_exponential(multiplier=1, min=1, max=30),
       stop=stop_after_attempt(5))
async def extract_chunk(chunk):

    prompt = build_prompt(chunk)

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {"role": "system",
             "content": "Extract entities, claims, evidence. Return JSON only."},
            {"role": "user", "content": prompt}
        ]
    )

    output = response.choices[0].message.content

    try:
        data = orjson.loads(output)
    except Exception:
        data = {"entities": [], "claims": [], "evidence": []}

    return {
        "chunk_id": chunk["chunk_id"],
        "artifact_id": chunk["artifact_id"],
        "entities": data.get("entities", []),
        "claims": data.get("claims", []),
        "evidence": data.get("evidence", [])
    }


# -----------------------------
# Worker
# -----------------------------

async def worker(queue, results, progress_bar, last_save):

    while True:

        chunk = await queue.get()

        if chunk is None:
            queue.task_done()
            break

        try:
            result = await extract_chunk(chunk)
            results.append(result)

        except Exception as e:
            print("error:", e)

        progress_bar.update(1)

        # timed checkpoint
        if time.time() - last_save[0] > SAVE_INTERVAL:
            save_results(results)
            last_save[0] = time.time()
            print("\ncheckpoint saved")

        queue.task_done()


# -----------------------------
# Pipeline
# -----------------------------

async def run_pipeline(chunks, workers=8):

    queue = asyncio.Queue()

    for chunk in chunks:
        queue.put_nowait(chunk)

    results = []

    progress_bar = tqdm(total=len(chunks), desc="Extracting")

    last_save = [time.time()]

    tasks = [
        asyncio.create_task(worker(queue, results, progress_bar, last_save))
        for _ in range(workers)
    ]

    await queue.join()

    for _ in range(workers):
        queue.put_nowait(None)

    await asyncio.gather(*tasks)

    progress_bar.close()

    return results


# -----------------------------
# Main
# -----------------------------

async def main():

    chunks = load_chunks()

    print("chunks:", len(chunks))

    try:
        results = await run_pipeline(chunks, workers=8)

        save_results(results)

    except KeyboardInterrupt:
        print("\nProcess interrupted. Saving progress...")
        save_results(results)

    print("done")


if __name__ == "__main__":
    asyncio.run(main())