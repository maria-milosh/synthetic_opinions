import os, json, time, uuid
from datetime import datetime, timezone

from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

QUESTION = """Which module should receive the additional 10% grade bonus? Please make a choice (A, B, C, or D) and explain your reasoning. In your answer:
    - State your choice clearly.
    - Explain your reasoning in 2–5 sentences.
    - Write a short 1–2 sentence argument that could be shown to other students.
    """

MODEL = os.environ.get("MODEL", "gpt-4.1-mini")
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.8"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "600"))

RUN_ID = os.environ.get("RUN_ID", str(uuid.uuid4()))
TIMESTAMP = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M-%S")
OUT_PATH = os.environ.get("OUT_PATH", "outputs/run_{}/transcripts.jsonl".format(TIMESTAMP))
MANIFEST_PATH = os.environ.get("MANIFEST_PATH", "outputs/run_{}/run_manifest.json".format(TIMESTAMP))

def load_personas(path="personas.jsonl"):
    personas = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                personas.append(json.loads(line))
    return personas

def load_template(path="prompt_template.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def render_prompt(template: str, persona: dict, question: str = "") -> str:
    return template.format(
        persona=json.dumps(persona, ensure_ascii=False),
        question=question)

def call_llm(prompt: str, retries: int = 3, backoff: float = 1.5):
    last_err = None
    for attempt in range(retries):
        try:
            resp = client.responses.create(
                model=MODEL,
                input=prompt,
                temperature=TEMPERATURE,
                max_output_tokens=MAX_TOKENS,
            )
            # responses API returns output in a structured way; easiest is output_text:
            text = resp.output_text
            usage = getattr(resp, "usage", None)
            usage_dict = usage.model_dump() if usage else None
            return text, usage_dict, None, attempt
        except Exception as e:
            last_err = str(e)
            time.sleep(backoff ** attempt)
    return None, None, last_err, retries

def parse_response(text):
    try:
        return json.loads(text), None
    except json.JSONDecodeError as e:
        return None, str(e)

def append_jsonl(path, record: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def main():
    personas = load_personas()
    template = load_template()

    # Write run manifest once
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    manifest = {
        "run_id": RUN_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "temperature": TEMPERATURE,
        "max_output_tokens": MAX_TOKENS,
        "n_personas": len(personas),
        "template_path": "prompt_template.txt",
        "personas_path": "personas.jsonl",
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    for persona in personas:
        persona_id = persona.get("id") or persona.get("persona_id") or "unknown"
        prompt = render_prompt(template, persona, question=QUESTION)

        text, usage, err, retry_count = call_llm(prompt)

        record = {
            "run_id": RUN_ID,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model": MODEL,
            "temperature": TEMPERATURE,
            "max_output_tokens": MAX_TOKENS,
            "persona_id": persona_id,
            # "prompt": prompt, # store full prompt for auditability
            "response_text": text, # raw transcript
            "parsed_response": parse_response(text)[0],
            "usage": usage,
            "error": err,
            "retry_count": retry_count,
        }
        append_jsonl(OUT_PATH, record)
        print(f"Wrote persona {persona_id} (err={bool(err)})")

if __name__ == "__main__":
    main()
