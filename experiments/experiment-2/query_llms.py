#!/usr/bin/env python3
"""
Experiment 2 extension: query multimodal LLMs on the SDPC scatterplot pairs.

For every model, each PNG pair in data/SDPC/imgpairs-png is sent together with the
prompt in experiments/experiment-2/prompt.md to an OpenAI-compatible endpoint
(vLLM, SGLang, LiteLLM, ...). The answer must be a JSON object with the keys
"plot" (A, B or BOTH) and "motivation". Invalid answers are requested again.
Valid answers are appended to responses/responses-<model>.csv with the columns
id,model,response,motivation,reasoning: the first ones as in responses-gpt4o.csv,
plus the model id and the model's reasoning text (empty if the model does not reason).

The script is resumable: ids already present in a model's CSV are skipped.

Examples
--------
# Models listed in a config file (see models.example.json)
python query_llms.py --config models.json

# Models given on the command line, all on the same endpoint
python query_llms.py --base-url http://caredgx:8000/v1 --models Qwen/Qwen2.5-VL-72B-Instruct

# Every model served by the endpoint (/v1/models)
python query_llms.py --base-url http://caredgx:8000/v1
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import csv
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

DEFAULT_PROMPT = HERE / "prompt.md"
DEFAULT_IMAGES = REPO / "data" / "SDPC" / "imgpairs-png"
DEFAULT_OUT = HERE / "responses"

VALID_PLOTS = {"A", "B", "BOTH"}
CSV_HEADER = ["id", "model", "response", "motivation", "reasoning"]


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class ModelSpec:
    model: str                      # model name as served by the endpoint
    base_url: str                   # e.g. http://host:8000/v1
    name: str | None = None         # alias used for the output file name
    api_key: str | None = None      # literal key (prefer api_key_env)
    api_key_env: str | None = None  # environment variable holding the key
    extra: dict[str, Any] = field(default_factory=dict)  # extra request params

    @property
    def alias(self) -> str:
        raw = self.name or self.model.split("/")[-1]
        return re.sub(r"[^A-Za-z0-9._-]+", "_", raw)

    def key(self) -> str:
        if self.api_key:
            return self.api_key
        if self.api_key_env and os.environ.get(self.api_key_env):
            return os.environ[self.api_key_env]
        return os.environ.get("OPENAI_API_KEY", "EMPTY")


def load_specs(args: argparse.Namespace) -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    if args.config:
        cfg = json.loads(Path(args.config).read_text())
        default_url = cfg.get("base_url", args.base_url)
        for m in cfg["models"]:
            m = dict(m)
            m.setdefault("base_url", default_url)
            if not m["base_url"]:
                sys.exit(f"No base_url for model {m.get('model')}")
            specs.append(ModelSpec(**m))
    elif args.models:
        if not args.base_url:
            sys.exit("--base-url is required with --models")
        specs = [ModelSpec(model=m, base_url=args.base_url) for m in args.models]
    elif args.base_url:
        specs = asyncio.run(discover_models(args.base_url))
        if not specs:
            sys.exit(f"No models found at {args.base_url}/models")
    else:
        sys.exit("Provide --config, or --base-url (optionally with --models)")
    if args.only:
        specs = [s for s in specs if s.alias in args.only or s.model in args.only]
    return specs


async def discover_models(base_url: str) -> list[ModelSpec]:
    client = AsyncOpenAI(base_url=base_url, api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    models = await client.models.list()
    return [ModelSpec(model=m.id, base_url=base_url) async for m in models]


# --------------------------------------------------------------------------- #
# Response validation
# --------------------------------------------------------------------------- #
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)
PLOT_RE = re.compile(r"^(?:(?:SCATTER)?PLOT\s*)?(A|B|BOTH)$")


class InvalidAnswer(ValueError):
    pass


def split_thinking(text: str | None) -> tuple[str, str]:
    """Split a message content into (answer, thinking) using <think> tags."""
    if not text:
        return "", ""
    parts = [m.strip() for m in THINK_RE.findall(text)]
    text = THINK_RE.sub("", text)
    low = text.lower()
    if "</think>" in low:  # opening tag consumed by the chat template
        cut = low.rfind("</think>")
        parts.append(text[:cut].replace("<think>", "").strip())
        text = text[cut + len("</think>"):]
    return text.strip(), "\n\n".join(p for p in parts if p)


def message_reasoning(message: Any) -> str:
    """Reasoning returned in a separate field (vLLM/SGLang reasoning parsers)."""
    for attr in ("reasoning_content", "reasoning"):
        value = getattr(message, attr, None)
        if value is None:
            value = (getattr(message, "model_extra", None) or {}).get(attr)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def parse_answer(text: str | None, strict: bool) -> tuple[str, str]:
    """Return (plot, motivation) or raise InvalidAnswer. `text` has no thinking."""
    if not text or not text.strip():
        raise InvalidAnswer("empty response")
    text = text.strip()

    m = FENCE_RE.match(text)
    if m:
        text = m.group(1)

    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        if strict:
            raise InvalidAnswer(f"not valid JSON: {e}") from None
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise InvalidAnswer("no JSON object found") from None
        try:
            obj = json.loads(text[start:end + 1])
        except json.JSONDecodeError as e2:
            raise InvalidAnswer(f"not valid JSON: {e2}") from None

    if not isinstance(obj, dict):
        raise InvalidAnswer("JSON is not an object")

    obj = {str(k).strip().lower(): v for k, v in obj.items()}
    if "plot" not in obj or "motivation" not in obj:
        raise InvalidAnswer(f"missing keys, got {sorted(obj)}")

    plot = obj["plot"]
    if not isinstance(plot, str):
        raise InvalidAnswer(f"'plot' is not a string: {plot!r}")
    pm = PLOT_RE.match(plot.strip().upper())
    if not pm:
        raise InvalidAnswer(f"'plot' not in A/B/BOTH: {plot!r}")

    motivation = obj["motivation"]
    if not isinstance(motivation, str) or not motivation.strip():
        raise InvalidAnswer("'motivation' is empty or not a string")

    return pm.group(1), motivation.strip()


# --------------------------------------------------------------------------- #
# Querying
# --------------------------------------------------------------------------- #
def image_data_url(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def build_messages(prompt: str, image_url: str, system: bool) -> list[dict]:
    image_part = {"type": "image_url", "image_url": {"url": image_url}}
    if system:
        return [
            {"role": "system", "content": prompt},
            {"role": "user", "content": [image_part]},
        ]
    return [{"role": "user", "content": [{"type": "text", "text": prompt}, image_part]}]


class Stats:
    def __init__(self) -> None:
        self.invalid = 0
        self.api_errors = 0
        self.failed = 0


async def query_one(
    client: AsyncOpenAI,
    spec: ModelSpec,
    pair_id: str,
    image: Path,
    prompt: str,
    args: argparse.Namespace,
    sem: asyncio.Semaphore,
    stats: Stats,
    raw_log,
) -> tuple[str, str, str] | None:
    """Return (plot, motivation, reasoning) or None if no valid answer."""
    async with sem:
        messages = build_messages(prompt, image_data_url(image), args.system_prompt)
        params: dict[str, Any] = {}
        if args.temperature is not None:
            params["temperature"] = args.temperature
        if args.max_tokens is not None:
            params["max_tokens"] = args.max_tokens
        if args.json_mode:
            params["response_format"] = {"type": "json_object"}
        params.update(spec.extra)

        for attempt in range(1, args.max_attempts + 1):
            try:
                resp = await client.chat.completions.create(
                    model=spec.model, messages=messages, **params
                )
            except Exception as e:  # network errors, 5xx, timeouts, rate limits
                stats.api_errors += 1
                log_raw(raw_log, pair_id, attempt, None, f"api error: {e!r}")
                await asyncio.sleep(min(60, 2 ** attempt) + random.random())
                continue

            message = resp.choices[0].message if resp.choices else None
            text = message.content if message else None
            reasoning = message_reasoning(message) if message else ""
            answer, inline_thinking = split_thinking(text)
            full_reasoning = "\n\n".join(t for t in (reasoning, inline_thinking) if t)
            try:
                plot, motivation = parse_answer(answer, args.strict_json)
            except InvalidAnswer as e:
                stats.invalid += 1
                log_raw(raw_log, pair_id, attempt, text, str(e), reasoning)
                continue

            log_raw(raw_log, pair_id, attempt, text, None, reasoning)
            return plot, motivation, full_reasoning

        stats.failed += 1
        return None


def log_raw(fh, pair_id: str, attempt: int, text: str | None, error: str | None,
            reasoning: str | None = None) -> None:
    if fh is None:
        return
    fh.write(json.dumps({
        "id": pair_id, "attempt": attempt, "time": time.time(),
        "valid": error is None, "error": error, "raw": text, "reasoning": reasoning or None,
    }, ensure_ascii=False) + "\n")
    fh.flush()


def done_ids(csv_path: Path, model: str) -> set[str]:
    """Ids already answered. A CSV written by an older version of this script
    (id,response,motivation or a 'thinking' column) is upgraded in place."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        rows = list(reader)
    if header != CSV_HEADER:
        missing = [c for c in ("id", "response", "motivation") if c not in header]
        if missing:
            sys.exit(f"{csv_path}: unexpected header {header}")
        tmp = csv_path.with_suffix(".csv.tmp")
        with tmp.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADER, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                row.setdefault("model", model)
                row["model"] = row["model"] or model
                row["reasoning"] = row.get("reasoning") or row.get("thinking") or ""
                writer.writerow(row)
        tmp.replace(csv_path)
        print(f"Upgraded {csv_path} to columns {','.join(CSV_HEADER)}")
    return {row["id"] for row in rows}


async def run_model(spec: ModelSpec, images: list[Path], prompt: str, args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"responses-{spec.alias}.csv"

    done = done_ids(csv_path, spec.model)
    todo = [p for p in images if p.stem not in done]
    print(f"\n[{spec.alias}] model={spec.model} endpoint={spec.base_url}")
    print(f"[{spec.alias}] {len(done)} already done, {len(todo)} to query -> {csv_path}")
    if not todo:
        return

    new_file = not csv_path.exists() or csv_path.stat().st_size == 0
    client = AsyncOpenAI(base_url=spec.base_url, api_key=spec.key(),
                         timeout=args.timeout, max_retries=0)
    sem = asyncio.Semaphore(args.concurrency)
    stats = Stats()

    raw_log = None
    if not args.no_raw_log:
        (out_dir / "raw").mkdir(exist_ok=True)
        raw_log = (out_dir / "raw" / f"{spec.alias}.jsonl").open("a", encoding="utf-8")

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(CSV_HEADER)
            f.flush()

        async def task(p: Path):
            return p.stem, await query_one(client, spec, p.stem, p, prompt, args, sem, stats, raw_log)

        bar = tqdm(total=len(todo), desc=spec.alias, unit="pair", dynamic_ncols=True)
        try:
            for fut in asyncio.as_completed([task(p) for p in todo]):
                pair_id, result = await fut
                if result is not None:
                    plot, motivation, reasoning = result
                    writer.writerow([pair_id, spec.model, plot, motivation, reasoning])
                    f.flush()
                bar.update(1)
                bar.set_postfix(retry_json=stats.invalid, api_err=stats.api_errors,
                                failed=stats.failed, refresh=False)
        finally:
            bar.close()
            if raw_log:
                raw_log.close()
            await client.close()

    if stats.failed:
        print(f"[{spec.alias}] {stats.failed} pairs had no valid answer after "
              f"{args.max_attempts} attempts; run the script again to retry them.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_argument_group("models")
    src.add_argument("--config", help="JSON file with the models (see models.example.json)")
    src.add_argument("--base-url", help="OpenAI-compatible endpoint, e.g. http://host:8000/v1")
    src.add_argument("--models", nargs="+", help="model names served at --base-url")
    src.add_argument("--only", nargs="+", help="run only these models (alias or name)")

    io = ap.add_argument_group("input/output")
    io.add_argument("--prompt", default=str(DEFAULT_PROMPT))
    io.add_argument("--images", default=str(DEFAULT_IMAGES))
    io.add_argument("--out", default=str(DEFAULT_OUT))
    io.add_argument("--limit", type=int, help="only the first N pairs (for a quick test)")
    io.add_argument("--no-raw-log", action="store_true", help="do not keep raw answers in responses/raw/")

    q = ap.add_argument_group("querying")
    q.add_argument("--concurrency", type=int, default=20, help="simultaneous requests (default 20)")
    q.add_argument("--max-attempts", type=int, default=10,
                   help="attempts per pair before giving up for this run (default 10)")
    q.add_argument("--timeout", type=float, default=600, help="request timeout in seconds")
    q.add_argument("--temperature", type=float, default=None, help="default: server default")
    q.add_argument("--max-tokens", type=int, default=None, help="default: server default")
    q.add_argument("--json-mode", action="store_true",
                   help="send response_format=json_object (constrained decoding)")
    q.add_argument("--strict-json", action="store_true",
                   help="reject answers with text around the JSON object (code fences still allowed)")
    q.add_argument("--system-prompt", action="store_true",
                   help="send the prompt as system message instead of user text")
    args = ap.parse_args()

    prompt = Path(args.prompt).read_text(encoding="utf-8").strip()
    images = sorted(Path(args.images).glob("*.png"))
    if not images:
        sys.exit(f"No PNG files in {args.images}")
    if args.limit:
        images = images[: args.limit]

    specs = load_specs(args)
    print(f"{len(images)} image pairs, {len(specs)} model(s): {', '.join(s.alias for s in specs)}")
    for spec in specs:
        asyncio.run(run_model(spec, images, prompt, args))


if __name__ == "__main__":
    main()
