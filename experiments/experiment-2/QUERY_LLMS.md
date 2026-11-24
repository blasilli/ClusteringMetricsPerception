# Querying LLMs on the SDPC pairs (`query_llms.py`)

`query_llms.py` extends Experiment 2 to new multimodal LLMs. Each scatterplot pair in `data/SDPC/imgpairs-png` (879 PNG files) is sent to the model together with the prompt in `prompt.md`. The model must answer with a JSON object containing `plot` (`A`, `B` or `BOTH`) and `motivation`. Invalid answers are requested again, and valid answers are appended to one CSV per model.

## 1. Setup

Requires Python 3.9 or newer. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r experiments/experiment-2/requirements.txt
```

All the commands below are run from `experiments/experiment-2`:

```bash
cd experiments/experiment-2
```

## 2. Configure the models

Any OpenAI compatible endpoint works (vLLM, SGLang, LiteLLM, OpenAI). There are three ways to choose the models.

**Config file (recommended).** A JSON file listing the models, with an optional per model endpoint, output alias, API key and extra request parameters. See `models.example.json` for all the fields, and `models.test.json` for the test server.

```json
{
  "base_url": "http://HOST:PORT/v1",
  "models": [
    {"name": "alias-used-in-file-name", "model": "org/model-id"},
    {"name": "other", "model": "org/other-id", "base_url": "http://HOST:OTHER_PORT/v1",
     "api_key_env": "MY_KEY_VAR", "extra": {"temperature": 0}}
  ]
}
```

```bash
python query_llms.py --config models.json
```

**Command line.** All the models must be on the same endpoint.

```bash
python query_llms.py --base-url http://HOST:PORT/v1 --models org/model-a org/model-b
```

**Auto discovery.** Every model listed by the endpoint's `/v1/models` is queried.

```bash
python query_llms.py --base-url http://HOST:PORT/v1
```

The API key is read from `api_key`, then from the variable named in `api_key_env`, then from `OPENAI_API_KEY`. If none is set, a dummy key is sent, which is fine for servers that need no key.

## 3. Preliminary test

The test server at `http://151.100.96.175:9000/v1` needs no API key and serves two models, both configured in `models.test.json`:

| Model ID | Alias (output file) |
| --- | --- |
| `cyankiwi/MiniMax-M3-AWQ-INT4` | `MiniMax-M3-AWQ-INT4` |
| `Qwen/Qwen3.8-27B` | `Qwen3.8-27B` |

Check that the server is reachable and lists the models:

```bash
curl http://151.100.96.175:9000/v1/models
```

Run a smoke test on 5 pairs per model, and write the results to a separate folder so they do not mix with the real runs:

```bash
python query_llms.py --config models.test.json --limit 5 --out responses-test
```

Then check:

1. `responses-test/responses-<alias>.csv` has 5 rows per model with a label in `A`, `B` or `BOTH`.
2. `responses-test/raw/<alias>.jsonl` shows how many attempts were needed and why the rejected ones failed.

If every attempt fails with an API error saying the model does not accept images, the model is text only and cannot be used for this experiment. If answers are often truncated, raise `--max-tokens` (reasoning models need room for their thinking).

When the test looks good, delete `responses-test` and run the full study:

```bash
python query_llms.py --config models.test.json
```

Run a single model of the config with `--only`:

```bash
python query_llms.py --config models.test.json --only Qwen3.8-27B
```

## 4. Output

```
experiments/experiment-2/responses/
├── responses-<alias>.csv     # id,model,response,motivation,reasoning
└── raw/<alias>.jsonl         # every attempt: raw answer, reasoning, validity, error
```

| Column | Content |
| --- | --- |
| `id` | Pair id (the PNG file name without extension), matching `id` in `data/SDPC/pairs.csv` |
| `model` | Model id as served by the endpoint, the same on every row |
| `response` | `A`, `B` or `BOTH` |
| `motivation` | The model's motivation |
| `reasoning` | The model's reasoning text, empty if the model does not reason |

`reasoning` is taken from the separate reasoning field returned by servers with a reasoning parser enabled (`reasoning_content` or `reasoning`) and from any `<think>...</think>` block in the answer. When both are present, they are joined.

The first columns match `responses-gpt4o.csv`. A CSV written by an earlier version of the script (`id,response,motivation`) is upgraded in place on the next run: `model` is filled in and `reasoning` is left empty (a `thinking` column, if present, is renamed to `reasoning`).

A row is written as soon as a valid answer arrives, so an interrupted run keeps everything completed so far.

## 5. Validation and retries

An answer is valid when:

* it is a JSON object, after removing `<think>...</think>` reasoning and Markdown code fences;
* it has the keys `plot` and `motivation` (key case is ignored);
* `plot` is `A`, `B` or `BOTH` (`Plot A` is also accepted);
* `motivation` is a non empty string.

By default, text around the JSON object is tolerated and the object is extracted. Use `--strict-json` to reject such answers.

Invalid answers and API errors (timeouts, 5xx, rate limits) are retried up to `--max-attempts` times per pair, with exponential backoff on API errors. A pair still without a valid answer is left out of the CSV and counted as `failed` in the progress bar.

## 6. Resuming

Ids already in a model's CSV are skipped, so running the same command again only queries the missing pairs, including those that failed. Repeat until the progress bar reports `0 to query`. Stopping with `Ctrl+C` is safe.

## 7. Progress bar

One bar per model, with elapsed time, ETA and throughput. The postfix shows:

* `retry_json`: answers rejected by validation and requested again;
* `api_err`: failed requests (network, server errors);
* `failed`: pairs that exhausted `--max-attempts`.

## 8. Options

| Option | Default | Description |
| --- | --- | --- |
| `--config FILE` | | JSON file with the models |
| `--base-url URL` | | OpenAI compatible endpoint, for example `http://host:8000/v1` |
| `--models M [M ...]` | | Model ids served at `--base-url` |
| `--only NAME [NAME ...]` | | Run only these models (alias or model id) |
| `--prompt FILE` | `prompt.md` | Prompt sent with every image |
| `--images DIR` | `../../data/SDPC/imgpairs-png` | Folder with the PNG pairs |
| `--out DIR` | `responses` | Output folder |
| `--limit N` | all | Query only the first N pairs (testing) |
| `--no-raw-log` | off | Do not write `raw/<alias>.jsonl` |
| `--concurrency N` | 20 | Simultaneous requests per model |
| `--max-attempts N` | 10 | Attempts per pair in one run |
| `--timeout S` | 600 | Request timeout in seconds |
| `--temperature T` | server default | Sampling temperature |
| `--max-tokens N` | server default | Maximum generated tokens |
| `--json-mode` | off | Send `response_format={"type": "json_object"}` (constrained decoding) |
| `--strict-json` | off | Reject text around the JSON object |
| `--system-prompt` | off | Send the prompt as a system message instead of user text |

For comparability across models, set `--temperature` (and `--max-tokens` if needed) explicitly and keep the same values for every model, or set them per model in the config's `extra` field. `--json-mode` guarantees well formed JSON on servers that support it, but constrained decoding may change the answers compared with the GPT-4o run, so it is off by default.
