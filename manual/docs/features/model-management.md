---
sidebar_position: 1
title: Model Management
---

# Model Management

Everything starts with a loaded model. miServer downloads models from HuggingFace (or imports from a local path), loads one model at a time onto the GPU, and optionally quantizes it as it loads.

![miServer Models Page](/img/miServer_Models_01.jpg)

## Downloading a Model

1. Navigate to **Models** in the sidebar
2. Enter a HuggingFace repository ID (e.g., `google/gemma-2-2b`) — or choose **Local Path** to import weights already on disk
3. Click **Preview** to see the model's size, architecture, and estimated memory per quantization before committing
4. Select **Quantization**:

| Mode | Bits | VRAM Savings | Quality | Best For |
|------|------|-------------|---------|----------|
| **FP16** | 16 | Baseline | Maximum | Precision research; enables `torch.compile` |
| **Q8** | 8 | ~50% | Minimal loss | Good balance |
| **Q4** | 4 | ~75% | Moderate loss | Consumer GPUs |
| **Q2** | 2 | ~87% | Significant loss | Maximum compression |

:::warning Q2 needs a pre-quantized checkpoint or a GGUF file
bitsandbytes has no 2-bit mode. A Q2 transformers checkpoint that is not already quantized (one whose `config.json` carries no `quantization_config`) is **refused at load** with `UNSUPPORTED_QUANTIZATION`, before the served model is unloaded — it would otherwise load unquantized in bfloat16, eight times the memory its Q2 estimate assumes. Choose Q4, Q8 or FP16 for it, or serve a Q2 GGUF of the model.
:::

5. Optionally enter a **HuggingFace Token** for gated models (Gemma and Llama are gated — accept the license on the model page first)
6. Check **Trust Remote Code** only if the model requires custom code (explicit opt-in, per download)
7. Click **Download & Load Model**

Quantization happens **at load time**, not at download time. Whatever quantization you pick, the download stores the repository's checkpoint exactly as it was published. Then, on every load, bitsandbytes quantizes each weight as it is placed on the GPU. So picking `Q4` does not make the download smaller. Downloading the same repository as both `FP16` and `Q4` stores the same files twice, once in each model's cache directory. Download progress streams over WebSocket to the UI; downloads can be cancelled but not paused.

:::tip Choosing quantization for steering work
Prefer **FP16** for a model that fits: quantized (bitsandbytes) models cannot use `torch.compile`, so FP16 decodes faster on capable GPUs despite the extra memory. See [Hardware Requirements](/getting-started/hardware) for sizing tables.
:::

## GGUF models and quantization

miServer serves **GGUF** files the way Ollama does, alongside HuggingFace checkpoints. A GGUF repository usually publishes several quantizations of the same weights, and **Preview** lists each one with its file size so you can pick before downloading.

The distinction that matters against the table above: bitsandbytes quantizes a full-precision checkpoint *every time it loads*, while a GGUF file was quantized ahead of time by whoever published it. What you choose is a file, not a mode — `IQ4_XS`, `Q4_K_M`, `Q5_K_M` and so on.

### Several quantizations of one repository can coexist

They are different models with different sizes and different quality, so miServer keeps them apart by naming a GGUF model **`repo:QUANT`** — the convention Ollama already established:

```
gemma-4-31b-it-3MPER0RR-abliterated-GGUF:IQ4_XS
gemma-4-31b-it-3MPER0RR-abliterated-GGUF:Q4_K_M
```

Both appear separately in `/v1/models` instead of one shadowing the other. A **custom name** you supply is left alone — that is your choice, not a derived one.

A **bare** repository name still resolves while exactly one quantization of it exists, so existing scripts and saved client selections keep working. Once a second is downloaded the bare name becomes ambiguous, and miServer returns a **400** naming the tags that exist rather than picking one:

```json
{ "error": { "code": "AMBIGUOUS_MODEL_NAME",
             "message": "'…-GGUF' matches 2 quantizations: …:IQ4_XS, …:Q4_K_M. Name one of them exactly." } }
```

Choosing silently would make the served model depend on which was downloaded first, with nothing on the wire to say which answered.

### Every precision of a repository has its own name

The same rule holds for HuggingFace checkpoints. The native-precision (FP16) registration keeps the bare repository name; a registration at any other precision is tagged with it — `repo:Q8`, `repo:Q4`, `repo:FP32`:

```
jevify-gemma4-e4b        # FP16 (served in the checkpoint's native dtype)
jevify-gemma4-e4b:Q8     # the same weights, quantized by bitsandbytes at load
```

No two models share a name (the database enforces it). If a model registered before this rule already holds the bare name, a new registration is tagged instead, so a name your clients already use never moves; a **custom name** that is already taken is refused with `409 MODEL_NAME_TAKEN`.

### Context window

A GGUF model's context is **derived, not configured**: miServer reads what the file declares, predicts what free VRAM can hold, and loads at the largest window that fits under a **ceiling**. See [Hardware Requirements](/getting-started/hardware#gguf-context-windows-are-derived-not-guessed) for the arithmetic and the KV-cache lever that doubles it.

The ceiling comes from the first of these that is set:

| Source | How to set it | Applies to |
|---|---|---|
| The load request | `POST /api/models/{id}/load` with `{"context_ceiling": 131072}` or `{"context_ceiling": "max"}` | that load only |
| The model | `PUT /api/models/{id}/context-ceiling` with `{"context_ceiling": 131072 \| "max" \| null}` | every later load of the model, including the automatic loads a `/v1` request makes |
| The server | [`GGUF_CONTEXT_LENGTH`](/reference/configuration#gguf-models) (default 32,768) | every GGUF model with neither |

**In the admin UI**, open a GGUF model from the **Downloaded Models** list. Its **Context window** section shows the ceiling, and for the loaded model the window it got, its key-value (KV) cache type and why. Choose **Server default**, **A number of tokens** or **As large as fits**, then **Save for this model** (it applies from the model's next load) or **Load once with this ceiling** (this load only; the model's setting is unchanged). On the loaded model the button is **Reload with this ceiling**, which unloads the model and loads it again with that ceiling (see [Loading & Unloading](#loading--unloading)).

`"max"` (or `0`) asks for **as large a window as fits**, up to the length the model declares: miServer starts at the window it predicts for the card the model is placed on, walks down if that does not load, and, if it does, tries up to three longer windows (each attempt loads the model again, so a "max" load takes longer). It does not spread the model over more cards to get a longer window; ask for `gpu: "all"` for that. A ceiling is refused (`422`) on a model that is not GGUF, below 2,048, or above the declared length.

**What "fits" means.** A window fits only when the model can *serve* in it, not merely when llama.cpp can create it:

- **Headroom.** Every card the model is on must keep [`GGUF_CONTEXT_HEADROOM_MB`](/reference/configuration#gguf-models) (default 1,024 MiB) free after the context is created. llama.cpp reserves its main buffers at load, but it still allocates memory while it decodes, and if that fails the whole server process stops. This applies to every GGUF window, not only "max".
- **The decode check.** A window above the predicted one is first tried in a separate, short-lived process. That process loads the model at that window, runs one full batch of 512 throwaway tokens and one more token, and the card's free memory is read while it holds them. If it crashes, or the card would keep less than the headroom, only that process ends: the load steps down to a smaller window. The served model never sees the throwaway tokens.
- **The guarantee.** "max" never serves a window above the prediction unless that window passed the decode check with the headroom kept. If no longer window passes, it serves the predicted window. With [`GGUF_DECODE_CHECK=off`](/reference/configuration#gguf-models), "max" serves the predicted window and tries nothing above it.

Each check is one more full load of the model, in the other process. On 2026-11-21, "max" took model 17 (Qwen3.6-27B Q4_K_M) to 222,208 tokens on a 24 GB card with almost nothing left free, and the first longer prompt crashed the server. From that load's own figures, the same load should now settle at about 172,000 tokens with about 1.8 GB free (a projection, to be confirmed on the card).

The card is shared. The headroom protects the model against its own decoding, not against another program (for example a miWorkbench job) that takes the free memory later.

An AI agent's load waits for your approval. The card shows **Context ceiling** with its value and where it came from, and the request digest covers it. For the loaded model, the card's **Reloads** line gives the window it has now and the ceiling it will be loaded with after approval. An agent's change to a model's ceiling waits too (**Change a model's context ceiling**, approvable with "Approve all settings changes").

Once loaded, the model record shows `context_length` (the window the model got), `kv_cache_type` (the cache it was created with) and `gguf_load`, which says why: the ceiling and its source, the declared length, the prediction, every attempt, and why the cache type and embeddings are what they are. `gguf_load.headroom.free_mb_after_load` gives the memory left free on each card once the model loaded, and `gguf_load.decode_check` gives the checks run (`trials`, each with its window, exit code and free memory), how many failed (`step_downs`), and whether the served window passed one (`ok`: `true`, or `null` when no check covered it).

### Embeddings

GGUF models are loaded with embedding output enabled so `/v1/embeddings` works without a second load, at a measured ~6.7% generation cost (113.5 → 105.9 tok/s on `zora-v1.13 Q5_K_M`). Set [`GGUF_ENABLE_EMBEDDINGS=false`](/reference/configuration#gguf-models) to buy that back on a deployment that never embeds.

Some models **cannot be loaded with embeddings** at any context length, and fail context creation at every length rather than reporting anything useful. For a hybrid model (Qwen3.5 and Qwen3.6, which hold a recurrent state in three layers of four) the reason is known in advance: with embeddings on, llama-cpp-python sizes that state for 256 parallel sequences — 37 GiB for Qwen3.6-27B — so miServer loads such a model without embeddings from the start when the card cannot hold it, and says why in `gguf_load.embeddings_reason`.

When a load fails at every length for any other reason, miServer finds the cause before changing anything else: it tries the smallest length once with embeddings off and the cache unchanged, then once with an `f16` cache and embeddings as configured, then once with both. The first that loads names the cause, and only that capability is given up. (Before 2026-11-21 the cache was switched to `f16` first, so a model whose only problem was embeddings ended up on an `f16` cache that holds half the context.) The record's `supports_embeddings: false` tells a caller up front instead of leaving it to a confusing failure at `/v1/embeddings`. Serving the model matters more than embedding it.

## Loading & Unloading

One model is resident on the GPU at a time. Clicking **Load** on another ready model unloads the current one first. Before loading, miServer estimates the memory requirement and warns if it exceeds free VRAM.

**Loading the model that is already loaded** (since 2026-11-21):

- If the load would change its context ceiling (see [Context window](#context-window)) or its graphics card (`gpu` names another card, or `"all"` for a model on one card), it is a **reload**. miServer unloads the model, giving running requests time to finish as an unload does, and then loads it again with the new value. One request does it, and an agent's request needs one approval. To apply a ceiling you have just saved for the loaded model, load it once more.
- If it would change nothing (the same ceiling and no other card; leaving out `gpu`, or sending `"auto"`, keeps the model where it is), it is refused with `409 MODEL_ALREADY_LOADED`. `details.fix` says what to send instead. An agent's request is refused before an approval is filed.

In the admin UI, the loaded model's **Context window** section offers **Reload with this ceiling**. The button is disabled when the model is already loaded with the chosen ceiling.

A reload is an unload followed by a load:

- its lease ends;
- its armed probes are disarmed;
- its steering lock is released;
- its attached SAEs are detached and its active circuits and profiles deactivated (below);
- chat requests wait until the model is back.

**Every unload detaches the attached sparse autoencoders (SAEs) and deactivates what steers through them** (since 2026-11-21). This holds for an explicit unload, a switch to another model, a `/v1` request that loads another model, a reload, and the weights API's full reload. After running requests finish and before the weights are released, miServer:

1. deactivates every active circuit (its steering stops and its layer claims are released);
2. deactivates the active profile or cluster;
3. detaches every attached SAE, as **Detach** does: the hook is removed, its GPU memory is freed, and the SAE reads `cached`.

**Nothing is attached or activated again by the next load, a reload of the same model included.** Attach the SAEs and activate the circuits and profiles again after the load. An approval card for an unload, a switch or a reload names each SAE, circuit and profile it takes, by name and id. A restart ends them the same way. (Before this change, an unload left the SAEs listed as attached, holding their memory, with hooks on the unloaded model that steered nothing.)

Unloading is **graceful**: in-flight inference requests get up to `GRACEFUL_UNLOAD_TIMEOUT` (default 30 s) to complete before the model is released.

To load a model automatically at server startup, set `AUTO_LOAD_MODEL` — see [Configuration](/reference/configuration).

:::warning Hybrid Models (Mamba/SSM)
Models with Mamba/SSM layers (e.g., `granite-4.0-h-*`) require the `mamba-ssm` package for efficient inference. Without it, the naive fallback creates massive intermediate tensors that cause OOM errors. miServer automatically selects the hybrid KV-cache these architectures need.
:::

## Model Locking

When an SAE is attached, the model is automatically **locked**, so a `/v1` request naming another model cannot auto-load over it (`409 model_locked`). Detaching the SAE unlocks the model automatically; you can also lock/unlock manually from the model details or via [`POST /api/models/{id}/lock`](/api/models). The lock icon's tooltip reads "Locked for steering".

⚠ The management **Load**, **Switch** and **Unload** actions do not read the lock today — a steering model can be unloaded from here. This is tracked debt; the lease below guards every path.

## Model Lease

A caller such as a long labelling job can **lease** the resident model: while the lease is live, nobody else can load another model, unload it, or swap it — from this page, the API, or a `/v1` request — and the refusal (`409 MODEL_LEASED`) names who holds it and until when. Requests that *use* the leased model keep working.

A leased model shows an amber key badge beside the lock icon, on the loaded-model card and in the model details: **"Leased by midataworks · expires in 1h 12m"**, with the reason and exact expiry in the tooltip. The badge counts down and disappears when the lease expires; the page refreshes every 10 seconds while a lease exists.

The Admin UI only displays leases — it cannot take, renew or release one, and there is no force-release. A lease lasts at most 2 hours unless its holder renews it. **A restart of miServer ends every lease**; holders take a new one once their model is resident. See [Model lease](/api/models#model-lease).

## Deleting

Delete removes the model from disk and the registry (hard delete). A loaded or locked model must be unloaded/unlocked first.

:::info Dynamic Architecture Support
miServer uses dynamic layer discovery to support any transformer architecture — Llama, Gemma, GPT-2, LFM, Granite, Mistral, Phi, and more. No configuration needed. When you attach an SAE, the attach response reports the exact module hooked (`layer_module_path`) so you can verify layer resolution on unusual architectures.
:::

## API

All of the above is scriptable — see the [Models API reference](/api/models). The model list, download, load/unload, lock/unlock, preview, and delete operations map 1:1 to endpoints.
