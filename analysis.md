# Day 4 Task — Will It Fit, and May I Use It?

## Scenario

I want to run a small open-weight language model locally on my laptop as a personal AI assistant / coding helper. The machine has a **16 GB memory budget** for this exercise. The user is myself, so the scenario is personal/local rather than a public commercial service.

The decision has two parts: **fit** (memory) and **permission** (licence). The Day 4 lab formula is treated as an estimate rather than an exact prediction.

## 3.1 Explanation of the concepts

### Model weights

Model weights are the stored parameters of the model. Their memory requirement mainly depends on the number of parameters and the precision/quantization used to store them. More parameters normally mean more memory.

In my estimator, the weight component is calculated from parameter count multiplied by the bytes-per-parameter value for the selected precision. For example, my 8B Q4_K_M configuration uses about 4.56 GB of weights.

If I ignore weights, I could select a model that simply cannot be loaded. The limitation is that the estimate is not an exact runtime measurement because actual model files and runtimes have additional metadata and implementation details.

### Quantization

Quantization stores model parameters using fewer bits/bytes. This reduces the weight memory footprint, normally with some quality trade-off.

For the same 8B model at 8K context, my results were:

- Q3_K_M: 3.44 GB weights, 5.19 GB total
- Q4_K_M: 4.56 GB weights, 6.42 GB total
- Q5_K_M: 5.44 GB weights, 7.39 GB total
- Q8_0: 8.00 GB weights, 10.21 GB total
- FP16: 16.00 GB weights, 19.01 GB total

Therefore, quantization changes the **weights** substantially. Ignoring it could make a model look too large or too small. My limitation is that lower precision can reduce model quality, so the smallest memory number is not automatically the best choice.

### KV cache and context length

The KV cache stores attention information for the tokens currently represented in the context. Its memory requirement grows with context length.

My 8B Q4_K_M experiment shows:

- 4K context: 0.64 GB KV, 5.72 GB total
- 8K context: 1.28 GB KV, 6.42 GB total
- 32K context: 5.12 GB KV, 10.65 GB total
- 128K context: 20.48 GB KV, 27.54 GB total

The weights stayed at 4.56 GB while the KV cache grew. This matters for an agent because conversations and tool results can keep increasing the context. A model that fits at 4K or 8K can stop fitting at a very long context.

### Model card

A model card is the documentation page for a model. It gives information such as the model family, parameters, context length, intended use and licence.

I used model cards together with the Ollama library pages when comparing Qwen2.5, Llama 3.2 and Gemma 3. The model card controls the information I use for both fit and permission decisions. If I only looked at download size, I could miss a licence condition or a context/tool-use limitation.

### Open-weight vs open-source licensing

Open-weight means the trained weights are available, but it does not automatically mean the model is released under a standard open-source software licence. The exact licence and its conditions decide what uses and redistribution are allowed.

For this reason, licence checking is separate from memory checking. Qwen2.5 7B Instruct is listed under Apache License 2.0, while Llama 3.2 uses the Llama 3.2 Community License, and Gemma 3 uses the Gemma Terms of Use. These are different licence families and should not be treated as interchangeable.

## 3.2 Memory estimate table

**Available memory budget: 16 GB**

| Model/configuration | Params (B) | Precision | Context (K) | Weights (GB) | KV cache (GB) | Total (GB) | Fits in 16 GB? |
|---|---:|---|---:|---:|---:|---:|---|
| Qwen small | 1.5 | Q4_K_M | 8 | 0.85 | 0.24 | 1.20 | Yes |
| Granite / Qwen mid | 8.0 | Q4_K_M | 8 | 4.56 | 1.28 | 6.42 | Yes |
| Mid at FP16 | 8.0 | FP16 | 8 | 16.00 | 1.28 | 19.01 | No |
| Large local | 30.0 | Q4_K_M | 32 | 17.10 | 4.80 | 24.09 | No |
| Server class | 70.0 | Q4_K_M | 32 | 39.90 | 11.20 | 56.21 | No |

These are the figures produced by my estimator run. The estimator is a planning tool, not a prediction of exact runtime usage.

## Model comparison

| Basis | Qwen2.5-7B-Instruct | Llama-3.2-3B-Instruct | Gemma-3-4B-IT |
|---|---|---|---|
| Full model name/version | Qwen2.5-7B-Instruct | Llama-3.2-3B-Instruct | Gemma 3 4B instruction-tuned |
| Publisher | Alibaba Cloud / Qwen | Meta | Google DeepMind |
| Total / active parameters | 7.62B | 3B | 4.3B |
| Context window | Up to 128K | 128K in Ollama listing | 128K |
| Licence | Apache License 2.0 | Llama 3.2 Community License | Gemma Terms of Use |
| Commercial use allowed? | Yes, subject to Apache 2.0 terms | Intended for commercial and research use, subject to the custom licence and AUP | Use is governed by Gemma Terms of Use; check those terms for the exact deployment |
| Extra conditions | Standard Apache 2.0 conditions | Attribution/notice and acceptable-use/commercial terms apply | Google-specific Gemma terms apply |
| Tool calling stated on card/library? | Ollama page exposes `tools` support | Ollama page lists `tools` and describes tool use | Ollama page supports chat API; this comparison does not treat tool calling as established from the model card alone |
| GGUF / Ollama available? | Yes, Ollama Q4_K_M build | Yes, Ollama | Yes, Ollama Q4_K_M build |
| Q4 Ollama download size | 4.7 GB | 2.0 GB | 3.3 GB |
| My 8 GB-class memory estimate | About 6.1 GB at 8K using the estimator's approximate Q4 rate | About 3.7 GB at 8K using the same approximate method | About 4.9 GB at 8K using the same approximate method |
| Fits my 16 GB scenario? | Yes at a practical context | Yes | Yes |
| Date checked | 2026-09-30 | 2026-09-30 | 2026-09-30 |

### Sources checked

- Qwen2.5-7B-Instruct model card: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
- Qwen2.5 7B Ollama page: https://ollama.com/library/qwen2.5:7b-instruct
- Llama 3.2 3B Instruct model card: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
- Llama 3.2 Ollama page: https://ollama.com/library/llama3.2
- Gemma 3 4B model card: https://huggingface.co/google/gemma-3-4b-it
- Gemma 3 4B Ollama page: https://ollama.com/library/gemma3:4b

## 3.3 Context length and quantization observation

I used the 8B configuration for the controlled experiments.

### Context length — Q4_K_M fixed

| Setting | Value | Weights (GB) | KV cache (GB) | Total (GB) | Fits? |
|---|---:|---:|---:|---:|---|
| Context | 4K | 4.56 | 0.64 | 5.72 | Yes |
| Context | 8K | 4.56 | 1.28 | 6.42 | Yes |
| Context | 32K | 4.56 | 5.12 | 10.65 | Yes |
| Context | 128K | 4.56 | 20.48 | 27.54 | No |

### Quantization — 8K fixed

| Setting | Value | Weights (GB) | KV cache (GB) | Total (GB) | Fits? |
|---|---|---:|---:|---:|---|
| Quantization | Q3_K_M | 3.44 | 1.28 | 5.19 | Yes |
| Quantization | Q4_K_M | 4.56 | 1.28 | 6.42 | Yes |
| Quantization | Q5_K_M | 5.44 | 1.28 | 7.39 | Yes |
| Quantization | Q8_0 | 8.00 | 1.28 | 10.21 | Yes |
| Quantization | FP16 | 16.00 | 1.28 | 19.01 | No |

When context grew, the **KV cache** changed while the weights stayed fixed. When quantization changed, the **weights** changed while the KV cache stayed fixed. With a 16 GB memory budget, the largest tested context that fit for this 8B Q4_K_M configuration was **32K**; 128K did not fit.

I would choose **Q4_K_M** for this laptop scenario because it gives a useful balance between memory use and quality. Q3 saves more memory but gives up more precision; Q5/Q8 and FP16 use more memory. Q4 therefore provides a practical middle ground.

## 3.4 Estimate versus reality

I installed Qwen2.5-1.5B Q4_K_M locally using Ollama.

| Model | ollama list size | ollama ps size | Processor | Estimate |
|---|---:|---:|---|---|
| qwen2.5:1.5b | 986 MB | 1.2 GB | 100% GPU | 0.85 GB weights / 1.20 GB total |

My estimate was close to the runtime result. The installed model was listed as about 986 MB, while `ollama ps` reported about 1.2 GB while running. This difference is expected because the formula is an estimate and runtime memory can include overhead, buffers and other implementation details.

The `100% GPU` processor result shows that Ollama placed the running model on the GPU in this run. Therefore, the processor column is important when deciding whether a model will actually run in the intended local configuration.

## 3.5 Suitability analysis

### Recommendation: Qwen2.5-7B-Instruct, Q4_K_M

For my scenario I recommend **Qwen2.5-7B-Instruct in Q4_K_M**, using a practical context rather than trying to use the maximum 128K context all the time. Qwen2.5 supports long context and is available through Ollama, and the 7B Q4 build is about 4.7 GB in the Ollama listing. The Apache License 2.0 also makes the licence comparatively straightforward for a personal project, subject to its terms.

The runner-up is **Llama 3.2 3B Instruct**. It is much smaller and therefore attractive for limited hardware, and the Ollama page explicitly describes tool use. I rejected it as the primary choice because the 3B model gives up some model capacity compared with the 7B Qwen model, while my 16 GB scenario has enough memory for the larger model at a practical context.

My recommendation would change if the machine had much less memory, because then Llama 3.2 3B or an even smaller model would become more attractive. A requirement for a very long context would also change the decision because KV cache can become the dominant memory cost. Finally, a commercial/public release would make the exact licence conditions much more important than raw model size.

## 3.6 Conclusion

Model **size** should decide the choice when the machine has a strict memory limit. A larger model is not useful if it cannot be loaded or leaves too little memory for the runtime.

**Quantization** should carry more weight when a model is close to the memory limit. Moving from FP16 to Q4 can turn a model from a non-fitting configuration into a practical one, but lower precision can reduce quality.

**Context length** should carry more weight for agents, long conversations and workflows that accumulate tool results. The weights do not grow with context, but the KV cache does, so an otherwise fitting model can stop fitting at a very long context.

**Licence** should be the deciding factor when the project involves redistribution, a commercial product, a public service or other legally important deployment. Two models with similar memory requirements can have very different permissions and conditions.

The main lesson from this task is therefore: first check **memory fit**, then check **context and quantization trade-offs**, and always check the **exact licence** before deciding that an open-weight model is suitable.

## Submission checklist

- [x] Estimator script
- [x] Context-length experiment
- [x] Quantization experiment
- [x] Ollama installation/run evidence
- [x] `analysis.md`
- [x] `screenshots/` folder
- [ ] Add model-card/licence screenshots if your instructor strictly requires the web pages as images
- [ ] Push the complete folder to GitHub
