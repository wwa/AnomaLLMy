# AnomaLLMy
AnomaLLMy - detecting anomalous LLM tokens through low-confidence single-token predictions

[AnomaLLMy paper](https://arxiv.org/abs/2406.19840)

# What does it do?
Anomalous tokens are tokens that are under-trained, sometimes hilariously so:
[SolidGoldMagikarp](https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation)


Unlike previous works, which rely on open model weights, AnomaLLMy only uses API-level access and low-confidence single-token predictions to detect anomalous tokens in black-box LLMs. 

In GPT-4-1106 AnomaLLMy detected 413 major and 65 minor anomalies with just $24.39 spent in API credits.

# How does it work?
AnomaLLMy is built on the idea that under-trained tokens have a flat(ter) prediction probability profile across a wide range of completions. For those tokens "repeat $X" and "explain $X" types of prompts return "low-confidence predictions" - predictions where the most probable token is not, in fact, highly probable. 

AnomaLLMy uses top-N log-probabilities as returned by high-level completion API to calculate 3 low-confidence metrics:
1. High entropy
2. High tail proability
3. Small difference between top1 and top2 prediction

# Why?
It was fun! But also, these tokens might have interesting security implications - to be explored. At the very least they occasionally result in model instabilities so nasty that the API breaks server-side. I've seen JSON schema violations, a few 400:BadRequests with error messages along the lines of "model does not exist" and in rare cases no results at all - an empty array of log-probabilities, which should not be possible with a Transformer.

# Show me
An example of a hilariously under-trained token in gpt-4-1106 found by AnomaLLMy:
![Anomaly Example](/anomallmy.png?raw=true "Anomaly example")
