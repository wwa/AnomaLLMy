import pandas as pd
import tiktoken
import json
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
client  = OpenAI()
encoder = tiktoken.get_encoding("cl100k_base")
def getLogprobes(prompt, probnum=5, toknum=1):
  res = client.chat.completions.create(
    model        = "gpt-4-1106-preview",
    messages     = [{"role":"system","content":"Repeat user message exactly"},{"role": "user","content": prompt}],
    max_tokens   = toknum,
    logprobs     = True,
    top_logprobs = probnum,
    temperature  = 0.0,
  )
  try:
    return [{encoder.encode(r.token)[0]: r.logprob} for r in res.choices[0].logprobs.content[0].top_logprobs]
  except IndexError:
    return []

def scan(start=0):
  for idx, k in enumerate(encoder._mergeable_ranks):
    if idx < start: continue
    val = encoder._mergeable_ranks[k]
    tok = encoder.decode([val])
    print({val:getLogprobes(tok)})

scan()