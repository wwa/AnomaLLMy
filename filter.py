import json
import os
import numpy
import tiktoken
import pandas as pd
from scipy.stats import entropy
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
client  = OpenAI()
encoder = tiktoken.get_encoding("cl100k_base")

def metrics():
  ifile = open("./data/cl100k_logprobes.txt","r", encoding="utf-8")
  ofile = open("./data/metrics.txt",   "w", encoding="utf-8")
  for line in ifile:
    data = eval(line.strip())
    tok  = list(data.keys())[0]
    if len(data[tok])==0:
      # completion missing entirely, these are super-anomalies 
      #   .. but filter them out to avoid special cases later.
      print(f"{tok}, no completion")
      continue
    logp = data[tok]
    if len(logp)==0:
      continue
    comp = list(logp[0].items())[0][0]
    toks = encoder.decode([tok])
    if not toks.strip():
      continue
    logps = [list(logp[i].items())[0][1] for i in range(0,5)]
    probs = numpy.exp(logps)
    tail  = numpy.abs(1.0 - numpy.sum(probs)) # numerical errors...
    probs = numpy.append(probs, tail)
    ofile.write(f"[{tok}, {comp}, {entropy(probs)}, {probs[0]}, {probs[0]-probs[1]}, {tail}]\n")

def filter():
  ifile = open("./data/metrics.txt",    "r", encoding="utf-8")
  ofile = open("./data/candidates.txt", "w", encoding="utf-8")
  for line in ifile:
    tokn, comp, entr, prob, diff, tail = eval(line.strip())
    toks = repr(encoder.decode([tokn]))
    pad  = " "*(30-len(toks))
    type = "'normal' "
    if tail > 0.1:
      type = "'bigtail'"
    elif entr > 1.0:
      type = "'entropy'"
    elif diff < 0.2:
      type = "'lowdiff'"
    elif comp == 39818:
      # 39818 == 'Repeat'; Model repeats system prompt because it thinks user prompt's empty.
      type = "'clarify'"
    ofile.write(f"[{type}, {toks},{pad}{tokn}, {comp}, {entr}, {prob}, {diff}, {tail}]\n")

def repeat(tok, retry=10, probnum=1, toknum=1):
  txt = encoder.decode([tok])
  cnt = {}
  while retry>0:
    res = client.chat.completions.create(
      model        = "gpt-4-1106-preview",
      messages     = [{"role":"system","content":"Repeat user message exactly"},{"role": "user","content": txt}],
      max_tokens   = toknum,
      logprobs     = True,
      top_logprobs = probnum,
    )
    try:
      retry -= 1
      pred = encoder.encode(res.choices[0].logprobs.content[0].top_logprobs[0].token)[0]
      cnt[pred] = cnt.get(pred,0)+1
    except Exception as e:
      print(f"repeat: oops at {tok}: {e}")
      cnt[-1]  = cnt.get(-1,0)+1
  return cnt

def verify(start=0):
  ifile = open("./data/candidates.txt", "r", encoding="utf-8")
  ofile = open("./data/anomalies.txt",  "a", encoding="utf-8")
  for line in ifile:
    type, _, tokn, comp, entr, prob, diff, tail = eval(line.strip())
    if tokn < start:
      continue
    if type != 'clarify':
      continue
    if type == 'normal':
      continue
    txt = str({tokn:repeat(tokn)})
    print(txt)
    ofile.write(txt+"\n")
    
def classify():
  ifile = open("./data/anomalies.txt",      "r", encoding="utf-8")
  ofile = open("./data/classification.txt", "w", encoding="utf-8")
  false = 0
  major = 0
  minor = 0
  for line in ifile:
    tokn, dict = list(eval(line.strip()).items())[0]
    toks = encoder.decode([tokn])
    tokl = toks.strip().lower()
    bad  = 0
    good = 0
    text = ""
    for k,v in dict.items():
      if k == -1:
        continue
      coms = encoder.decode([k])
      text += ","+repr(coms)
      coml = coms.strip().lower()
      if tokl.startswith(coml):
        good += v
      else:
        bad  += v
    type = "false"
    if -1 in dict.keys():
      type = "major"
      major += 1
    elif bad >= good:
      type = "major"
      major += 1
    elif bad > 0:
      type = "minor"
      minor += 1
    else:
      false += 1
    ofile.write(f"{type}, {tokn}, {repr(toks)} {text}\n")
  print(f"major={major},minor={minor},false={false}")

#calculate()
#filter()
#verify()
classify()