import numpy

data = []
for line in open("./data/metrics.txt","r", encoding="utf-8"):
  data.append(eval(line.strip()))

entrp = [d[2] for d in data]
probs = [d[3] for d in data]
diffs = [d[4] for d in data]
tails = [d[5] for d in data]

print(f"entrp: avg={numpy.average(entrp)}, min={numpy.min(entrp)}, max={numpy.max(entrp)}")
print(f"top:   avg={numpy.average(probs)}, min={numpy.min(probs)}, max={numpy.max(probs)}")
print(f"diff:  avg={numpy.average(diffs)}, min={numpy.min(diffs)}, max={numpy.max(diffs)}")
print(f"tail:  avg={numpy.average(tails)}, min={numpy.min(tails)}, max={numpy.max(tails)}")
