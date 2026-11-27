"""Merge the hand-written entries in content/curated/*.txt into dict/d_*.json.
Line format:  word | part of speech | Arabic | definition | example 1 | example 2
A line starting with "+" adds another meaning to the word above. Curated meanings replace
the automatic ones; British spellings get a copy of the American entry."""
import os, json, glob, re
os.chdir(os.path.dirname(os.path.abspath(__file__)))
entries, order = {}, []
for f in sorted(glob.glob('../content/curated/*.txt')):
    last = None
    for n, line in enumerate(open(f, encoding='utf8'), 1):
        line = line.strip()
        if not line or line.startswith('#'): continue
        parts = [p.strip() for p in line.lstrip('+').split('|')]
        if line.startswith('+'):
            assert last, f'{f}:{n} "+" without a word'
            p, ar, g, *ex = parts; w = last
        else:
            w, p, ar, g, *ex = parts; last = w
            if w not in entries: entries[w] = []; order.append(w)
        sd = {'p': p, 'ar': [a.strip() for a in re.split('[،,]', ar) if a.strip()] if '؛' not in ar else [a.strip() for a in ar.split('؛') if a.strip()], 'g': g, 'c': 1}
        if ex: sd['ex'] = ex[0]
        if len(ex) > 1: sd['ex2'] = ex[1]
        entries[w].append(sd)
BRIT = {'analyze': 'analyse', 'labor': 'labour', 'license': 'licence', 'utilize': 'utilise', 'maximize': 'maximise', 'minimize': 'minimise', 'skeptical': 'sceptical', 'behavior': 'behaviour'}
shards = {}
def shard(k):
    c = k[0] if k[0].isalpha() else '_'
    if c not in shards: shards[c] = json.load(open(f'../dict/d_{c}.json'))
    return shards[c]
n_new = 0
for w in order:
    for key in [w] + ([BRIT[w]] if w in BRIT else []):
        d = shard(key); e = d.get(key)
        if e is None: e = d[key] = {'reg': ['neutral']}; n_new += 1
        e['m'] = entries[w]; e['cur'] = 1
        e['pos'] = list(dict.fromkeys(sd['p'] for sd in entries[w]))
        e.pop('ar', None)
for c, d in shards.items():
    json.dump(d, open(f'../dict/d_{c}.json', 'w'), ensure_ascii=False, separators=(',', ':'))
core = json.load(open('../dict/core.json'))
core['cur'] = sorted(set(order) | {BRIT[w] for w in order if w in BRIT})
json.dump(core, open('../dict/core.json', 'w'), ensure_ascii=False, separators=(',', ':'))
print('curated words', len(order), 'meanings', sum(len(v) for v in entries.values()), 'new entries', n_new)
