"""Verb + noun collocation quiz items → content/colloc.json.
Distractors are common verbs that never occur with the noun anywhere in the corpora (Brown + Reuters),
so a wrong option is very unlikely to be an acceptable collocation. Run after build_dict.py (uses its cache)."""
import os, json, pickle, collections, random
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ['NLTK_DATA'] = os.path.abspath('data/nd')
from nltk.corpus import brown, wordnet as wn
PERSON = wn.synset('person.n.01'); GROUP = wn.synset('social_group.n.01')
def is_person(n):
    ss = wn.synsets(n, 'n')[:1]
    return any(PERSON in x.closure(lambda y: y.hypernyms()) or GROUP in x.closure(lambda y: y.hypernyms()) for x in ss)
random.seed(3)
core = json.load(open('../dict/core.json'))
RANK = {l: i + 1 for i, l in enumerate(core['lemmas'])}
FORMS = core['forms']
lem = lambda w: FORMS.get(w.lower(), w.lower())
UT = {'NOUN': 'n', 'VERB': 'v', 'ADJ': 'a', 'DET': 'd', 'PRON': 'o', 'ADP': 'p', 'PRT': 'p'}
sents = [[(w, UT.get(t, 'x')) for w, t in s] for s in brown.tagged_sents(tagset='universal')]
for toks, tg in pickle.load(open('data/reuters_tagged.pkl', 'rb')).values():
    sents.append([(w, UT.get(t, 'x')) for w, t in tg])
pair = collections.Counter(); det = collections.defaultdict(collections.Counter); anyv = collections.defaultdict(set)
for s in sents:
    ls = [(lem(w), t) for w, t in s]
    for i, (l, t) in enumerate(ls):
        if t != 'v': continue
        j, d = i + 1, ''
        while j < len(ls) and j <= i + 3 and ls[j][1] in ('d', 'a', 'o'):
            if ls[j][1] == 'd' and not d: d = ls[j][0]
            j += 1
        if j < len(ls) and j <= i + 4 and ls[j][1] == 'n':
            n = ls[j][0]; pair[(l, n)] += 1; anyv[n].add(l)
            det[(l, n)][d if d in ('a', 'an', 'the', '') else '~'] += 1
        for k in range(i + 1, min(len(ls), i + 6)):  # any later noun nearby counts as "seen together"
            if ls[k][1] == 'n': anyv[ls[k][0]].add(l)
VERBS = ['make', 'do', 'take', 'have', 'give', 'get', 'keep', 'pay', 'raise', 'reach', 'meet', 'set', 'run', 'hold', 'draw', 'play', 'catch', 'bear', 'break', 'carry', 'lose', 'miss', 'put', 'save', 'spend', 'tell', 'face', 'follow', 'gain', 'achieve', 'cause', 'pose', 'conduct', 'launch', 'reduce', 'provide', 'solve', 'win', 'join', 'lead', 'build', 'form', 'offer', 'cut', 'fill', 'sign', 'express', 'adopt', 'commit', 'deliver', 'hit', 'serve', 'strike', 'tackle', 'address', 'overcome', 'reach', 'attract', 'grant', 'impose', 'issue', 'lift', 'place', 'seize', 'waste', 'withdraw', 'bridge', 'close', 'open', 'shed', 'meet', 'fulfil', 'fulfill', 'undergo', 'exert', 'harbour', 'harbor', 'voice', 'wage', 'stage', 'cast', 'run', 'tell', 'answer', 'ask', 'seek', 'pass', 'earn']
VERBS = list(dict.fromkeys(VERBS))
items = []
for (v, n), c in pair.most_common():
    if v in ('tell', 'ask', 'have', 'get', 'answer', 'pass', 'run', 'join', 'lead', 'meet', 'follow', 'serve', 'hold', 'place') or is_person(n): continue
    if c < 4 or v not in VERBS or n not in RANK or RANK[n] < 400 or len(n) < 3: continue
    dist = [x for x in VERBS if x not in anyv[n] and x != v and x not in ('strike', 'make', 'do', 'take', 'set', 'close', 'open', 'hold', 'meet', 'reach', 'place', 'get', 'have')]
    if len(dist) < 3: continue
    d = det[(v, n)].most_common(1)[0][0]; d = '' if d == '~' else d
    if d == 'a' and n[0] in 'aeiou': d = 'an'
    items.append({'n': n, 'v': v, 'd': d, 'x': random.sample(dist, 3), 'c': c})
seen = set(); out = []
for it in items:
    if it['n'] in seen: continue
    seen.add(it['n']); out.append(it)
json.dump(out, open('../content/colloc.json', 'w'), separators=(',', ':'))
print(len(out), out[:8])
