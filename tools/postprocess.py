"""Post-process dict/d_*.json after build_dict.py:
1. clean broken diacritics (two short vowels on one letter → drop the harakat on that word);
2. add the most common WordNet meanings for a part of speech that the Wiktionary translation table
   doesn't cover (e.g. "run" as a noun), marked as English-only, so every example has a matching meaning."""
import os, re, json, collections
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ['NLTK_DATA'] = os.path.abspath('data/nd')
from nltk.corpus import wordnet as wn
V = 'ًٌٍَُِْ'
BAD = re.compile('[' + V + ']{2,}|ّ{2,}|[' + V + ']ّ[' + V + ']')
HAR = re.compile('[ً-ْ]')
PN = {'n': 'noun', 'v': 'verb', 'a': 'adjective', 's': 'adjective', 'r': 'adverb'}
def clean(a): return ' '.join(HAR.sub('', w) if BAD.search(w) else w for w in a.split(' '))
fixed = added = 0
for f in sorted(os.listdir('../dict')):
    if not f.startswith('d_'): continue
    d = json.load(open('../dict/' + f))
    for k, e in d.items():
        for sd in e.get('m', []):
            if 'ar' in sd:
                new = [clean(a) for a in sd['ar']]
                if new != sd['ar']: fixed += 1; sd['ar'] = new
        if 'ar' in e and e['ar']: e['ar'] = [clean(a) for a in e['ar']]
        m = e.get('m', [])
        if not m or not any('ar' in sd for sd in m): continue
        have = {sd['p'] for sd in m}
        syns = wn.synsets(k.replace(' ', '_'))
        cnt = collections.Counter()
        for s in syns: cnt[PN[s.pos()]] += sum(l.count() for l in s.lemmas() if l.name().lower() == k.replace(' ', '_'))
        xpos = {x[2] for x in e.get('x', []) if len(x) > 2}
        for p, c in cnt.most_common():
            if p in have: continue
            letter = {'noun': 'n', 'verb': 'v', 'adjective': 'a', 'adverb': 'r'}[p]
            if c < 3 and letter not in xpos: continue
            s = next(s for s in syns if PN[s.pos()] == p)
            sd = {'p': p, 'g': s.definition(), 'en': 1}
            stem = k.split()[0][:max(3, len(k.split()[0]) - 2)]
            ex = [x for x in s.examples() if re.search(r'\b' + re.escape(stem), x, re.I)]
            if ex: sd['ex'] = ex[0]
            m.append(sd); have.add(p); added += 1
            if len(m) >= 6: break
    json.dump(d, open('../dict/' + f, 'w'), ensure_ascii=False, separators=(',', ':'))
print('diacritics cleaned', fixed, '· English-only meanings added', added)
