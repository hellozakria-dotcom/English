"""Re-rank dict/core.json lemmas: frequency of the lemma + all its forms, and British spellings
(-ise, -isation, -our, -tre, -lled…) ranked like their American form. Also records the zipf
frequency cut-off used for "common words" (YouGlish-style: words you hear often in real videos)."""
import json, os, re, collections, wordfreq
os.chdir(os.path.dirname(os.path.abspath(__file__)))
core = json.load(open('../dict/core.json'))
L, F = core['lemmas'], core['forms']
forms = collections.defaultdict(set)
for f, l in F.items(): forms[l].add(f)
def us(w):
    v = re.sub(r'is(e|ed|es|ing|ation|ations|er|able)$', r'iz\1', w)
    v = re.sub(r'our(s|ed|ing|able|ite|ful)?$', r'or\1', v) if not re.search(r'(four|pour|tour|your|hour|flour|sour|our|dour|contour|detour|amour|velour|troubadour)s?$', w) else v
    v = re.sub(r'tre(s)?$', r'ter\1', v); v = v.replace('ogue', 'og') if w.endswith('ogue') and w not in ('vogue', 'rogue', 'league') else v
    v = {'drily': 'dryly', 'millilitre': 'milliliter', 'litre': 'liter', 'grey': 'gray', 'programme': 'program', 'defence': 'defense', 'offence': 'offense', 'licence': 'license', 'cheque': 'check', 'jewellery': 'jewelry', 'tyre': 'tire', 'pyjamas': 'pajamas', 'plough': 'plow', 'mould': 'mold', 'aluminium': 'aluminum', 'sceptical': 'skeptical', 'manoeuvre': 'maneuver', 'enrol': 'enroll', 'fulfil': 'fulfill'}.get(w, v)
    return v
score = {}
for l in L:
    score[l] = wordfreq.word_frequency(l, 'en') + sum(wordfreq.word_frequency(f, 'en') for f in forms[l])
for l in L:
    v = us(l)
    if v != l:
        sv = wordfreq.word_frequency(v, 'en') + sum(wordfreq.word_frequency(f, 'en') for f in forms.get(v, ()))
        score[l] = max(score[l], sv)
new = sorted(L, key=lambda l: -score[l])
core['lemmas'] = new
core['zipf'] = [round(wordfreq.zipf_frequency(l, 'en') if score[l] <= 0 else 9 + __import__('math').log10(score[l]), 2) for l in new]
json.dump(core, open('../dict/core.json', 'w'), ensure_ascii=False, separators=(',', ':'))
cut = next(i for i, z in enumerate(core['zipf']) if z < 3.0)
print('re-ranked', len(new), 'first =', new.index('first') + 1, 'capitalise =', new.index('capitalise') + 1, '| zipf<3 starts at rank', cut + 1)
