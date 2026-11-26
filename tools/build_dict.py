"""Run tools/setup.sh once (downloads the sources into tools/data/), then: python3 tools/build_dict.py
Build the offline English→Arabic learner dictionary for the English PWA.
Sources: Princeton WordNet 3.0 + Arabic WordNet (OMW 1.4), FreeDict/Arabeyes eng-ara,
wordfreq (frequency), ipa-dict (IPA), spaCy lookups (lemmas), Brown + Reuters corpora
(usage examples, collocations, register).
Output: ../app/dict/*.json
"""
import json, gzip, re, os, math, random, collections, xml.etree.ElementTree as ET
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ['NLTK_DATA'] = os.path.abspath('data/nd')
import nltk
from nltk.corpus import wordnet as wn, brown, reuters
import wordfreq

OUT = os.path.abspath('../dict'); os.makedirs(OUT, exist_ok=True)
random.seed(7)
W = re.compile(r"^[a-z]+(?:-[a-z]+)*$")

# ---------------- lemma lookup ----------------
import spacy_lookups_data
SP = json.load(gzip.open(os.path.join(os.path.dirname(spacy_lookups_data.__file__), 'data', 'en_lemma_lookup.json.gz')))
KEEP_OWN = set('data media better best news glasses goods arms clothes means series species politics physics economics mathematics statistics ethics aids people police cattle criteria phenomena outskirts surroundings belongings earnings savings premises headquarters thanks remains left lying fewer less least more most others ours yours theirs hers its whose whom us them him her me my your our their these those an'.split())
FIX = {'an': 'a', 'was': 'be', 'were': 'be', 'is': 'be', 'am': 'be', 'are': 'be', 'been': 'be', 'being': 'be', "'s": 'be', 'has': 'have', 'had': 'have', 'having': 'have', 'does': 'do', 'did': 'do', 'done': 'do', 'doing': 'do',
       'better': 'good', 'best': 'good', 'worse': 'bad', 'worst': 'bad', 'lying': 'lie', 'left': 'leave', 'people': 'people', 'data': 'data', 'media': 'media',
       'children': 'child', 'men': 'man', 'women': 'woman', 'feet': 'foot', 'teeth': 'tooth', 'mice': 'mouse', 'geese': 'goose', 'criteria': 'criterion', 'phenomena': 'phenomenon', 'analyses': 'analysis', 'hypotheses': 'hypothesis', 'theses': 'thesis', 'crises': 'crisis',
       'us': 'we', 'them': 'they', 'him': 'he', 'her': 'she', 'me': 'i', 'my': 'i', 'your': 'you', 'our': 'we', 'their': 'they', 'his': 'he', 'its': 'it', 'these': 'this', 'those': 'that', 'mine': 'i', 'yours': 'you', 'ours': 'we', 'theirs': 'they', 'hers': 'she'}
# "left/better/best" are mapped to the verb/adjective, but the reader also offers the form itself.
for k in ('better', 'best', 'left', 'lying'): FIX.pop(k, None)
FIX['better'] = 'good'; FIX['best'] = 'good'

LL = json.load(open('data/langle-en/lemmas.json'))
KEEP_OWN |= {'agenda', 'kinda', 'gonna', 'wanna', 'gotta'}
def lem(w):
    w = w.lower()
    if w in FIX: return FIX[w]
    if w in KEEP_OWN: return w
    v = LL.get(w)
    if isinstance(v, list): v = v[0]
    if v and W.match(v): return v
    return SP.get(w, w)
ENAR = {**json.load(open('data/langle-en-ar/dict-1.json')), **json.load(open('data/langle-en-ar/dict-2.json'))}

# ---------------- frequency → lemma ranks ----------------
forms = wordfreq.top_n_list('en', 120000)
FUNC_EXTRA = set("the a of and to in that it for you he she they we i with as at by from this but not or his her their our your my its me him them us what which who whom whose there here when where why how if than then so because while although though unless whether until since nor yet also very too only just even each every either neither both all any some no none many much more most few less least such own same other another one two three four five six seven eight nine ten hundred thousand million billion first second third into onto upon about above below over under between among through during before after against without within across along around behind beyond toward towards off out up down near via per despite except like unlike would could should might must shall will can may ought do be have get".split())

def is_name(l):
    syn = wn.synsets(l)
    return bool(syn) and all(s.instance_hypernyms() for s in syn)

# Brown capitalisation (mid-sentence) to drop proper names
cap = collections.Counter(); low = collections.Counter()
for s in brown.sents():
    for t in s[1:]:
        if t.isalpha():
            (cap if t[0].isupper() else low)[t.lower()] += 1

lf = collections.Counter(); lemma_forms = collections.defaultdict(set)
for w in forms:
    if not W.match(w): continue
    l = lem(w)
    if not W.match(l): continue
    f = wordfreq.word_frequency(w, 'en')
    lf[l] += f; lemma_forms[l].add(w)

def accepted(l):
    if l in FUNC_EXTRA: return True
    if len(l) < 2 and l not in ('a', 'i'): return False
    in_ipa = l in IPA_UK or l in IPA_US
    if len(l) <= 3 and low[l] < 5 and l not in ENAR and wordfreq.zipf_frequency(l, 'en') < 3.5: return False
    if l not in ENAR and not in_ipa: return False
    has_wn = bool(wn.synsets(l.replace('-', '_')) or wn.synsets(l))
    if not has_wn and not (l in ENAR and in_ipa): return False
    if has_wn and is_name(l): return False
    if l not in ENAR and cap[l] > 3 * max(1, low[l]) and cap[l] >= 3: return False   # mostly capitalised = a name
    return True


# ---------------- IPA ----------------
def load_ipa(fn):
    d = {}
    for line in open(fn, encoding='utf8'):
        w, p = line.rstrip('\n').split('\t'); d[w.lower()] = p.split(', ')[0].strip('/')
    return d
IPA_UK, IPA_US = load_ipa('data/ipa_en_UK.txt'), load_ipa('data/ipa_en_US.txt')
VOW = 'aeiouæɑɒɔəɛɜɪʊʌɐɚɝiː'
ONSETS = {'pr', 'tr', 'kr', 'br', 'dr', 'ɡr', 'fr', 'θr', 'ʃr', 'pl', 'kl', 'bl', 'ɡl', 'fl', 'sl', 'sp', 'st', 'sk', 'sm', 'sn', 'sw', 'tw', 'kw', 'dw', 'spr', 'str', 'skr', 'spl', 'tʃ', 'dʒ', 'pj', 'kj', 'bj', 'fj', 'mj', 'nj', 'hj', 'vj', 'tj', 'dj', 'sj'}
def fix_stress(p):
    # espeak puts ˈ/ˌ just before the vowel; move it before the syllable onset
    out = p
    for mark in 'ˈˌ':
        res = ''; i = 0; s = out
        while i < len(s):
            if s[i] == mark:
                j = len(res)
                k = j
                while k > 0 and res[k - 1] not in VOW + 'ˈˌː' and res[k - 1] not in 'ɪʊ': k -= 1
                cons = res[k:j]
                best = ''
                for n in range(min(3, len(cons)), 0, -1):
                    c = cons[-n:]
                    if n == 1 or c in ONSETS: best = c; break
                res = res[:j - len(best)] + mark + best
                i += 1; continue
            res += s[i]; i += 1
        out = res
    return out
def norm_uk(p):
    p = p.replace('ɹ', 'r').replace('ɐ', 'ə').replace('ɡ', 'ɡ')
    p = fix_stress(p)
    return p
def norm_us(p):
    p = p.replace('ɹ', 'r')
    p = re.sub(r'ˈ([^ˈˌəʌaeiouæɑɒɔɛɜɪʊɚɝ]*)ə', r'ˈ\1ʌ', p)
    return p
def ipa_of(w):
    uk = IPA_UK.get(w); us = IPA_US.get(w)
    if not uk and ' ' in w:
        parts = [IPA_UK.get(x) for x in w.split()]
        if all(parts): uk = ' '.join(parts)
        parts = [IPA_US.get(x) for x in w.split()]
        if all(parts): us = ' '.join(parts)
    return (norm_uk(uk) if uk else None), (norm_us(us) if us else None)

ranked = [l for l, _ in lf.most_common() if accepted(l)][:32000]
RANK = {l: i + 1 for i, l in enumerate(ranked)}
print('lemmas', len(ranked), ranked[:40], ranked[9000:9010], ranked[29990:30000])

# ---------------- Arabic ----------------
AWN = collections.defaultdict(list)
for line in open('data/nd/corpora/omw-1.4/arb/wn-data-arb.tab', encoding='utf8'):
    if line.startswith('#'): continue
    p = line.rstrip('\n').split('\t')
    if len(p) == 3 and p[1] == 'arb:lemma':
        v = p[2].strip().replace('_', ' ')
        if v and v not in AWN[p[0]]: AWN[p[0]].append(v)
def off(s): return '%08d-%s' % (s.offset(), 's' if False else s.pos().replace('s', 'a'))

AE = collections.defaultdict(list)
ns = {'t': 'http://www.tei-c.org/ns/1.0'}
for ent in ET.parse('data/eng-ara.tei').getroot().iter('{http://www.tei-c.org/ns/1.0}entry'):
    o = ent.find('.//t:orth', ns)
    if o is None or not o.text: continue
    k = o.text.strip().lower()
    for q in ent.findall('.//t:quote', ns):
        if q.text:
            for v in re.split(r'[،,;]', q.text):
                v = v.strip()
                if v and v not in AE[k] and len(v) < 40: AE[k].append(v)
print('arabeyes', len(AE))

# ---------------- corpus: sentences, examples, collocations ----------------
BAD = re.compile(r"\b(negro|negroes|nigger|niggers|darky|darkies|jap|japs|chink|faggot|fag|whore|cunt|fuck|fucking|shit|bitch|bastard|slut|rape|raped|raping|kill yourself|homosexuals?|queer)\b", re.I)
def detok(toks):
    s = ' '.join(toks)
    s = s.replace('``', '"').replace("''", '"')
    s = re.sub(r' ([,.;:!?%)\]])', r'\1', s); s = re.sub(r'([(\[$]) ', r'\1', s)
    s = re.sub(r" (n't|'s|'re|'ve|'ll|'d|'m)\b", r'\1', s)
    s = re.sub(r'" (.*?) "', r'"\1"', s)
    s = s.replace(' -- ', ' – ').replace('--', '–')
    return s

sents = []  # (text, lemmas, src)
tagged_sents = []
UT = {'NOUN': 'n', 'VERB': 'v', 'ADJ': 'a', 'ADV': 'r', 'ADP': 'p', 'PRT': 'p', 'DET': 'd', 'PRON': 'o', 'NUM': 'x', 'CONJ': 'c', '.': '.', 'X': 'x'}
for fid in brown.fileids():
    cat = brown.categories(fid)[0]
    for s in brown.tagged_sents(fid, tagset='universal'):
        tagged_sents.append([(w, UT.get(t, 'x')) for w, t in s])
        toks = [w for w, _ in s]
        if 7 <= len(toks) <= 24 and toks[-1] in '.?!' and toks[0][:1].isupper():
            sents.append((detok(toks), 'Brown · ' + cat))
print('brown done')
import pickle
RC = 'data/reuters_tagged.pkl'
rcache = pickle.load(open(RC, 'rb')) if os.path.exists(RC) else {}
for fid in reuters.fileids():
    raw = reuters.raw(fid)
    body = raw.split('\n', 1)[1] if '\n' in raw else raw
    if sum(c.isupper() for c in body) > .3 * sum(c.isalpha() for c in body): continue
    for s in nltk.sent_tokenize(re.sub(r'\s+', ' ', body.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&'))):
        if s in rcache: toks, tg = rcache[s]
        else:
            toks = nltk.word_tokenize(s); tg = nltk.pos_tag(toks, tagset='universal'); rcache[s] = (toks, tg)
        tagged_sents.append([(w, UT.get(t, 'x')) for w, t in tg])
        if 7 <= len(toks) <= 24 and s[-1] in '.?!' and s[0].isupper() and '<' not in s and sum(c.isdigit() for c in s) < 6:
            sents.append((s, 'Reuters news'))
pickle.dump(rcache, open(RC, 'wb'))
print('reuters done', len(sents), len(tagged_sents))

sents = [(t, src) for t, src in sents if not BAD.search(t) and len(t) <= 140 and t.count('"') % 2 == 0 and not re.search(r"\b[A-Z]{3,}\b", t)]
TOK = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*")
lemma_sents = collections.defaultdict(list)
for i, (t, src) in enumerate(sents):
    for l in set(lem(x) for x in TOK.findall(t)):
        if l in RANK: lemma_sents[l].append(i)

# collocations
uni = collections.Counter(); pair = collections.Counter(); pdet = collections.defaultdict(collections.Counter)
STOPV = {'be', 'have', 'do', 'say', 'get', 'go', 'make', 'take'} - {'make', 'take', 'get'}
for s in tagged_sents:
    ls = [(lem(w) if w.isalpha() else w.lower(), t, w) for w, t in s]
    for l, t, _ in ls: uni[(l, t)] += 1
    n = len(ls)
    for i in range(n):
        l1, t1, w1 = ls[i]
        if not W.match(l1): continue
        if i + 1 < n:
            l2, t2, w2 = ls[i + 1]
            if W.match(l2):
                if t1 == 'a' and t2 == 'n': pair[(l1, 'a', l2, 'n', 'AN')] += 1
                elif t1 == 'n' and t2 == 'n': pair[(l1, 'n', l2, 'n', 'NN')] += 1
                elif t1 == 'r' and t2 == 'a' and l1 not in ('so', 'too', 'very', 'more', 'most', 'as', 'how', 'not', 'less', 'also', 'even', 'only', 'just', 'quite', 'rather'): pair[(l1, 'r', l2, 'a', 'RA')] += 1
                elif t1 == 'v' and t2 == 'p' and l1 not in STOPV and l1 != 'be': pair[(l1, 'v', l2, 'p', 'VP')] += 1
                elif t1 == 'a' and t2 == 'p': pair[(l1, 'a', l2, 'p', 'AP')] += 1
                elif t1 == 'n' and t2 == 'p' and l2 not in ('of',): pair[(l1, 'n', l2, 'p', 'NP')] += 1
                elif t1 == 'v' and t2 == 'r' and l1 not in ('be', 'have', 'do') and l2 not in ('not', 'also', 'then', 'now', 'so', 'only', 'even', 'just', 'still', 'already', 'never', 'always', 'n\'t', 'there', 'here', 'too', 'again', 'as', 'ever'): pair[(l1, 'v', l2, 'r', 'VR')] += 1
        if t1 == 'v' and l1 not in ('be', 'have', 'do'):
            det = ''
            j = i + 1
            while j < n and j <= i + 3 and ls[j][1] in ('d', 'a') and ls[j][0] not in ('which', 'what', 'that', 'this', 'these', 'those'):
                if ls[j][1] == 'd' and not det: det = ls[j][0]
                j += 1
            if j < n and j <= i + 3 and ls[j][1] == 'n' and W.match(ls[j][0]):
                pair[(l1, 'v', ls[j][0], 'n', 'VN')] += 1; pdet[(l1, ls[j][0])][det if det in ('a', 'the', 'an', '') else '~'] += 1
tot = sum(uni.values())
FUNCW = FUNC_EXTRA | {'mr', 'mrs', 'pct', 'mln', 'dlrs', 'cts', 'billion', 'year', 'company', 'inc', 'corp', 'co', 'ltd', 'shr', 'net', 'qtr'}
colls = collections.defaultdict(list)
for (l1, t1, l2, t2, pat), f in pair.items():
    if f < 3 or l1 in FUNCW and pat not in ('VP', 'AP', 'NP') or (l2 in FUNCW and pat not in ('VP', 'AP', 'NP')): continue
    if l1 not in RANK or (l2 not in RANK and pat not in ('VP', 'AP', 'NP')): continue
    dice = 14 + math.log2(2 * f / (uni[(l1, t1)] + uni[(l2, t2)]))
    if dice < 5: continue
    if pat == 'VN':
        d = pdet[(l1, l2)].most_common(1)[0][0]
        d = '' if d == '~' else d
        txt = f"{l1} {d + ' ' if d else ''}{l2}"
    else: txt = f"{l1} {l2}"
    if pat in ('VP', 'AP', 'NP'):
        colls[l1].append((dice + math.log2(f) / 3, txt))
    else:
        colls[l1].append((dice + math.log2(f) / 3, txt)); colls[l2].append((dice + math.log2(f) / 3, txt))
print('colls', len(colls))

# register: spoken (subtitles) vs written (Brown + Reuters)
sub = collections.Counter()
for line in open('data/en_50k.txt', encoding='utf8'):
    w, c = line.split(); sub[lem(w)] += int(c)
written = collections.Counter()
for (l, t), c in uni.items(): written[l] += c
st, wt = sum(sub.values()), sum(written.values())
# Brown "learned" + government genres vs fiction → academic keyness
acad = collections.Counter(); fict = collections.Counter()
for fid in brown.fileids():
    cat = brown.categories(fid)[0]
    tgt = acad if cat in ('learned', 'government') else fict if cat in ('fiction', 'romance', 'adventure', 'mystery', 'science_fiction', 'humor') else None
    if tgt is None: continue
    for w in brown.words(fid):
        if w.isalpha(): tgt[lem(w)] += 1
at, ft = sum(acad.values()), sum(fict.values())

AWL = {}
for line in open('awl.txt', encoding='utf8'):
    line = line.strip()
    if not line or line.startswith('#'): continue
    n, words = line.split(':', 1)
    for x in words.split(','):
        x = x.strip()
        if x: AWL[x] = int(n)

def register(l, syns):
    tags = []
    usage = set()
    for s in syns[:3]:
        for u in s.usage_domains(): usage.add(u.name().split('.')[0])
    if 'slang' in usage: tags.append('slang')
    elif 'colloquialism' in usage: tags.append('informal')
    if l in AWL: tags.append('academic')
    s_, w_ = sub[l], written[l]
    if s_ + w_ >= 30 and not tags:
        r = math.log2(((s_ + 1) / st) / ((w_ + 1) / wt))
        if r > 2.3: tags.append('informal / spoken')
        elif r < -2.3:
            a_, f_ = acad[l], fict[l]
            tags.append('formal')
    if not tags and acad[l] + fict[l] >= 20:
        r = math.log2(((acad[l] + 1) / at) / ((fict[l] + 1) / ft))
        if r > 2.5: tags.append('formal')
    return tags or ['neutral']

# ---------------- WordNet entries ----------------
PNAME = {'n': 'noun', 'v': 'verb', 'a': 'adjective', 's': 'adjective', 'r': 'adverb'}
FPOS = {}
for w in 'the a an this that these those every each either neither some any no all both half many much few little several enough such what which whose another other'.split(): FPOS[w] = 'determiner'
for w in 'i you he she it we they me him her us them myself yourself himself herself itself ourselves themselves who whom whoever someone somebody something anyone anybody anything everyone everybody everything nobody nothing one mine yours ours theirs hers'.split(): FPOS[w] = 'pronoun'
for w in 'of to in for with on at by from about into onto upon above below over under between among through during before after against without within across along around behind beyond toward towards off near via per despite except like unlike since until till beside besides throughout inside outside underneath'.split(): FPOS.setdefault(w, 'preposition')
for w in 'and or but nor yet so because although though unless whether while whereas if than once'.split(): FPOS.setdefault(w, 'conjunction')
for w in 'can could may might must shall should will would ought'.split(): FPOS[w] = 'modal verb'
for w in 'be have do'.split(): FPOS[w] = 'verb (also auxiliary)'

def sense_list(l):
    key = l.replace(' ', '_')
    syns = wn.synsets(key)
    scored = []
    for i, s in enumerate(syns):
        c = sum(x.count() for x in s.lemmas() if x.name().lower() == key)
        scored.append((-c, i, s))
    scored.sort(key=lambda x: (x[0], x[1]))
    order = [s for _, _, s in scored]
    pick, seen_pos = [], collections.Counter()
    for s in order:
        p = 'a' if s.pos() == 's' else s.pos()
        if seen_pos[p] >= 3: continue
        pick.append(s); seen_pos[p] += 1
        if len(pick) >= 5: break
    return syns, pick

def entry(l, is_phrase=False):
    syns, pick = sense_list(l)
    key = l.replace(' ', '_')
    e = {}
    uk, us = ipa_of(l)
    if uk: e['uk'] = uk
    if us and us != uk: e['us'] = us
    pos = []
    for s in pick:
        p = PNAME[s.pos()]
        if p not in pos: pos.append(p)
    if l in FPOS: pos = [FPOS[l]] + [p for p in pos if p != FPOS[l]]
    e['pos'] = pos
    senses = []
    for s in pick:
        sd = {'p': PNAME[s.pos()][:3] if s.pos() != 'r' else 'adv', 'd': s.definition()}
        ar = AWN.get('%08d-%s' % (s.offset(), 'a' if s.pos() == 's' else s.pos()), [])
        if ar and s.pos() == 'n' and not is_phrase: sd['ar'] = ar[:2]
        syn = [x.name().replace('_', ' ') for x in s.lemmas() if x.name().lower() != key][:3]
        if syn: sd['syn'] = syn
        exs = [x for x in s.examples() if re.search(r'\b' + re.escape(l.split()[0][:max(3, len(l.split()[0]) - 2)]), x, re.I)][:1]
        if exs: sd['ex'] = exs[0]
        senses.append(sd)
    if senses: e['s'] = senses
    wk = ENAR.get(l)
    if wk:
        wk = [[(x + ['', '', ''])[0], (x + ['', '', ''])[1], (x + ['', '', ''])[2]] for x in wk[0] if len(x) >= 2]
        wk = [wk]
        e['arw'] = [[p, a, g] for p, a, g in wk[0][:5] if p != 'proper noun']
    ar = [] if wk else list(AE.get(l, []))[:4]
    if not ar and not wk:
        for s in senses:
            for a in s.get('ar', []):
                if a not in ar: ar.append(a)
        ar = ar[:4]
    if ar: e['ar'] = ar
    if not senses and wk:
        e['s'] = [{'p': p[:3], 'd': g, 'ar': [a]} for p, a, g in wk[0][:4]]
        if not e.get('pos'): e['pos'] = list(dict.fromkeys(p for p, _, _ in wk[0][:4]))
    e['reg'] = register(l, syns) if not is_phrase else (['informal'] if any(u.name().startswith(('colloquialism', 'slang')) for s in syns[:2] for u in s.usage_domains()) else ['neutral'])
    if l in AWL: e['awl'] = AWL[l]
    return e, syns

# ---------------- MWEs (phrasal verbs, idioms, compounds) ----------------
PARTS = {'up', 'down', 'out', 'off', 'on', 'in', 'over', 'away', 'back', 'around', 'about', 'through', 'along', 'across', 'by', 'forward', 'apart', 'aside', 'together', 'into', 'onto', 'after', 'for', 'with', 'from', 'to', 'at', 'of', 'upon', 'against', 'behind', 'ahead'}
corpus_text = ' ' + ' '.join(t.lower() for t, _ in sents) + ' '
mwe = {}
for s in wn.all_synsets():
    for lm in s.lemmas():
        n = lm.name()
        if '_' not in n or not n.islower(): continue
        toks = n.split('_')
        if not (2 <= len(toks) <= 5) or not all(re.match(r'^[a-z]+$', t) for t in toks): continue
        if not all((t in RANK and RANK[t] <= 20000) or t in FUNC_EXTRA or t in PARTS for t in toks): continue
        phrase = ' '.join(toks)
        pos = s.pos()
        typ = None
        if pos == 'v':
            if len(toks) <= 3 and all(t in PARTS for t in toks[1:]): typ = 'phrasal verb'
            else: typ = 'idiom'
        elif pos in 'asr':
            typ = 'idiom' if len(toks) >= 2 else None
        elif pos == 'n':
            if f' {phrase} ' in corpus_text: typ = 'compound'
        if typ and phrase not in mwe: mwe[phrase] = typ
PH_IDIOM = {'phrase', 'proverb', 'prepositional phrase', 'adverb', 'adjective', 'interjection', 'conjunction', 'preposition'}
for k, v in ENAR.items():
    if ' ' not in k or not re.match(r'^[a-z][a-z ]+[a-z]$', k): continue
    toks = k.split()
    if len(toks) > 6: continue
    if not all((t in RANK and RANK[t] <= 25000) or t in FUNC_EXTRA or t in PARTS or t in ('one', "one's", 'someone', 'something') for t in toks): continue
    pos = v[0][0][0]
    if pos == 'verb': typ = 'phrasal verb' if len(toks) <= 3 and all(t in PARTS for t in toks[1:]) else 'idiom'
    elif pos == 'proverb': typ = 'proverb'
    elif pos in PH_IDIOM: typ = 'idiom'
    elif pos == 'noun' and f' {k} ' in corpus_text: typ = 'compound'
    else: continue
    if k not in mwe: mwe[k] = typ
print('mwe', len(mwe), collections.Counter(mwe.values()))

# own idiom list (content/idioms.json) is merged in the app build

# ---------------- write ----------------
def examples_for(l, exclude=()):
    ids = lemma_sents.get(l, [])
    cand = [sents[i] for i in ids]
    cand.sort(key=lambda x: (abs(len(x[0].split()) - 13), x[0]))
    out, srcs = [], set()
    for t, src in cand:
        if src.split(' ·')[0] in srcs and len(cand) > 6: continue
        out.append([t, src]); srcs.add(src.split(' ·')[0])
        if len(out) >= 3: break
    return out

def coll_for(l):
    xs = sorted(colls.get(l, []), reverse=True)
    out = []
    for _, t in xs:
        if t not in out: out.append(t)
        if len(out) >= 6: break
    return out

shards = collections.defaultdict(dict)
def shard_of(k):
    c = k[0]
    return c if c.isalpha() else '_'

index = []  # lemma, rank
missing_ar = 0
for l in ranked:
    e, syns = entry(l)
    e['r'] = RANK[l]
    ex = examples_for(l)
    if ex: e['x'] = ex
    co = coll_for(l)
    # phrases starting with this lemma are good collocations too
    if co: e['c'] = co
    if 'ar' not in e and 'arw' not in e: missing_ar += 1
    shards[shard_of(l)][l] = e
print('missing arabic', missing_ar, 'of', len(ranked))
miss_top = [l for l in ranked[:15000] if 'ar' not in shards[shard_of(l)][l] and 'arw' not in shards[shard_of(l)][l]]
print('missing top15k', len(miss_top), miss_top[:80])

sents_lo = [t.lower() for t, _ in sents]
for p, typ in mwe.items():
    e, syns = entry(p, True)
    e['t'] = typ
    e['pos'] = [typ]
    ex = []
    for (t, src), lo in zip(sents, sents_lo):
        if p in lo and re.search(r'\b' + re.escape(p) + r'\b', lo): ex.append([t, src])
        if len(ex) >= 2: break
    if ex: e['x'] = ex
    shards[shard_of(p)][p] = e

# phrasal verbs / idioms listed under their head word
for p, typ in mwe.items():
    head = p.split()[0]
    if head in RANK and typ != 'compound':
        shards[shard_of(head)][head].setdefault('ph', []).append(p)
for sh in shards.values():
    for k, e in sh.items():
        if 'ph' in e:
            e['ph'] = sorted(e['ph'], key=lambda p: (len(p), p))[:12]

tot_size = 0
for k, sh in shards.items():
    s = json.dumps(sh, ensure_ascii=False, separators=(',', ':'))
    open(f'{OUT}/d_{k}.json', 'w', encoding='utf8').write(s); tot_size += len(s.encode())
print('dict bytes', tot_size)

# core index: lemma list by rank, MWE list, form→lemma map
fl = {}
for l in ranked:
    for f in lemma_forms.get(l, ()):
        if f != l: fl[f] = l
for w, l in SP.items():
    if w != l and l in RANK and W.match(w) and w not in fl and w not in RANK: fl[w] = l
for w, l in FIX.items():
    if l in RANK: fl[w] = l
# forms that are also lemmas in their own right ("left", "found", "saw")
also = {f: l for f, l in fl.items() if f in RANK}
PROF = sorted(l for l in ranked if re.search(r'^(fuck|shit|cunt|bitch|bastard|dick|cock|pussy|whore|slut|fag|nigg|rape|sodom|porn|wank|twat|tit|boob|horny|sex|anal|orgasm|penis|vagina|cum|piss|arse|ass|damn|goddamn|crap|bollock|bugger|sodding|bloody|hell|jizz|dildo|suck|nude|naked|kill|murder|suicide|drug|cocaine|heroin)', l))
def pcount(p):
    toks = p.split(); n = corpus_text.count(' ' + p + ' ')
    for f in lemma_forms.get(toks[0], ()):
        if f != toks[0]: n += corpus_text.count(' ' + ' '.join([f] + toks[1:]) + ' ')
    return n
mwe = {k: [t, pcount(k)] for k, t in mwe.items()}
core = {'awl': {w: n for w, n in AWL.items() if w in RANK}, 'prof': PROF, 'lemmas': ranked, 'forms': fl, 'mwe': mwe, 'shards': sorted(shards.keys())}
s = json.dumps(core, ensure_ascii=False, separators=(',', ':'))
open(f'{OUT}/core.json', 'w', encoding='utf8').write(s)
print('core bytes', len(s.encode()), 'forms', len(fl))

# ---------------- pseudo-words for the placement test ----------------
allw = set(forms) | set(IPA_UK) | set(IPA_US) | set(SP)
tri = collections.defaultdict(collections.Counter)
for l in ranked[:15000]:
    if len(l) < 5 or '-' in l: continue
    w = '^^' + l + '$'
    for i in range(len(w) - 2): tri[w[i:i + 2]][w[i + 2]] += 1
fakes = set()
tries = 0
while len(fakes) < 400 and tries < 200000:
    tries += 1
    w = '^^'
    while len(w) < 16:
        c = tri[w[-2:]]
        if not c: break
        ch = random.choices(list(c), weights=c.values())[0]
        if ch == '$': break
        w += ch
    w = w[2:]
    if 6 <= len(w) <= 11 and w not in allw and not wn.synsets(w) and wordfreq.zipf_frequency(w, 'en') == 0 and not any(w.startswith(x) and len(x) >= len(w) - 2 for x in ranked[:20000] if len(x) > 4):
        fakes.add(w)
json.dump(sorted(fakes), open(f'{OUT}/fakes.json', 'w'))
print('fakes', len(fakes), sorted(fakes)[:30])
