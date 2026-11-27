"""Build step for the English app (run from the repo root: python3 tools/build.py).
1. content/texts.md  → content/texts.json
2. content/learn.py  → content/learn.json
3. sw.js from tools/sw.template.js with the file list and versions
"""
import json, os, re, hashlib, subprocess, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

# ---------- texts ----------
texts = []
cur = None
for line in open('content/texts.md', encoding='utf8'):
    line = line.rstrip('\n')
    if line.startswith('=== '):
        if cur: texts.append(cur)
        tid, style, level, title = [x.strip() for x in line[4:].split('|', 3)]
        cur = {'id': tid, 'style': style, 'level': level, 'title': title, 'q': [], 'body': [], '_in': 'head'}
    elif cur is None: continue
    elif cur['_in'] == 'head' and line.startswith('Q:'):
        q, a = line[2:].split('||', 1); cur['q'].append([q.strip(), a.strip()])
    elif cur['_in'] == 'head' and line.strip() == '---': cur['_in'] = 'body'
    elif cur['_in'] == 'body':
        cur['body'].append(line)
if cur: texts.append(cur)
for t in texts:
    paras, buf = [], []
    for l in t.pop('body') + ['']:
        if l.strip(): buf.append(l.strip())
        elif buf: paras.append(' '.join(buf)); buf = []
    if t['style'] in ('dialogue', 'ielts-listening', 'podcast') and any(re.match(r'^[A-Z][A-Z ]+:', p) for p in paras):
        lines = []
        for p in paras:
            for l in re.split(r'\n', p):
                pass
        # dialogue: each "NAME: text" line is its own paragraph
        out = []
        for p in paras:
            parts = re.split(r'\s(?=[A-Z][A-Z]+: )', p)
            out.extend(parts)
        paras = out
    t['paras'] = paras
    t.pop('_in')
    t['words'] = sum(len(re.findall(r"[A-Za-z']+", p)) for p in paras)
json.dump({'texts': texts}, open('content/texts.json', 'w', encoding='utf8'), ensure_ascii=False, indent=0)
print('texts', len(texts))

# ---------- learn.json ----------
subprocess.run([sys.executable, 'learn.py'], cwd='content', check=True)

# ---------- service worker ----------
def h(files):
    m = hashlib.sha256()
    for f in files: m.update(open(f, 'rb').read())
    return m.hexdigest()[:10]
shell = ['./', 'index.html', 'manifest.webmanifest', 'icons/icon-192.png', 'icons/icon-512.png', 'icons/maskable-512.png',
         'content/texts.json', 'content/learn.json', 'content/colloc.json', 'dict/core.json', 'dict/fakes.json']
dictf = sorted('dict/' + f for f in os.listdir('dict') if f.startswith('d_'))
sv = h([f for f in shell if f != './'])
dv = h(dictf)
sw = open('tools/sw.template.js', encoding='utf8').read()
sw = sw.replace('__VERSION__', sv).replace('__DICTV__', dv).replace('__FILES__', json.dumps(shell)).replace('__DICT__', json.dumps(dictf))
open('sw.js', 'w', encoding='utf8').write(sw)
print('sw', sv, dv, len(dictf), 'dict files')
