#!/bin/bash
# Downloads the dictionary sources into tools/data/ (about 120 MB) and installs the Python packages.
set -e
cd "$(dirname "$0")"; mkdir -p data/nd/corpora data/nd/taggers data/nd/tokenizers data/langle-en data/langle-en-ar
pip install --quiet nltk wordfreq spacy-lookups-data
N=https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages
for f in corpora/wordnet corpora/omw-1.4 corpora/brown corpora/reuters taggers/averaged_perceptron_tagger_eng taggers/universal_tagset tokenizers/punkt_tab; do
  curl -sSL -o data/tmp.zip $N/$f.zip && unzip -qo data/tmp.zip -d data/nd/$(dirname $f)
done
rm -f data/tmp.zip; chmod -R go-w data/nd
curl -sSL -o data/en_50k.txt https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018/en/en_50k.txt
curl -sSL -o data/ipa_en_UK.txt https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/en_UK.txt
curl -sSL -o data/ipa_en_US.txt https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/en_US.txt
curl -sSL -o data/eng-ara.tei https://raw.githubusercontent.com/freedict/fd-dictionaries/master/eng-ara/eng-ara.tei
curl -sSL https://registry.npmjs.org/@langle/en/-/en-2026.1121.2.tgz | tar xz -C data/langle-en --strip-components=1
curl -sSL https://registry.npmjs.org/@langle/en-ar/-/en-ar-2026.1121.7.tgz | tar xz -C data/langle-en-ar --strip-components=1
echo "Sources ready. Now: python3 tools/build_dict.py && python3 tools/build.py"
