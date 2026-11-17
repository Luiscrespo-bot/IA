"""
construir_dataset.py
====================
Pipeline completo:
  1. Lee todos los .txt de la carpeta conocimientos
  2. Limpia y normaliza el texto
  3. Entrena un tokenizador BPE (Byte Pair Encoding) propio
     que comprime al maximo el vocabulario
  4. Convierte los fragmentos en un dataset comprimido (.json.gz)
  5. Calcula embeddings TF-IDF y los guarda comprimidos (.json.gz)

Salida en:  Modelo/datos/
  - tokenizer.json      : vocabulario y reglas BPE
  - dataset.json.gz     : fragmentos tokenizados (muy liviano)
  - embeddings.json.gz  : matriz TF-IDF sparsa + metadatos
"""

import os
import re
import gzip
import json
import math
import unicodedata
from collections import Counter, defaultdict


# ------------------------------------------------------------------
# RUTAS
# ------------------------------------------------------------------

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_DIR  = os.path.join(BASE_DIR, "conocimientos")
OUTPUT_DIR     = os.path.join(BASE_DIR, "datos")

os.makedirs(OUTPUT_DIR, exist_ok=True)

TOKENIZER_FILE  = os.path.join(OUTPUT_DIR, "tokenizer.json")
DATASET_FILE    = os.path.join(OUTPUT_DIR, "dataset.json.gz")
EMBEDDINGS_FILE = os.path.join(OUTPUT_DIR, "embeddings.json.gz")


# ------------------------------------------------------------------
# PASO 1 - LEER Y LIMPIAR TEXTOS
# ------------------------------------------------------------------

def normalize(text):
    """Elimina tildes y pasa a minusculas."""
    nfkd = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def clean_text(text):
    """Elimina separadores decorativos y colapsa espacios."""
    text = re.sub(r"[=\-#*]{3,}", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_into_fragments(text, min_chars=40, max_chars=400):
    """
    Divide el texto en fragmentos por parrafo.
    Fragmentos cortos se fusionan. Fragmentos largos se parten.
    """
    raw = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    fragments = []
    buffer    = ""

    for paragraph in raw:

        paragraph = re.sub(r"\s+", " ", paragraph).strip()

        if len(paragraph) < min_chars:
            buffer += " " + paragraph
            continue

        if buffer:
            paragraph = (buffer + " " + paragraph).strip()
            buffer = ""

        if len(paragraph) <= max_chars:
            fragments.append(paragraph)
        else:
            sentences = re.split(r"(?<=[.!?])\s+", paragraph)
            current   = ""
            for sent in sentences:
                if len(current) + len(sent) + 1 <= max_chars:
                    current = (current + " " + sent).strip()
                else:
                    if current:
                        fragments.append(current)
                    current = sent
            if current:
                fragments.append(current)

    if buffer.strip():
        if fragments:
            fragments[-1] += " " + buffer.strip()
        else:
            fragments.append(buffer.strip())

    return fragments


print("=" * 55)
print("PASO 1 - Leyendo documentos")
print("=" * 55)

raw_fragments = []

for filename in sorted(os.listdir(KNOWLEDGE_DIR)):

    if not filename.lower().endswith(".txt"):
        continue

    path = os.path.join(KNOWLEDGE_DIR, filename)

    with open(path, "r", encoding="utf-8-sig") as f:
        content = f.read()

    content = clean_text(content)
    frags   = split_into_fragments(content)

    for frag in frags:
        raw_fragments.append({"file": filename, "text": frag})

    print(f"  {filename}: {len(frags)} fragmentos")

print(f"\nTotal fragmentos: {len(raw_fragments)}")

all_texts_normalized = [normalize(d["text"]) for d in raw_fragments]


# ------------------------------------------------------------------
# PASO 2 - TOKENIZADOR BPE ACELERADO
# ------------------------------------------------------------------

print("\n" + "=" * 55)
print("PASO 2 - Entrenando tokenizador BPE")
print("=" * 55)


class BPETokenizer:
    """
    Byte Pair Encoding con indice inverso para actualizacion O(1) por merge.
    - Empieza con caracteres individuales (+ marcador de inicio de palabra ▁)
    - Fusiona los pares mas frecuentes iterativamente
    - Genera un vocabulario compacto ideal para el corpus dado
    """

    PAD   = "<pad>"
    UNK   = "<unk>"
    BOS   = "<bos>"
    EOS   = "<eos>"
    SPACE = "_"      # marcador inicio de palabra (ASCII para JSON limpio)

    def __init__(self):
        self.vocab       = {}
        self.id_to_token = {}
        self.merges      = []

    # ---- Helpers de corpus ------------------------------------

    def _word_to_chars(self, word):
        return (self.SPACE,) + tuple(word)

    def _build_corpus(self, texts):
        """
        Retorna:
          word_freq  : Counter {word_str -> count}
          corpus     : dict {word_tuple -> count}
        """
        word_freq = Counter()
        for text in texts:
            word_freq.update(re.findall(r"[a-z0-9]+", text))

        corpus = {
            self._word_to_chars(w): c
            for w, c in word_freq.items()
        }
        return word_freq, corpus

    def _build_pair_index(self, corpus):
        """
        Construye:
          pair_freq  : Counter {(a,b) -> total_frecuencia}
          pair_words : dict {(a,b) -> set de word_tuples que lo contienen}
        """
        pair_freq  = Counter()
        pair_words = defaultdict(set)

        for word, count in corpus.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_freq[pair]  += count
                pair_words[pair].add(word)

        return pair_freq, pair_words

    def _apply_merge(self, corpus, pair_freq, pair_words, pair):
        """
        Aplica un merge al corpus y actualiza pair_freq / pair_words
        de forma incremental (solo las palabras afectadas).
        """
        a, b    = pair
        merged  = a + b
        affected = list(pair_words.get(pair, set()))

        for old_word in affected:

            if old_word not in corpus:
                continue

            count = corpus[old_word]

            # Construir nueva palabra con el merge aplicado
            new_word_list = []
            i = 0
            while i < len(old_word):
                if (
                    i < len(old_word) - 1
                    and old_word[i] == a
                    and old_word[i + 1] == b
                ):
                    new_word_list.append(merged)
                    i += 2
                else:
                    new_word_list.append(old_word[i])
                    i += 1
            new_word = tuple(new_word_list)

            # Actualizar corpus
            del corpus[old_word]
            corpus[new_word] = corpus.get(new_word, 0) + count

            # Restar pares de old_word
            for j in range(len(old_word) - 1):
                p = (old_word[j], old_word[j + 1])
                pair_freq[p] -= count
                if pair_freq[p] <= 0:
                    del pair_freq[p]
                    pair_words.pop(p, None)
                else:
                    pair_words[p].discard(old_word)

            # Sumar pares de new_word
            for j in range(len(new_word) - 1):
                p = (new_word[j], new_word[j + 1])
                pair_freq[p]  += count
                pair_words[p].add(new_word)

        # Limpiar el par fusionado
        pair_freq.pop(pair, None)
        pair_words.pop(pair, None)

    # ---- Entrenamiento ----------------------------------------

    def train(self, texts, vocab_size=600, min_freq=2):

        specials  = [self.PAD, self.UNK, self.BOS, self.EOS, self.SPACE]
        all_chars = sorted(set(
            ch for text in texts
            for ch in text
            if re.match(r"[a-z0-9]", ch)
        ))

        initial_vocab    = specials + all_chars
        self.vocab       = {t: i for i, t in enumerate(initial_vocab)}
        self.id_to_token = {i: t for t, i in self.vocab.items()}

        _, corpus = self._build_corpus(texts)

        pair_freq, pair_words = self._build_pair_index(corpus)

        num_merges  = vocab_size - len(self.vocab)
        merges_done = 0

        print(f"  Vocab inicial:     {len(self.vocab)} simbolos")
        print(f"  Pares unicos:      {len(pair_freq)}")
        print(f"  Fusiones objetivo: {num_merges}")

        for step in range(num_merges):

            if not pair_freq:
                break

            best_pair = max(pair_freq, key=lambda p: pair_freq[p])
            best_freq = pair_freq[best_pair]

            if best_freq < min_freq:
                break

            self._apply_merge(corpus, pair_freq, pair_words, best_pair)
            self.merges.append(best_pair)

            merged_token = best_pair[0] + best_pair[1]
            if merged_token not in self.vocab:
                new_id = len(self.vocab)
                self.vocab[merged_token] = new_id
                self.id_to_token[new_id] = merged_token

            merges_done += 1

            if merges_done % 100 == 0:
                print(f"  ... {merges_done} fusiones (pares restantes: {len(pair_freq)})")

        print(f"  Fusiones realizadas: {merges_done}")
        print(f"  Vocabulario final:   {len(self.vocab)} tokens")

    # ---- Tokenizacion -----------------------------------------

    def _tokenize_word(self, word):
        symbols = list(self._word_to_chars(word))
        for a, b in self.merges:
            merged = a + b
            i = 0
            new_symbols = []
            while i < len(symbols):
                if (
                    i < len(symbols) - 1
                    and symbols[i] == a
                    and symbols[i + 1] == b
                ):
                    new_symbols.append(merged)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
        return symbols

    def encode(self, text):
        text  = normalize(text)
        ws    = re.findall(r"[a-z0-9]+", text)
        ids   = [self.vocab.get(self.BOS, 2)]
        unk   = self.vocab.get(self.UNK, 1)
        for word in ws:
            for t in self._tokenize_word(word):
                ids.append(self.vocab.get(t, unk))
        ids.append(self.vocab.get(self.EOS, 3))
        return ids

    def decode(self, ids):
        skip = {
            self.vocab.get(self.PAD, 0),
            self.vocab.get(self.BOS, 2),
            self.vocab.get(self.EOS, 3),
        }
        tokens = [
            self.id_to_token.get(i, self.UNK)
            for i in ids if i not in skip
        ]
        return "".join(tokens).replace(self.SPACE, " ").strip()

    # ---- Serializar -------------------------------------------

    def save(self, path):
        data = {
            "vocab":  self.vocab,
            "merges": [list(m) for m in self.merges],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        size = os.path.getsize(path)
        print(f"  Guardado: {path}")
        print(f"  Tamano: {size:,} bytes ({size/1024:.1f} KB)")

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok              = cls()
        tok.vocab        = data["vocab"]
        tok.id_to_token  = {int(i): t for t, i in tok.vocab.items()}
        tok.merges       = [tuple(m) for m in data["merges"]]
        return tok


tokenizer = BPETokenizer()
tokenizer.train(all_texts_normalized, vocab_size=600, min_freq=2)
tokenizer.save(TOKENIZER_FILE)


# ------------------------------------------------------------------
# PRUEBA DE ENCODE / DECODE
# ------------------------------------------------------------------

sample = "la musica es armonia y ritmo"
enc    = tokenizer.encode(sample)
dec    = tokenizer.decode(enc)
print(f"\n  Prueba encode/decode:")
print(f"  Original : {sample!r}")
print(f"  IDs ({len(enc)}): {enc}")
print(f"  Decodif. : {dec!r}")
ratio = len(enc) / max(len(sample.split()), 1)
print(f"  Tokens por palabra: {ratio:.2f}")


# ------------------------------------------------------------------
# PASO 3 - DATASET COMPRIMIDO
# ------------------------------------------------------------------

print("\n" + "=" * 55)
print("PASO 3 - Generando dataset comprimido")
print("=" * 55)

files_list = sorted(set(d["file"] for d in raw_fragments))
file_to_id = {f: i for i, f in enumerate(files_list)}

dataset_records = []
original_chars  = 0
encoded_tokens  = 0

for frag in raw_fragments:
    ids             = tokenizer.encode(frag["text"])
    original_chars += len(frag["text"])
    encoded_tokens += len(ids)

    dataset_records.append({
        "f": file_to_id[frag["file"]],
        "t": frag["text"],
        "i": ids
    })

dataset = {
    "files":   files_list,
    "records": dataset_records
}

raw_json = json.dumps(dataset, ensure_ascii=False, separators=(",", ":"))

with gzip.open(DATASET_FILE, "wt", encoding="utf-8", compresslevel=9) as f:
    f.write(raw_json)

size_gz  = os.path.getsize(DATASET_FILE)
size_raw = len(raw_json.encode("utf-8"))

print(f"  Fragmentos:            {len(dataset_records)}")
print(f"  Caracteres originales: {original_chars:,}")
print(f"  Tokens BPE generados:  {encoded_tokens:,}")
print(f"  Tokens/caracter:       {encoded_tokens/max(original_chars,1):.3f}")
print(f"  Tamano JSON puro:      {size_raw:,} bytes  ({size_raw/1024:.1f} KB)")
print(f"  Tamano gzip final:     {size_gz:,} bytes  ({size_gz/1024:.1f} KB)")
print(f"  Compresion gzip:       {size_gz/max(size_raw,1):.1%}")


# ------------------------------------------------------------------
# PASO 4 - EMBEDDINGS TF-IDF
# ------------------------------------------------------------------

print("\n" + "=" * 55)
print("PASO 4 - Calculando embeddings TF-IDF")
print("=" * 55)

N  = len(dataset_records)
df = Counter()

for rec in dataset_records:
    df.update(set(rec["i"]))

idf = {
    tid: round(math.log((N + 1) / (count + 1)) + 1.0, 6)
    for tid, count in df.items()
}

embeddings = []

for rec in dataset_records:

    ids   = rec["i"]
    total = len(ids)

    if total == 0:
        embeddings.append({})
        continue

    tf       = Counter(ids)
    tfidf    = {}

    for tid, count in tf.items():
        val       = (count / total) * idf.get(tid, 1.0)
        tfidf[tid] = val

    # Normalizar L2
    norm = math.sqrt(sum(v * v for v in tfidf.values()))

    if norm > 0:
        tfidf = {k: round(v / norm, 6) for k, v in tfidf.items()}

    embeddings.append(tfidf)

nonzero_avg = sum(len(e) for e in embeddings) / max(N, 1)
density     = nonzero_avg / max(len(tokenizer.vocab), 1)

print(f"  Fragmentos embebidos:  {N}")
print(f"  Dimension del espacio: {len(tokenizer.vocab)} tokens")
print(f"  Valores != 0 (prom.):  {nonzero_avg:.1f}")
print(f"  Densidad del vector:   {density:.1%}")

# Guardar embeddings comprimidos
emb_data = {
    "vocab_size": len(tokenizer.vocab),
    "idf":        {str(k): v for k, v in idf.items()},
    "vectors":    [
        {str(k): v for k, v in vec.items()}
        for vec in embeddings
    ]
}

emb_json = json.dumps(emb_data, ensure_ascii=False, separators=(",", ":"))

with gzip.open(EMBEDDINGS_FILE, "wt", encoding="utf-8", compresslevel=9) as f:
    f.write(emb_json)

size_emb = os.path.getsize(EMBEDDINGS_FILE)
print(f"  Tamano embeddings gz:  {size_emb:,} bytes  ({size_emb/1024:.1f} KB)")


# ------------------------------------------------------------------
# VERIFICACION - Busqueda semantica con embeddings
# ------------------------------------------------------------------

print("\n" + "=" * 55)
print("VERIFICACION - Busqueda semantica")
print("=" * 55)


def cosine_sim(va, vb):
    return sum(va.get(k, 0.0) * v for k, v in vb.items())


def query_vec(text):
    ids   = tokenizer.encode(text)
    total = len(ids)
    if total == 0:
        return {}
    tf   = Counter(ids)
    vec  = {}
    norm = 0.0
    for tid, count in tf.items():
        val      = (count / total) * idf.get(tid, 1.0)
        vec[tid] = val
        norm    += val * val
    norm = math.sqrt(norm)
    if norm > 0:
        vec = {k: v / norm for k, v in vec.items()}
    return vec


def search(query, top_k=3):
    qv     = query_vec(query)
    scores = [
        (cosine_sim(qv, emb), i)
        for i, emb in enumerate(embeddings)
        if emb
    ]
    scores.sort(reverse=True)
    return scores[:top_k]


tests = [
    "musica teoria notas acorde",
    "suma operacion matematica",
    "literatura narrativa novela",
    "programacion python lenguaje",
    "biologia celula",
]

for q in tests:
    results = search(q)
    print(f"\n  '{q}'")
    for score, idx in results:
        rec  = dataset_records[idx]
        file = files_list[rec["f"]]
        text = rec["t"][:70].replace("\n", " ")
        print(f"    [{score:.3f}] {file} | {text}...")


# ------------------------------------------------------------------
# RESUMEN FINAL
# ------------------------------------------------------------------

print("\n" + "=" * 55)
print("RESUMEN FINAL")
print("=" * 55)

total_original = sum(
    os.path.getsize(os.path.join(KNOWLEDGE_DIR, fn))
    for fn in os.listdir(KNOWLEDGE_DIR)
    if fn.endswith(".txt")
)

total_salida = (
    os.path.getsize(TOKENIZER_FILE)
    + os.path.getsize(DATASET_FILE)
    + os.path.getsize(EMBEDDINGS_FILE)
)

print(f"  TXT originales:  {total_original:,} bytes  ({total_original/1024:.1f} KB)")
print(f"  Datos generados: {total_salida:,} bytes  ({total_salida/1024:.1f} KB)")
print(f"  Ratio total:     {total_salida/max(total_original,1):.1%}")
print()
print(f"  datos/tokenizer.json   : vocabulario + reglas BPE")
print(f"  datos/dataset.json.gz  : fragmentos tokenizados")
print(f"  datos/embeddings.json.gz: vectores TF-IDF comprimidos")
print()
print("Listo.")
