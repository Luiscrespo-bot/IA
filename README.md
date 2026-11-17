# MiniGPT 6.1

Asistente de escritorio en Python que responde preguntas consultando documentos TXT propios, resuelve operaciones matemáticas y aprende de cualquier archivo de texto que le agregues.

---

## Tabla de contenidos

1. [¿Qué hace?](#qué-hace)
2. [Estructura del proyecto](#estructura-del-proyecto)
3. [Cómo funciona por dentro](#cómo-funciona-por-dentro)
   - [Motor de respuestas](#1-motor-de-respuestas-minigpt6py)
   - [Tokenizador BPE](#2-tokenizador-bpe-propio)
   - [Dataset comprimido](#3-dataset-comprimido)
   - [Embeddings TF-IDF](#4-embeddings-tf-idf)
4. [Archivos generados](#archivos-generados-carpeta-datos)
5. [Cómo agregar conocimiento](#cómo-agregar-conocimiento)
6. [Requisitos](#requisitos)
7. [Cómo ejecutar](#cómo-ejecutar)

---

## ¿Qué hace?

| Capacidad | Ejemplo |
|---|---|
| Responde preguntas de tus documentos | `"¿qué es un acorde?"` |
| Resuelve matemáticas | `"3 * 3 - 9"` → `0` |
| Lista documentos cargados | `"qué sabes"` |
| Muestra fragmentos de memoria | `"qué fragmentos son"` |
| Responde saludos e identidad | `"hola"`, `"quién eres"` |

---

## Estructura del proyecto

```
Modelo/
│
├── minigpt6.py              ← Programa principal (interfaz + motor)
├── construir_dataset.py     ← Pipeline: tokenizador → dataset → embeddings
├── README.md                ← Este archivo
│
├── conocimientos/           ← Aquí pones tus archivos .txt
│   ├── 01_matematicas.txt
│   ├── 02_fisica.txt
│   ├── 03_quimica.txt
│   ├── 04_biologia.txt
│   ├── 05_programacion.txt
│   ├── 06_musica_y_artes.txt
│   ├── 07_literatura_y_escritura_creativa.txt
│   ├── 08_metodo_cientifico_y_pensamiento_critico.txt
│   ├── mi_informacion.txt
│   └── musica.txt
│
└── datos/                   ← Generado automáticamente por construir_dataset.py
    ├── tokenizer.json        ← Vocabulario BPE + reglas de fusión
    ├── dataset.json.gz       ← Fragmentos tokenizados (comprimido)
    └── embeddings.json.gz    ← Vectores TF-IDF (comprimido)
```

---

## Cómo funciona por dentro

### 1. Motor de respuestas (`minigpt6.py`)

Cuando escribes una pregunta, el programa sigue este flujo en orden:

```
Tu pregunta
     │
     ▼
 ¿Es un saludo?        → "¡Hola! Soy MiniGPT 6.1..."
     │ no
     ▼
 ¿Pregunta de identidad? → "Soy MiniGPT 6.1, creado con Python..."
     │ no
     ▼
 ¿Qué sabes / qué documentos? → Lista los archivos cargados
     │ no
     ▼
 ¿Fragmentos?          → Muestra los primeros 10 fragmentos
     │ no
     ▼
 ¿Dime un tema?        → Lista los documentos disponibles
     │ no
     ▼
 ¿Contiene números y operadores? → Calculadora segura
     │ no
     ▼
 Búsqueda en documentos (TF-IDF + normalización de tildes)
     │
     ▼
 ¿Score ≥ 0.10?        → Muestra el fragmento más relevante
     │ no
     ▼
 "No encontré información suficiente..."
```

#### Búsqueda de documentos

La búsqueda funciona con **palabras clave normalizadas** (sin tildes) para que `"musica"` encuentre `"música"`:

```
Score = F1(recall, precisión)

recall    = palabras_coincidentes / palabras_en_la_pregunta
precisión = palabras_coincidentes / palabras_en_el_fragmento
F1        = 2 × recall × precisión / (recall + precisión)

Bonus +0.30 si el fragmento contiene TODAS las palabras de la pregunta
```

---

### 2. Tokenizador BPE propio

**BPE** (Byte Pair Encoding) es el mismo algoritmo que usan GPT-4 y LLaMA para convertir texto en números.

#### ¿Cómo aprende?

Empieza con caracteres individuales y busca el par más frecuente en el corpus para fusionarlo. Repite hasta alcanzar el vocabulario objetivo.

```
Iteración 0:  ['_', 'm', 'u', 's', 'i', 'c', 'a']
              par más frecuente: ('i', 'o') → fusionar
Iteración 1:  ['_', 'm', 'u', 's', 'i', 'c', 'a']
              par más frecuente: ('c', 'io') → fusionar
...
Iteración 559: '_musica' ya es un solo token
```

#### Resultado sobre este corpus

| Métrica | Valor |
|---|---|
| Vocabulario inicial | 41 caracteres |
| Vocabulario final | **600 tokens** |
| Fusiones aprendidas | 559 |
| Compresión | **0.362 tokens/carácter** |
| Ejemplo | `"la musica es armonia y ritmo"` → 13 IDs |

> Los tokens son **específicos del corpus** — aprende las sílabas y raíces más comunes de *tus* documentos, no de internet.

#### Tokens especiales

| Token | Significado |
|---|---|
| `<pad>` | Relleno (padding) |
| `<unk>` | Palabra desconocida |
| `<bos>` | Inicio de secuencia |
| `<eos>` | Fin de secuencia |
| `_` | Inicio de palabra |

---

### 3. Dataset comprimido

El script `construir_dataset.py` procesa todos los TXT y genera un dataset ultra-compacto:

#### Pipeline de procesamiento

```
TXT originales (194 KB)
        │
        ▼
  Limpieza del texto
  (elimina ===, ###, espacios múltiples)
        │
        ▼
  División en fragmentos
  (40–400 caracteres por fragmento)
        │
        ▼
  Normalización
  (minúsculas, sin tildes)
        │
        ▼
  Tokenización BPE
  (texto → lista de IDs enteros)
        │
        ▼
  JSON compacto  →  gzip nivel 9
        │
        ▼
  dataset.json.gz (152 KB) ← 40% del tamaño JSON puro
```

#### Formato interno de cada registro

```json
{
  "f": 2,
  "t": "La melodía es una sucesión de notas musicales.",
  "i": [2, 81, 410, 75, 80, 155, 528, 3]
}
```

- `f` → índice del archivo (1 número en vez del nombre largo)
- `t` → texto original (para mostrar en las respuestas)
- `i` → lista de IDs BPE (para búsqueda y embeddings)

#### Estadísticas del dataset

| Métrica | Valor |
|---|---|
| Fragmentos totales | 646 |
| Caracteres originales | 165,112 |
| Tokens BPE | 59,821 |
| Ahorro de representación | −64% |
| Tamaño final comprimido | **152 KB** |

---

### 4. Embeddings TF-IDF

Los embeddings convierten cada fragmento en un **vector numérico** que representa su significado. Dos fragmentos sobre el mismo tema tendrán vectores similares.

#### ¿Qué es TF-IDF?

```
TF  (Term Frequency)  = veces que aparece el token / total tokens del fragmento
IDF (Inverse Document Frequency) = log( (N+1) / (df+1) ) + 1

  N  = total de fragmentos
  df = en cuántos fragmentos aparece ese token

TF-IDF = TF × IDF
```

- **TF** premia tokens que aparecen mucho **en ese fragmento**.
- **IDF** penaliza tokens que aparecen en **todos** los fragmentos (palabras comunes).
- El resultado es que palabras raras y específicas tienen más peso.

#### Búsqueda por similitud coseno

```
similitud = (vector_query · vector_fragmento) / (|query| × |fragmento|)

Rango: 0.0 (sin relación) → 1.0 (idénticos)
```

Ambos vectores se normalizan a longitud unitaria (L2), así que el producto punto ya es la similitud coseno directamente.

#### Representación sparsa

Los vectores tienen **600 dimensiones** (una por token del vocabulario) pero la mayoría son 0. Solo se guardan los valores no nulos:

| Métrica | Valor |
|---|---|
| Dimensiones totales | 600 |
| Valores ≠ 0 por fragmento (promedio) | 65.9 |
| Densidad | **11%** del espacio |

Esto hace que los embeddings sean muy rápidos de calcular y muy livianos de almacenar.

#### Resultados de la verificación

```
'musica teoria notas acorde'
  [0.509] musica.txt           | MUSICA TEORICA La teoria musical...
  [0.438] 06_musica_y_artes.txt| MÚSICA Y ARTES: TEORÍA, TÉCNICA...

'biologia celula'
  [0.582] mi_informacion.txt   | 8. BIOLOGÍA 8.1 La célula...
  [0.551] 04_biologia.txt      | BIOLOGÍA: DE LA CÉLULA A LA BIOSFERA...
```

---

## Archivos generados (carpeta `datos/`)

| Archivo | Formato | Tamaño | Contenido |
|---|---|---|---|
| `tokenizer.json` | JSON legible | 31.7 KB | Vocabulario de 600 tokens + 559 reglas BPE |
| `dataset.json.gz` | JSON + gzip | 152.3 KB | 646 fragmentos con texto + IDs BPE |
| `embeddings.json.gz` | JSON + gzip | 209.3 KB | 646 vectores TF-IDF sparsos + tabla IDF |

> **Nota:** Los archivos `.json.gz` se abren con `gzip.open()` en Python. No son corruptos, solo están comprimidos.

---

## Cómo agregar conocimiento

1. Crea un archivo `.txt` con el contenido que quieras enseñarle
2. Colócalo en la carpeta `conocimientos/`
3. En la interfaz haz clic en **"Actualizar conocimientos"**
4. Opcionalmente, re-ejecuta `construir_dataset.py` para regenerar el dataset y los embeddings

```
conocimientos/
  └── mi_nuevo_tema.txt   ← cualquier texto en español
```

No hace falta formato especial. Párrafos separados por líneas en blanco funcionan mejor.

---

## Requisitos

```bash
pip install torch
```

Python 3.9 o superior. Las demás librerías (`tkinter`, `gzip`, `json`, `unicodedata`, `re`, `math`, `collections`) son parte de la librería estándar — **no necesitan instalación**.

---

## Cómo ejecutar

### Iniciar el asistente

```bash
python minigpt6.py
```

### Regenerar dataset y embeddings

Ejecutar cuando agregues o modifiques archivos en `conocimientos/`:

```bash
python construir_dataset.py
```

El script imprime un resumen completo al terminar, incluyendo estadísticas de compresión y una prueba de búsqueda automática.

---

## Resumen visual del flujo completo

```
 conocimientos/*.txt
        │
        ▼
 construir_dataset.py
        │
        ├─→ tokenizer.json    (vocabulario BPE)
        ├─→ dataset.json.gz   (fragmentos + IDs)
        └─→ embeddings.json.gz (vectores TF-IDF)
        
        │
        ▼
 minigpt6.py  (interfaz gráfica)
        │
        ├─ carga documentos al iniciar
        ├─ recibe preguntas del usuario
        ├─ normaliza tildes y stopwords
        ├─ calcula similitud F1 con fragmentos
        └─ muestra la respuesta más relevante
```
