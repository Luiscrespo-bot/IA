
import os
import re
import ast
import json
import random
import threading
import unicodedata
import tkinter as tk
from tkinter import scrolledtext, messagebox

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================================
# MINIGPT 6.1
# Asistente con memoria documental y calculadora
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "conocimientos")

MODEL_FILE = os.path.join(BASE_DIR, "minigpt6_model.pt")
HISTORY_FILE = os.path.join(BASE_DIR, "historial6.json")

os.makedirs(KNOWLEDGE_DIR, exist_ok=True)

DEVICE = "cpu"

torch.manual_seed(42)
random.seed(42)

print("=" * 45)
print("MINIGPT 6.1")
print("=" * 45)
print("Carpeta del programa:", BASE_DIR)
print("Carpeta de conocimientos:", KNOWLEDGE_DIR)
print("¿Existe la carpeta?:", os.path.isdir(KNOWLEDGE_DIR))
print("\n========== DIAGNOSTICO ==========")

print("Archivo Python ejecutado:")
print(__file__)

print("\nCarpeta donde busca:")
print(KNOWLEDGE_DIR)

print("\nArchivos encontrados:")
print(os.listdir(KNOWLEDGE_DIR))

print("\nArchivos TXT detectados:")

for archivo in os.listdir(KNOWLEDGE_DIR):
    if archivo.lower().endswith(".txt"):
        print("ENCONTRADO:", repr(archivo))

print("=================================\n")


# ==========================================================
# 1. PEQUENO TRANSFORMER
# ==========================================================

SEED_TEXT = """
Hola. Soy MiniGPT 6.1.
Soy un asistente experimental creado con Python y PyTorch.
Puedo consultar documentos de texto.
Python es un lenguaje de programacion.
La inteligencia artificial permite crear sistemas inteligentes.
Un Transformer utiliza mecanismos de atencion.
La memoria permite guardar informacion.
Puedo resolver operaciones matematicas.
"""

class MiniTransformer(nn.Module):

    def __init__(self, vocab_size, max_len=128):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, 32)
        self.position_embedding = nn.Embedding(max_len, 32)

        layer = nn.TransformerEncoderLayer(
            d_model=32,
            nhead=4,
            dim_feedforward=64,
            dropout=0.0,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=1
        )

        self.output = nn.Linear(32, vocab_size)

    def forward(self, x):

        batch, length = x.shape

        positions = torch.arange(
            length,
            device=x.device
        ).unsqueeze(0)

        x = (
            self.token_embedding(x)
            + self.position_embedding(positions)
        )

        mask = torch.triu(
            torch.ones(
                length,
                length,
                device=x.device,
                dtype=torch.bool
            ),
            diagonal=1
        )

        x = self.transformer(x, mask=mask)

        return self.output(x)


# ==========================================================
# 2. ENTRENAMIENTO Y GUARDADO
# ==========================================================

model = None
chars = []
char_to_id = {}
id_to_char = {}


def train_model():

    global model, chars, char_to_id, id_to_char

    print("\nEntrenando Transformer...")

    chars = sorted(set(SEED_TEXT))

    char_to_id = {
        character: index
        for index, character in enumerate(chars)
    }

    id_to_char = {
        index: character
        for index, character in enumerate(chars)
    }

    data = torch.tensor(
        [char_to_id[c] for c in SEED_TEXT],
        dtype=torch.long
    )

    model = MiniTransformer(len(chars)).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.003
    )

    model.train()

    for step in range(400):

        batch_x = []
        batch_y = []

        for _ in range(16):

            start = random.randint(0, len(data) - 17)

            batch_x.append(data[start:start + 16])
            batch_y.append(data[start + 1:start + 17])

        x = torch.stack(batch_x).to(DEVICE)
        y = torch.stack(batch_y).to(DEVICE)

        logits = model(x)

        loss = F.cross_entropy(
            logits.reshape(-1, len(chars)),
            y.reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(
                f"Entrenamiento {step}/400 "
                f"- Loss: {loss.item():.4f}"
            )

    model.eval()

    torch.save(
        {
            "state_dict": model.state_dict(),
            "chars": chars
        },
        MODEL_FILE
    )

    print("Modelo guardado correctamente.")


def create_or_load_model():

    global model, chars, char_to_id, id_to_char

    if os.path.exists(MODEL_FILE):

        try:

            checkpoint = torch.load(
                MODEL_FILE,
                map_location=DEVICE,
                weights_only=True
            )

            chars = checkpoint["chars"]

            char_to_id = {
                c: i for i, c in enumerate(chars)
            }

            id_to_char = {
                i: c for i, c in enumerate(chars)
            }

            model = MiniTransformer(len(chars)).to(DEVICE)

            model.load_state_dict(
                checkpoint["state_dict"]
            )

            model.eval()

            print("Modelo anterior cargado.")
            return

        except Exception as error:

            print("Error cargando modelo:", error)
            print("Se creara un modelo nuevo.")

    train_model()


def generate_text(prompt, max_new_tokens=80):

    if not prompt:
        prompt = "Hola"

    encoded = [
        char_to_id.get(c, char_to_id.get(" ", 0))
        for c in prompt
    ]

    x = torch.tensor(
        [encoded[-128:]],
        dtype=torch.long
    )

    model.eval()

    with torch.no_grad():

        for _ in range(max_new_tokens):

            x_input = x[:, -128:]

            logits = model(x_input)

            probabilities = F.softmax(
                logits[:, -1, :],
                dim=-1
            )

            next_id = torch.multinomial(
                probabilities,
                1
            )

            x = torch.cat(
                [x, next_id],
                dim=1
            )

    return "".join(
        id_to_char.get(int(i), "")
        for i in x[0].tolist()
    )


# ==========================================================
# 3. MEMORIA DOCUMENTAL
# ==========================================================

documents = []


def load_documents():

    global documents

    documents = []

    if not os.path.isdir(KNOWLEDGE_DIR):

        print("ERROR: No existe la carpeta conocimientos.")
        return

    files_found = 0

    for filename in os.listdir(KNOWLEDGE_DIR):

        if not filename.lower().endswith(".txt"):
            continue

        path = os.path.join(
            KNOWLEDGE_DIR,
            filename
        )

        try:

            with open(
                path,
                "r",
                encoding="utf-8-sig"
            ) as file:

                content = file.read().strip()

            if not content:
                print("Archivo vacio:", filename)
                continue

            files_found += 1

            paragraphs = [
                paragraph.strip()
                for paragraph in re.split(
                    r"\n\s*\n",
                    content
                )
                if paragraph.strip()
            ]

            for paragraph in paragraphs:

                documents.append({
                    "file": filename,
                    "text": paragraph
                })

            print("Documento cargado:", filename)

        except Exception as error:

            print(
                "Error leyendo",
                filename,
                error
            )

    print("\nArchivos encontrados:", files_found)
    print("Fragmentos disponibles:", len(documents))


# ==========================================================
# 4. BUSCADOR DE INFORMACION
# ==========================================================

STOPWORDS = {
    "que", "como", "cual", "cuales", "cuando",
    "donde", "por", "para", "con", "una", "uno",
    "unos", "unas", "los", "las", "del", "de",
    "la", "el", "en", "es", "soy", "eres",
    "son", "mi", "tu", "me", "te", "se",
    "un", "y", "o", "a", "al", "lo", "sobre",
    "explica", "dime", "cuentame", "puedes",
    "quiero", "saber", "hay", "has", "han",
    "mas", "muy", "si", "no", "ya", "tan",
    "sus", "del", "les", "nos"
}


def normalize(text):
    """Elimina tildes y pasa a minusculas para comparar sin acento."""
    nfkd = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def words(text):

    tokens = re.findall(
        r"[a-zaeiounu0-9]+",
        normalize(text)
    )

    return [
        word
        for word in tokens
        if word not in STOPWORDS and len(word) > 1
    ]


def search_knowledge(question):

    query_words = set(words(question))

    if not query_words:
        return []

    results = []

    for doc in documents:

        doc_words = set(words(doc["text"]))

        if not doc_words:
            continue

        common = query_words.intersection(doc_words)

        if not common:
            continue

        # Recall: cuantas palabras de la pregunta aparecen en el doc
        recall = len(common) / len(query_words)

        # Precision parcial: evitar fragmentos con solo 1 palabra comun
        # en documentos enormes
        precision = len(common) / len(doc_words)

        # F1 ponderado (recall pesa mas: queremos que el doc responda
        # todas las palabras clave de la pregunta)
        score = (2 * recall * precision) / (recall + precision) \
            if (recall + precision) > 0 else 0

        # Bonus si el fragmento contiene todas las palabras de la query
        if query_words.issubset(doc_words):
            score = min(1.0, score + 0.3)

        results.append((score, doc))

    results.sort(
        key=lambda item: item[0],
        reverse=True
    )

    return results[:5]


# ==========================================================
# 5. CALCULADORA SEGURA
# ==========================================================

def safe_calculate(expression):

    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Mod,
        ast.Load
    )

    def evaluate(node):

        if not isinstance(node, allowed_nodes):
            raise ValueError("Operacion no permitida")

        if isinstance(node, ast.Expression):
            return evaluate(node.body)

        if isinstance(node, ast.Constant):

            if isinstance(node.value, bool):
                raise ValueError("Valor no permitido")

            if isinstance(node.value, (int, float)):
                return node.value

            raise ValueError("Valor no permitido")

        if isinstance(node, ast.UnaryOp):

            value = evaluate(node.operand)

            if isinstance(node.op, ast.USub):
                return -value

            if isinstance(node.op, ast.UAdd):
                return value

        if isinstance(node, ast.BinOp):

            a = evaluate(node.left)
            b = evaluate(node.right)

            if abs(a) > 10**12 or abs(b) > 10**12:
                raise ValueError("Numero demasiado grande")

            if isinstance(node.op, ast.Add):
                result = a + b

            elif isinstance(node.op, ast.Sub):
                result = a - b

            elif isinstance(node.op, ast.Mult):
                result = a * b

            elif isinstance(node.op, ast.Div):
                result = a / b

            elif isinstance(node.op, ast.Mod):
                result = a % b

            elif isinstance(node.op, ast.Pow):

                if abs(b) > 10:
                    raise ValueError("Exponente demasiado grande")

                result = a ** b

            else:
                raise ValueError("Operacion no permitida")

            if abs(result) > 10**12:
                raise ValueError("Resultado demasiado grande")

            return result

        raise ValueError("Expresion no valida")

    tree = ast.parse(expression, mode="eval")

    return evaluate(tree)


def extract_math(question):

    text = question.lower().replace(",", ".")

    text = re.sub(
        r"(?<=\d)\s*x\s*(?=\d)",
        "*",
        text
    )

    matches = re.findall(
        r"(?<![a-záéíóúñ])\d+(?:\.\d+)?"
        r"(?:\s*[\+\-*/%]\s*\d+(?:\.\d+)?)*",
        text
    )

    if not matches:
        return None

    expression = max(matches, key=len).strip()

    if not re.search(r"[\+\-*/%]", expression):
        return None

    return expression


# ==========================================================
# 6. HISTORIAL
# ==========================================================

history = []


def load_history():

    global history

    if os.path.exists(HISTORY_FILE):

        try:

            with open(
                HISTORY_FILE,
                "r",
                encoding="utf-8"
            ) as file:

                history = json.load(file)

        except Exception:

            history = []


def save_history():

    try:

        with open(
            HISTORY_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                history[-100:],
                file,
                ensure_ascii=False,
                indent=2
            )

    except Exception as error:

        print("Error guardando historial:", error)


# ==========================================================
# 7. MOTOR DE RESPUESTAS
# ==========================================================

def answer_question(question):

    q = question.lower().strip()

    # Identidad
    if "quien eres" in q or "quién eres" in q:
        return (
            "Soy MiniGPT 6.1, un asistente experimental "
            "creado con Python y PyTorch. "
            "Puedo consultar documentos y resolver calculos."
        )

    # Saludos
    if any(x in q for x in ["hola", "buenas", "buenos dias"]):
        return "¡Hola! Soy MiniGPT 6.1. ¿En que puedo ayudarte?"

    # Consultar documentos disponibles
    if any(x in q for x in [
        "que sabes", "qué sabes",
        "que conoces", "qué conoces",
        "que documentos", "qué documentos"
    ]):

        if not documents:
            return "Todavia no tengo documentos cargados."

        names = sorted(set(d["file"] for d in documents))

        return (
            "Estos son mis documentos:\n\n"
            + "\n".join("• " + name for name in names)
            + f"\n\nTotal de fragmentos: {len(documents)}"
        )

    # Consultar fragmentos
    if "fragmentos" in q:

        if not documents:
            return "No tengo fragmentos cargados."

        response = "Estos son algunos fragmentos de mi memoria:\n\n"

        for i, doc in enumerate(documents[:10], 1):
            response += (
                f"{i}. Archivo: {doc['file']}\n"
                f"{doc['text'][:350]}\n\n"
            )

        response += (
            f"Total de fragmentos disponibles: {len(documents)}"
        )

        return response

    # Consultar temas disponibles
    if "dime un tema" in q or "que temas tienes" in q:

        if not documents:
            return "Todavia no tengo temas disponibles."

        names = sorted(set(d["file"] for d in documents))

        return (
            "Puedes preguntarme sobre estos documentos:\n\n"
            + "\n".join("• " + name for name in names)
        )

    # Calculadora
    expression = extract_math(question)

    if expression:

        try:
            result = safe_calculate(expression)
            return f"El resultado de {expression} es {result}."

        except Exception:
            return "No pude resolver esa operacion."

    # Busqueda documental
    results = search_knowledge(question)

    if results and results[0][0] >= 0.10:

        response = "Encontré esta informacion en mis documentos:\n\n"

        seen_texts = set()

        for score, doc in results:

            # Evitar mostrar fragmentos duplicados
            key = doc["text"][:100]

            if key in seen_texts:
                continue

            seen_texts.add(key)

            response += (
                f"• {doc['text']}\n"
                f"Fuente: {doc['file']}\n\n"
            )

        return response.strip()

    return (
        "No encontre informacion suficiente en mis documentos "
        "para responder esa pregunta. Puedes agregar mas "
        "informacion a la carpeta conocimientos."
    )


# ==========================================================
# 8. INTERFAZ GRAFICA
# ==========================================================

root = tk.Tk()

root.title("MiniGPT 6.1")
root.geometry("760x620")
root.minsize(520, 450)

root.configure(bg="#172033")


title = tk.Label(
    root,
    text="MINIGPT 6.1",
    font=("Arial", 21, "bold"),
    fg="white",
    bg="#172033"
)

title.pack(pady=(12, 2))


subtitle = tk.Label(
    root,
    text="Asistente con memoria documental",
    font=("Arial", 10),
    fg="#aab8d0",
    bg="#172033"
)

subtitle.pack(pady=(0, 10))


chat = scrolledtext.ScrolledText(
    root,
    wrap=tk.WORD,
    font=("Arial", 11),
    bg="#f4f6fa",
    fg="#172033",
    padx=12,
    pady=12
)

chat.pack(
    fill=tk.BOTH,
    expand=True,
    padx=14,
    pady=8
)

chat.configure(state=tk.DISABLED)


def show_message(speaker, message):

    chat.configure(state=tk.NORMAL)

    chat.insert(
        tk.END,
        f"{speaker}:\n{message}\n\n"
    )

    chat.see(tk.END)

    chat.configure(state=tk.DISABLED)


# ==========================================================
# 9. ENVIAR MENSAJES
# ==========================================================

def send_message():

    question = entry.get().strip()

    if not question:
        return

    entry.delete(0, tk.END)

    send_button.configure(state=tk.DISABLED)

    show_message("Tú", question)

    history.append({
        "role": "user",
        "text": question
    })

    def work():

        try:

            response = answer_question(question)

        except Exception as error:

            response = f"Error procesando pregunta: {error}"

        history.append({
            "role": "assistant",
            "text": response
        })

        save_history()

        def update_ui():

            show_message(
                "MiniGPT 6.1",
                response
            )

            send_button.configure(state=tk.NORMAL)

            entry.focus_set()

        root.after(0, update_ui)

    threading.Thread(
        target=work,
        daemon=True
    ).start()


# ==========================================================
# 10. ACTUALIZAR DOCUMENTOS
# ==========================================================

def refresh_knowledge():

    load_documents()

    if documents:

        messagebox.showinfo(
            "Memoria actualizada",
            f"Se cargaron {len(documents)} fragmentos "
            "de conocimiento."
        )

    else:

        messagebox.showwarning(
            "Sin documentos",
            "No se encontraron archivos TXT.\n\n"
            f"Revisa esta carpeta:\n{KNOWLEDGE_DIR}"
        )


# ==========================================================
# 11. MOSTRAR HISTORIAL
# ==========================================================

def show_history():

    if not history:

        messagebox.showinfo(
            "Historial",
            "Todavia no hay conversaciones."
        )

        return

    window = tk.Toplevel(root)

    window.title("Historial de MiniGPT 6.1")
    window.geometry("560x450")

    area = scrolledtext.ScrolledText(
        window,
        wrap=tk.WORD,
        font=("Arial", 10)
    )

    area.pack(
        fill=tk.BOTH,
        expand=True,
        padx=10,
        pady=10
    )

    for item in history[-100:]:

        role = (
            "Tú"
            if item["role"] == "user"
            else "MiniGPT"
        )

        area.insert(
            tk.END,
            f"{role}:\n{item['text']}\n\n"
        )

    area.configure(state=tk.DISABLED)


# ==========================================================
# 12. BARRA DE ENTRADA
# ==========================================================

bottom = tk.Frame(
    root,
    bg="#172033"
)

bottom.pack(
    fill=tk.X,
    padx=14,
    pady=10
)


entry = tk.Entry(
    bottom,
    font=("Arial", 12),
    bg="white",
    fg="#172033"
)

entry.pack(
    side=tk.LEFT,
    fill=tk.X,
    expand=True,
    ipady=9,
    padx=(0, 8)
)

entry.bind(
    "<Return>",
    lambda event: send_message()
)


send_button = tk.Button(
    bottom,
    text="Enviar",
    command=send_message,
    bg="#3478f6",
    fg="white",
    font=("Arial", 10, "bold"),
    relief=tk.FLAT,
    padx=18,
    pady=9
)

send_button.pack(side=tk.RIGHT)


# ==========================================================
# 13. BOTONES ADICIONALES
# ==========================================================

actions = tk.Frame(
    root,
    bg="#172033"
)

actions.pack(pady=(0, 12))


tk.Button(
    actions,
    text="Actualizar conocimientos",
    command=refresh_knowledge,
    bg="#263650",
    fg="white",
    relief=tk.FLAT,
    padx=10,
    pady=7
).pack(side=tk.LEFT, padx=5)


tk.Button(
    actions,
    text="Ver historial",
    command=show_history,
    bg="#263650",
    fg="white",
    relief=tk.FLAT,
    padx=10,
    pady=7
).pack(side=tk.LEFT, padx=5)


# ==========================================================
# 14. INICIO DEL PROGRAMA
# ==========================================================

load_history()
load_documents()
create_or_load_model()

show_message(
    "MiniGPT 6.1",
    "¡Hola! Estoy listo para trabajar.\n"
    "Puedes preguntarme sobre tus documentos TXT."
)

for item in history[-20:]:

    role = (
        "Tú"
        if item["role"] == "user"
        else "MiniGPT 6.1"
    )

    show_message(
        role,
        item["text"]
    )

entry.focus_set()

root.mainloop()
