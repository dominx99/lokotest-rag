import os
import sys
import httpx
from typing import List, Dict, Any, Tuple, Set
from fastapi import FastAPI, Query
from pydantic import BaseModel
from openai import OpenAI, BadRequestError, NotFoundError
import regex as re  # << używamy 'regex' (obsługuje \p{...})

# ------------------- Config -------------------
RETRIEVER_URL = os.environ.get("RETRIEVER_URL", "http://127.0.0.1:8000/search")
OPENAI_MODEL  = os.environ.get("RAG_CHAT_MODEL", "gpt-5")  # podmień na realnie dostępny, np. 'gpt-4o-mini'
TOP_K         = int(os.environ.get("RAG_TOPK", "5"))        # 8 -> 5 (krótszy, trafniejszy kontekst)
# krótkie fragmenty, żeby model trzymał się sedna:
CHARS_PER_HIT = int(os.environ.get("RAG_CHARS_PER_HIT", "500"))  # 1000 -> 500

SYSTEM_PROMPT = """Jesteś asystentem RAG odpowiadającym po polsku.
Zasady (stosuj rygorystycznie):
- Najpierw zwięzła odpowiedź: MAKS. 5 punktorów LUB 80–120 słów (jeśli punktorów nie da się użyć).
- Następnie sekcja „Źródła:” (tylko cytowane miejsca).
- Cytuj dokładne miejsca (tytuł, strona, chunk_id).
- Jeśli pytanie brzmi „co i w której instrukcji…”, wskaż nazwę dokumentu, paragraf/rozdział i stronę.
- Jeśli to procedura: wypunktuj kroki (maks. 7).
- Jeśli brakuje danych w źródłach, powiedz to wprost i zasugeruj doprecyzowanie.
- Nie wymyślaj faktów — korzystaj wyłącznie z dostarczonych fragmentów.
- Zawsze odpowiadaj krótko i konkretnie, unikaj dygresji i powtórzeń.
"""

USER_TEMPLATE = """Pytanie:
{question}

Dostępne źródła (skrót wycinków):
{context}

Instrukcje:
- Odpowiedz krótko (MAKS. 5 punktorów LUB 80–120 słów).
- Jeśli to definicja/zakres: wskaż dokument(y), rozdział/paragraf i stronę.
- Jeśli to procedura: wypunktuj kroki (maks. 7).
- Jeśli odpowiedź jest niepełna: powiedz, czego brakuje.
- Po odpowiedzi dodaj „Źródła:” z listą tylko tych cytowań, na które się powołałeś w treści.
"""

# ------------------- Helpers -------------------
def fmt_citation(hit: Dict[str, Any]) -> str:
    t = hit.get("title") or "Dokument"
    p = hit.get("page")
    cid = hit.get("chunk_id")
    return f"[{t}, str. {p}] ({cid})"

def _shorten(txt: str, limit: int) -> str:
    txt = (txt or "").strip().replace("\n", " ")
    return txt[:limit].rsplit(" ", 1)[0] if len(txt) > limit else txt

def build_context(hits: List[Dict[str, Any]], limit_per_hit: int) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        header = f"Źródło {i}: {h.get('title') or ''} | strona: {h.get('page')} | plik: {h.get('source_path')}\nID: {h.get('chunk_id')}"
        body = _shorten(h.get("text") or "", limit_per_hit)
        blocks.append(f"{header}\nTekst:\n{body}")
    return "\n\n".join(blocks)

def call_openai(messages: List[Dict[str, str]]) -> str:
    """
    Kompatybilne z nowszymi modelami czatowymi OpenAI.
    Limitujemy długość odpowiedzi (220 tokenów), bez ustawiania temperature.
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    model = OPENAI_MODEL
    base = dict(model=model, messages=messages)

    try:
        resp = client.chat.completions.create(**base, max_completion_tokens=220)
    except BadRequestError as e:
        msg = str(e).lower()
        if "max_completion_tokens" in msg and "unsupported" in msg:
            resp = client.chat.completions.create(**base, max_tokens=220)
        else:
            raise
    except NotFoundError:
        raise SystemExit(
            f"Model '{model}' nie jest dostępny dla Twojego klucza. "
            "Ustaw RAG_CHAT_MODEL na model, do którego masz dostęp (np. 'gpt-4o-mini')."
        )
    return resp.choices[0].message.content.strip() if resp.choices else ""

# ------------------- Extractive helpers -------------------
_WORD = re.compile(r"\p{L}+", re.UNICODE)

def _tokens(s: str):
    return [w.lower() for w in _WORD.findall(s or "")]

def _overlap_score(q_tokens, sent_tokens):
    if not q_tokens or not sent_tokens:
        return 0.0
    qs = set(q_tokens)
    inter = sum(1 for t in sent_tokens if t in qs)
    return inter / max(4, len(sent_tokens))

def _best_sentences(question: str, hits: List[Dict[str, Any]], per_hit: int = 2, max_total: int = 8):
    """Wybierz top zdania wg overlapu tokenów z pytaniem."""
    qtok = _tokens(question)
    picks: List[Tuple[str, Dict[str, Any]]] = []
    for h in hits:
        text = (h.get("text") or "").replace("\n", " ")
        # podział na zdania: znak końca + następna wielka litera (uwzgl. PL)
        sents = re.split(r"(?<=[\.\?!…])\s+(?=[\p{Lu}ĄĆĘŁŃÓŚŹŻ])", text)
        scored: List[Tuple[float, str]] = []
        for s in sents:
            st = _tokens(s)
            sc = _overlap_score(qtok, st)
            if sc > 0:
                scored.append((sc, s))
        scored.sort(key=lambda x: x[0], reverse=True)
        for sc, s in scored[:per_hit]:
            picks.append((s.strip(), h))
    # de-dup near-identical
    out: List[Tuple[str, Dict[str, Any]]] = []
    seen: Set[str] = set()
    for s, h in sorted(picks, key=lambda x: len(x[0])):  # krótsze zdania preferowane przy remisie
        key = s.lower()[:160]
        if key in seen:
            continue
        seen.add(key)
        out.append((s, h))
        if len(out) >= max_total:
            break
    return out

def _extract_deadlines(hits_list: List[Dict[str, Any]]):
    patt = re.compile(
        r"(termin\w*|najpóźniej|w\s+ciągu|do\s+dnia|nie\s+później\s+niż|w\s+terminie).{0,160}?"
        r"(\d{1,2}\.\d{1,2}\.\d{4}|\d+\s*(?:dni|godz(?:in)?|tyg(?:odni)?|mies(?:ięcy)?))",
        flags=re.IGNORECASE | re.UNICODE,
    )
    lines: List[Tuple[str, Dict[str, Any]]] = []
    for h in hits_list:
        text = (h.get("text") or "").replace("\n", " ")
        for m in patt.finditer(text):
            span = text[max(0, m.start()-80): m.end()+80]
            lines.append((span.strip(), h))
            if len(lines) >= 8:
                break
        if len(lines) >= 8:
            break
    return lines

def _extract_speeds(hits_list: List[Dict[str, Any]]):
    """
    Ekstrakcja fragmentów o prędkościach (dozwolona/dopuszczalna/maks./ograniczona/Vmax).
    Szuka słów-kluczy w pobliżu wartości liczbowych z jednostką km/h (różne warianty).
    Przykłady wykrywanych wzorców:
      - 'prędkość dozwolona 40 km/h', 'maks. 20 km/h', 'nie więcej niż 10 km/h'
      - 'Vmax = 40 km/h', 'V = 20 km/h'
      - zakresy: '5–10 km/h' albo '5-10 km/h'
      - 'km na godzinę', 'km na godz.', 'km/godz', 'km h' (rzadsze zapisy)
    """
    # słowa-klucze w kontekście prędkości
    speed_kw = r"(prędkość\w*|dozwolon\w*|dopuszczaln\w*|maksymaln\w*|ograniczon\w*|vmax|v\s*=?)"
    # jednostki km/h w różnych zapisach
    unit = r"(?:km\s*/?\s*h|km\s+na\s+g(?:odz(?:in(?:ę|y)?)?)?\.?)"
    # liczba lub zakres liczb
    num_or_range = r"(?:\d{1,3}(?:[.,]\d{1,2})?(?:\s*[-–]\s*\d{1,3}(?:[.,]\d{1,2})?)?)"
    # opcjonalne słowa ograniczające
    limit_words = r"(?:maks\.?|max|nie\s+więcej\s+niż|do|do\s+wartości|do\s+prędkości|≤|< =|<=)?"

    patt = re.compile(
        rf"{speed_kw}[^\.]{{0,160}}?{limit_words}\s*{num_or_range}\s*{unit}",
        flags=re.IGNORECASE | re.UNICODE,
    )
    lines: List[Tuple[str, Dict[str, Any]]] = []
    for h in hits_list:
        text = (h.get("text") or "").replace("\n", " ")
        for m in patt.finditer(text):
            span = text[max(0, m.start()-80): m.end()+80]
            # lekka normalizacja spacji
            span = re.sub(r"\s+", " ", span).strip()
            lines.append((span, h))
            if len(lines) >= 8:
                break
        if len(lines) >= 8:
            break
    # jeśli nic nie znaleziono, spróbuj bardziej liberalnie: dowolna liczba + km/h bez słów-kluczy
    if not lines:
        loose = re.compile(
            rf"{num_or_range}\s*{unit}",
            flags=re.IGNORECASE | re.UNICODE,
        )
        for h in hits_list:
            text = (h.get("text") or "").replace("\n", " ")
            for m in loose.finditer(text):
                span = text[max(0, m.start()-60): m.end()+60]
                span = re.sub(r"\s+", " ", span).strip()
                lines.append((span, h))
                if len(lines) >= 6:
                    break
            if len(lines) >= 6:
                break
    return lines

def _looks_like_deadline_q(question: str) -> bool:
    return bool(re.search(r"\btermin|najpóźniej|w\s+terminie\b", question, re.IGNORECASE))

def _looks_like_speed_q(question: str) -> bool:
    return bool(re.search(r"\b(prędkość|dozwolon\w*|dopuszczaln\w*|maksymaln\w*|ograniczon\w*|vmax)\b", question, re.IGNORECASE))

def _any_chunk_id_in_answer(answer: str, hits: List[Dict[str, Any]]) -> bool:
    if not answer or not hits:
        return False
    ids = [str(h.get("chunk_id") or "") for h in hits if h.get("chunk_id")]
    return any(i and i in answer for i in ids)

# ------------------- Main QA flow -------------------
def ask_once(question: str, top_k: int = TOP_K) -> Dict[str, Any]:
    # 1) Retrieve
    with httpx.Client(timeout=120) as c:
        r = c.get(RETRIEVER_URL, params={"q": question})
        r.raise_for_status()
        hits = r.json().get("hits", [])[:top_k]

    if not hits:
        return {"answer": "Brak wyników w bazie dla tego pytania.", "hits": []}

    # 2) Prompt build (skrót kontekstu, by utrzymać fokus)
    context = build_context(hits, CHARS_PER_HIT)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(question=question, context=context)},
    ]

    # 3) Spróbuj LLM
    answer = call_openai(messages)

    # 4) Warunek „fallback/augment”: jeśli odpowiedź jest pusta/krótka LUB brak cytowań chunk_id
    need_extractive = (not answer or len(answer.strip()) < 30) or (not _any_chunk_id_in_answer(answer, hits))

    used_hits: List[Dict[str, Any]] = []
    if need_extractive:
        bullets: List[str] = []
        used_set: Set[str] = set()

        # najpierw najistotniejsze zdania
        picks = _best_sentences(question, hits, per_hit=2, max_total=8)
        for s, h in picks:
            bullets.append(f"- {s}  ({fmt_citation(h)})")
            cid = str(h.get("chunk_id"))
            if cid and cid not in used_set:
                used_set.add(cid)
                used_hits.append(h)

        # jeśli pytanie wygląda na „terminy”, dodaj ekstrakcję terminów
        if _looks_like_deadline_q(question):
            deadlines = _extract_deadlines(hits)
            for span, h in deadlines:
                bullets.append(f"- {span}  ({fmt_citation(h)})")
                cid = str(h.get("chunk_id"))
                if cid and cid not in used_set:
                    used_set.add(cid)
                    used_hits.append(h)

        # jeśli pytanie wygląda na „prędkości”, dodaj ekstrakcję prędkości
        if _looks_like_speed_q(question):
            speeds = _extract_speeds(hits)
            for span, h in speeds:
                bullets.append(f"- {span}  ({fmt_citation(h)})")
                cid = str(h.get("chunk_id"))
                if cid and cid not in used_set:
                    used_set.add(cid)
                    used_hits.append(h)

        bullets = bullets[:8]

        if bullets:
            answer = "Najistotniejsze informacje ze źródeł:\n" + "\n".join(bullets)
        else:
            # brak sensownych dopasowań — wracamy do krótkiego komunikatu
            answer = answer.strip() if answer and len(answer.strip()) >= 30 else "Nie znajduję jednoznacznej odpowiedzi w dostarczonych źródłach."

    # 5) Blok „Źródła:” — zależnie od ścieżki
    if "Źródła:" not in answer:
        if used_hits:
            # w fallbacku pokazujemy tylko faktycznie użyte
            uniq = []
            seen_ids = set()
            for h in used_hits:
                cid = h.get("chunk_id")
                if cid and cid not in seen_ids:
                    uniq.append(h)
                    seen_ids.add(cid)
            citations = "\n".join(f"- {fmt_citation(h)}" for h in uniq)
        else:
            # dla odpowiedzi LLM (bez pewności jakie cytował) ograniczamy do 4 pierwszych
            citations = "\n".join(f"- {fmt_citation(h)}" for h in hits[:4])
        answer = f"{answer}\n\nŹródła:\n{citations}"

    return {"answer": answer, "hits": hits}

# ------------------- CLI / API -------------------
def cli():
    q = " ".join(sys.argv[1:]).strip()
    if not q:
        print('Użycie: python answer_rag.py "Twoje pytanie"  |  lub  python answer_rag.py serve')
        sys.exit(1)
    out = ask_once(q)
    print("\n" + out["answer"])

app = FastAPI(title="RAG Answerer (PL) — robust + speeds")

class AskIn(BaseModel):
    q: str

class AskOut(BaseModel):
    answer: str
    hits: list

@app.get("/ask", response_model=AskOut)
def ask_get(q: str = Query(..., description="Pytanie po polsku")):
    return ask_once(q)

@app.post("/ask", response_model=AskOut)
def ask_post(body: AskIn):
    return ask_once(body.q)

if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "serve":
        import uvicorn
        uvicorn.run(app, host="127.0.0.1", port=8010)
    else:
        cli()
