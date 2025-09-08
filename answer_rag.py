#!/usr/bin/env python3
"""
Refactored RAG Answer System
Improved structure, error handling, and response quality.
"""

import os
import sys
import logging
from typing import List, Dict, Any, Tuple, Set, Optional
from dataclasses import dataclass
from enum import Enum
import json
import hashlib
import time

import httpx
import regex as re_regex
from fastapi import FastAPI, Query, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
import re
from pydantic import BaseModel, Field
from openai import OpenAI, BadRequestError, NotFoundError

# Configure logging - level can be controlled by environment
log_level = os.environ.get("LOG_LEVEL", "WARNING").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.WARNING),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QuestionType(Enum):
    """Types of questions for specialized handling."""
    GENERAL = "general"
    SPEED = "speed"
    DEADLINE = "deadline"
    DEFINITION = "definition"
    PROCEDURE = "procedure"


@dataclass
class RAGConfig:
    """Configuration for RAG system."""
    retriever_url: str = os.environ.get(
        "RETRIEVER_URL", "http://127.0.0.1:8000/search")
    openai_model: str = os.environ.get("RAG_CHAT_MODEL", "gpt-5-mini")
    openai_api_key: str = os.environ.get("OPENAI_API_KEY", "")

    # Search parameters
    top_k: int = int(os.environ.get("RAG_TOPK", "5"))  # Reduced from 5
    chars_per_hit: int = int(os.environ.get(
        "RAG_CHARS_PER_HIT", "500"))  # Reduced from 500

    # Response limits
    max_completion_tokens: int = 4000

    # Cache settings
    cache_ttl: int = int(os.environ.get("CACHE_TTL", "3600"))  # 1 hour


class ResponseCache:
    """Simple in-memory response cache."""

    def __init__(self, ttl: int = 3600):
        self.ttl = ttl
        self.cache = {}

    def _get_key(self, question: str) -> str:
        """Generate cache key from question."""
        return hashlib.md5(question.lower().strip().encode()).hexdigest()

    def get(self, question: str) -> Optional[Dict[str, Any]]:
        """Get cached response if exists and not expired."""
        key = self._get_key(question)
        if key in self.cache:
            cached_time, response = self.cache[key]
            if time.time() - cached_time < self.ttl:
                return response
            else:
                # Remove expired entry
                del self.cache[key]
        return None

    def set(self, question: str, response: Dict[str, Any]) -> None:
        """Cache response."""
        key = self._get_key(question)
        self.cache[key] = (time.time(), response)

    def clear(self) -> None:
        """Clear all cached responses."""
        self.cache.clear()

    def size(self) -> int:
        """Get number of cached items."""
        return len(self.cache)


class PromptTemplates:
    """Improved prompt templates for better responses."""

    SYSTEM_PROMPT = """Odpowiadaj zwięźle na pytania o przepisy kolejowe używając TYLKO informacji ze źródeł. Odpowiadaj po polsku."""

    USER_TEMPLATE = """PYTANIE: {question}

ŹRÓDŁA:
{context}

Odpowiedz zwięźle używając informacji ze źródeł. Jeśli nie ma odpowiedzi w źródłach, napisz to wprost."""


class TextProcessor:
    """Enhanced text processing utilities."""

    # Improved regex patterns
    WORD_PATTERN = re_regex.compile(r"\p{L}+", re_regex.UNICODE)
    SENTENCE_PATTERN = re_regex.compile(
        r"(?<=[\.\?!…])\s+(?=[\p{Lu}ĄĆĘŁŃÓŚŹŻ])",
        re_regex.UNICODE
    )

    # Question type detection patterns
    SPEED_PATTERN = re.compile(
        r"\b(prędkość|dozwolon\w*|dopuszczaln\w*|maksymaln\w*|ograniczon\w*|vmax|szybkość)\b",
        re.IGNORECASE
    )
    DEADLINE_PATTERN = re.compile(
        r"\b(termin|najpóźniej|w\s+terminie|deadline|do\s+kiedy|kiedy\s+należy)\b",
        re.IGNORECASE
    )
    DEFINITION_PATTERN = re.compile(
        r"\b(co\s+to\s+jest|czym\s+jest|definicja|znaczenie|oznacza)\b",
        re.IGNORECASE
    )
    PROCEDURE_PATTERN = re.compile(
        r"\b(jak\s+(?:się|można|należy)|procedura|proces|kroki|instrukcja|sposób)\b",
        re.IGNORECASE
    )

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Extract word tokens from text."""
        return [w.lower() for w in TextProcessor.WORD_PATTERN.findall(text or "")]

    @staticmethod
    def split_sentences(text: str) -> List[str]:
        """Split text into sentences."""
        text = (text or "").replace("\n", " ")
        return [s.strip() for s in TextProcessor.SENTENCE_PATTERN.split(text) if s.strip()]

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text."""
        text = (text or "").strip()
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        return text

    @staticmethod
    def shorten_text(text: str, limit: int) -> str:
        """Intelligently shorten text to word boundaries."""
        text = TextProcessor.clean_text(text)
        if len(text) <= limit:
            return text

        # Try to cut at sentence boundary first
        sentences = TextProcessor.split_sentences(text)
        result = ""
        for sent in sentences:
            if len(result + sent) <= limit:
                result += sent + " "
            else:
                # If adding this sentence would exceed limit, but result is empty,
                # add at least part of the sentence
                if not result and len(sent) > 100:
                    # Add up to limit but try to end at word boundary
                    cutoff = text[:limit].rfind(' ')
                    if cutoff > limit * 0.8:  # Only use word boundary if it's not too far back
                        result = text[:cutoff]
                    else:
                        result = text[:limit]
                break

        if result:
            return result.strip()

        # Fall back to word boundary
        cutoff = text[:limit].rfind(' ')
        if cutoff > limit * 0.8:  # Only use word boundary if it's not too far back
            return text[:cutoff]
        else:
            return text[:limit]

    @staticmethod
    def detect_question_type(question: str) -> QuestionType:
        """Detect the type of question for specialized handling."""
        question_lower = question.lower()

        if TextProcessor.SPEED_PATTERN.search(question):
            return QuestionType.SPEED
        elif TextProcessor.DEADLINE_PATTERN.search(question):
            return QuestionType.DEADLINE
        elif TextProcessor.DEFINITION_PATTERN.search(question):
            return QuestionType.DEFINITION
        elif TextProcessor.PROCEDURE_PATTERN.search(question):
            return QuestionType.PROCEDURE
        else:
            return QuestionType.GENERAL


class DocumentRetriever:
    """Handle document retrieval from the retriever service."""

    def __init__(self, config: RAGConfig):
        self.config = config

    def retrieve_documents(self, question: str) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for the question."""
        try:
            # Reduced timeout for faster failure detection
            with httpx.Client(timeout=30) as client:
                response = client.get(
                    self.config.retriever_url,
                    params={"q": question}
                )
                response.raise_for_status()
                data = response.json()
                hits = data.get("hits", [])[:self.config.top_k]
                logger.info(f"Retriever returned {len(hits)} hits")
                return hits

        except httpx.RequestError as e:
            logger.error(f"Error connecting to retriever service: {e}")
            raise HTTPException(
                status_code=503,
                detail="Retriever service unavailable"
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"Retriever service error: {e}")
            raise HTTPException(
                status_code=e.response.status_code,
                detail="Error retrieving documents"
            )


class AnswerGenerator:
    """Generate answers using OpenAI with improved prompting."""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.client = OpenAI(
            api_key=config.openai_api_key,
            timeout=30.0  # 30 second timeout
        )

    def format_citation(self, hit: Dict[str, Any]) -> str:
        """Format citation for a document hit."""
        title = hit.get("title") or "Dokument"
        page = hit.get("page")
        chunk_id = hit.get("chunk_id")

        if page:
            return f"[{title}, str. {page}] ({chunk_id})"
        else:
            return f"[{title}] ({chunk_id})"

    def build_context(self, hits: List[Dict[str, Any]]) -> str:
        """Build context from document hits with improved formatting."""
        if not hits:
            return "Brak dostępnych źródeł."

        blocks = []
        for i, hit in enumerate(hits, 1):
            title = hit.get('title', 'Dokument')
            page = hit.get('page', 'N/A')
            source_path = hit.get('source_path', '')
            chunk_id = hit.get('chunk_id', '')
            text = hit.get('text', '')

            # Create clean source header
            header = f"Źródło {i}: {title}"
            if page != 'N/A':
                header += f" | strona: {page}"
            if source_path:
                header += f" | plik: {source_path}"
            header += f"\nID: {chunk_id}"

            # Clean and shorten text
            clean_text = TextProcessor.shorten_text(
                text, self.config.chars_per_hit)

            blocks.append(f"{header}\nTekst:\n{clean_text}")

        return "\n\n".join(blocks)

    def generate_llm_answer(self, question: str, hits: List[Dict[str, Any]]) -> str:
        """Generate answer using LLM."""
        if not self.config.openai_api_key:
            logger.warning("No OpenAI API key provided")
            return ""

        try:
            context = self.build_context(hits)

            messages = [
                {"role": "system", "content": PromptTemplates.SYSTEM_PROMPT},
                {"role": "user", "content": PromptTemplates.USER_TEMPLATE.format(
                    question=question,
                    context=context
                )}
            ]

            logger.info(f"Sending prompt with system: {len(
                PromptTemplates.SYSTEM_PROMPT)} chars, user: {len(messages[1]['content'])} chars")

            # Try with max_completion_tokens first (newer models)
            try:
                response = self.client.chat.completions.create(
                    model=self.config.openai_model,
                    messages=messages,
                    max_completion_tokens=self.config.max_completion_tokens,
                    reasoning_effort="minimal"
                )
            except BadRequestError as e:
                error_str = str(e).lower()
                logger.warning(
                    f"BadRequestError with max_completion_tokens: {e}")

                if "max_completion_tokens" in error_str or "max_tokens" in error_str:
                    logger.info("Trying without max_completion_tokens")
                    response = self.client.chat.completions.create(
                        model=self.config.openai_model,
                        messages=messages,
                        max_tokens=self.config.max_completion_tokens,
                        reasoning_effort="minimal"
                    )
                else:
                    raise

            if response.choices:
                choice = response.choices[0]
                content = choice.message.content
                finish_reason = choice.finish_reason
                logger.info(f"OpenAI response content: {
                            content[:200] if content else 'None'}...")
                logger.info(f"Finish reason: {finish_reason}")

                if content:
                    return content.strip()
                else:
                    logger.warning(f"OpenAI response content is empty. Finish reason: {
                                   finish_reason}")
                    if finish_reason == "length":
                        # Return partial response for length limits
                        return "Odpowiedź została skrócona ze względu na limit długości. Proszę sprecyzować pytanie."
            else:
                logger.warning("No choices in OpenAI response")

        except NotFoundError:
            logger.error(f"Model '{self.config.openai_model}' not available")
            raise HTTPException(
                status_code=400,
                detail=f"Model '{self.config.openai_model}' not available. "
                "Set RAG_CHAT_MODEL to an available model."
            )
        except Exception as e:
            logger.error(f"Error generating LLM answer: {e}")

        return ""


class RAGAnswerer:
    """Main RAG answering system."""

    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self.retriever = DocumentRetriever(self.config)
        self.generator = AnswerGenerator(self.config)
        self.cache = ResponseCache(ttl=self.config.cache_ttl)

        # Validate configuration
        if not self.config.openai_api_key:
            logger.warning(
                "OpenAI API key not set - LLM generation will be disabled")

    def ask(self, question: str) -> Dict[str, Any]:
        """Answer a question using RAG with caching."""
        logger.info(f"Processing question: {question[:100]}...")

        # Check cache first
        cached_response = self.cache.get(question)
        if cached_response:
            logger.info("Returning cached response")
            return cached_response

        try:
            # 1. Retrieve relevant documents
            hits = self.retriever.retrieve_documents(question)

            if not hits:
                return {
                    "answer": "Nie znaleziono żadnych dokumentów odnoszących się do tego pytania.",
                    "hits": [],
                    "method": "no_results"
                }

            # 2. Detect question type for specialized handling
            question_type = TextProcessor.detect_question_type(question)
            logger.info(f"Detected question type: {question_type.value}")

            # 3. Generate answer using LLM only
            llm_answer = self.generator.generate_llm_answer(question, hits)

            if llm_answer:
                logger.info("Using LLM-generated answer")
                final_answer = llm_answer
                method = "llm"
                used_hits = hits[:4]  # Show top hits for LLM answers
            else:
                final_answer = "Nie udało się wygenerować odpowiedzi. Sprawdź konfigurację modelu AI."
                method = "error"
                used_hits = []

            # 5. Add sources section if not already present
            if "Źródła:" not in final_answer:
                sources_hits = used_hits if used_hits else hits[:4]
                # Remove duplicates
                unique_hits = []
                seen_ids = set()
                for hit in sources_hits:
                    chunk_id = hit.get("chunk_id")
                    if chunk_id and chunk_id not in seen_ids:
                        seen_ids.add(chunk_id)
                        unique_hits.append(hit)

                if unique_hits:
                    sources = "\n".join(
                        f"• {self.generator.format_citation(hit)}"
                        for hit in unique_hits
                    )
                    final_answer = f"{final_answer}\n\nŹródła:\n{sources}"

            logger.info(f"Generated answer using {method} method")

            result = {
                "answer": final_answer,
                "hits": hits,
                "method": method,
                "question_type": question_type.value
            }

            # Cache successful responses
            if method != "error":
                self.cache.set(question, result)

            return result

        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return {
                "answer": f"Wystąpił błąd podczas przetwarzania pytania: {str(e)}",
                "hits": [],
                "method": "error"
            }


# FastAPI Application
app = FastAPI(
    title="RAG Answerer (Enhanced)",
    description="Enhanced RAG system with improved answer quality and specialized question handling",
    version="2.0"
)

# Custom CORS middleware with regex support


class RegexCORSMiddleware:
    def __init__(self, app, allow_origin_regex: str = None, **kwargs):
        self.app = app
        self.allow_origin_regex = allow_origin_regex
        self.allow_credentials = kwargs.get("allow_credentials", True)
        self.allow_methods = kwargs.get(
            "allow_methods", ["GET", "POST", "PUT", "DELETE", "OPTIONS"])
        self.allow_headers = kwargs.get("allow_headers", ["*"])
        self.expose_headers = kwargs.get("expose_headers", ["*"])

        if allow_origin_regex:
            self.origin_pattern = re.compile(allow_origin_regex)
        else:
            self.origin_pattern = None

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        origin = None
        headers = dict(scope.get("headers", []))

        # Get origin header
        for name, value in headers.items():
            if name == b"origin":
                origin = value.decode("utf-8")
                break

        # Handle preflight OPTIONS request
        if scope["method"] == "OPTIONS" and origin:
            if self.origin_pattern and self.origin_pattern.match(origin):
                await send({
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [
                        (b"access-control-allow-origin", origin.encode()),
                        (b"access-control-allow-credentials", b"true"),
                        (b"access-control-allow-methods",
                         b", ".join(self.allow_methods).encode()),
                        (b"access-control-allow-headers", b"*"),
                        (b"access-control-max-age", b"600"),
                        (b"vary", b"Origin"),
                        (b"content-length", b"2"),
                        (b"content-type", b"text/plain; charset=utf-8"),
                    ]
                })
                await send({"type": "http.response.body", "body": b"OK"})
                return

        # Add CORS headers to response
        async def send_wrapper(message):
            if message["type"] == "http.response.start" and origin and self.origin_pattern and self.origin_pattern.match(origin):
                headers = list(message.get("headers", []))
                headers.extend([
                    (b"access-control-allow-origin", origin.encode()),
                    (b"access-control-allow-credentials", b"true"),
                    (b"access-control-expose-headers", b"*"),
                    (b"vary", b"Origin"),
                ])
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_wrapper)

# Configure CORS origin regex


def get_cors_origin_regex() -> str:
    """Get CORS origin regex from environment variable."""
    cors_origin_regex = os.environ.get("CORS_ORIGIN_REGEX", "")
    if not cors_origin_regex:
        raise ValueError("CORS_ORIGIN_REGEX environment variable must be set")

    logger.info(f"CORS origin regex configured: {cors_origin_regex}")
    return cors_origin_regex


# Add custom CORS middleware
cors_origin_regex = get_cors_origin_regex()
app.add_middleware(RegexCORSMiddleware, allow_origin_regex=cors_origin_regex)

# Global RAG instance
rag_answerer = RAGAnswerer()


class AskRequest(BaseModel):
    q: str = Field(..., description="Pytanie po polsku")


class AskResponse(BaseModel):
    answer: str = Field(..., description="Odpowiedź na pytanie")
    hits: List[Dict[str, Any]] = Field(..., description="Znalezione dokumenty")
    method: str = Field(..., description="Metoda generowania odpowiedzi")
    question_type: Optional[str] = Field(None, description="Typ pytania")


@app.get("/ask", response_model=AskResponse)
def ask_get(q: str = Query(..., description="Pytanie po polsku")):
    """Ask a question via GET request."""
    return rag_answerer.ask(q)


@app.post("/ask", response_model=AskResponse)
def ask_post(request: AskRequest):
    """Ask a question via POST request."""
    return rag_answerer.ask(request.q)


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "2.0"}


@app.options("/ask")
def options_ask():
    """Handle CORS preflight requests for /ask endpoint."""
    return {"message": "CORS preflight successful"}


@app.get("/test-cors")
def test_cors(response: Response):
    """Test endpoint to verify CORS is working."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    return {"message": "CORS test", "cors": "should work"}


def cli():
    """Command line interface."""
    if len(sys.argv) < 2:
        print('Użycie: python answer_rag_refactored.py "Twoje pytanie"')
        print('   lub: python answer_rag_refactored.py serve')
        sys.exit(1)

    if sys.argv[1] == "serve":
        import uvicorn
        logger.info("Starting RAG Answer API server...")
        uvicorn.run(app, host="0.0.0.0", port=8010)
    else:
        question = " ".join(sys.argv[1:]).strip()
        result = rag_answerer.ask(question)
        print(f"\n{result['answer']}")

        if logger.isEnabledFor(logging.INFO):
            print(f"\n[Metoda: {result['method']}, Typ: {
                  result.get('question_type', 'N/A')}]")


if __name__ == "__main__":
    cli()
