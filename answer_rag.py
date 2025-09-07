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

import httpx
import regex as re
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI, BadRequestError, NotFoundError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
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
    retriever_url: str = os.environ.get("RETRIEVER_URL", "http://127.0.0.1:8000/search")
    openai_model: str = os.environ.get("RAG_CHAT_MODEL", "gpt-4o-mini")
    openai_api_key: str = os.environ.get("OPENAI_API_KEY", "")
    
    # Search parameters
    top_k: int = int(os.environ.get("RAG_TOPK", "5"))
    chars_per_hit: int = int(os.environ.get("RAG_CHARS_PER_HIT", "600"))
    
    # Response limits
    max_completion_tokens: int = 300
    max_extractive_bullets: int = 8
    max_sentences_per_hit: int = 2
    
    # Context window management
    context_overlap_threshold: float = 0.15
    min_answer_length: int = 30


class PromptTemplates:
    """Improved prompt templates for better responses."""
    
    SYSTEM_PROMPT = """Jesteś ekspertem RAG odpowiadającym precyzyjnie po polsku na podstawie dostarczonych źródeł.

ZASADY ODPOWIADANIA:
1. STRUKTURA: Zwięzła odpowiedź (maks. 5 punktów LUB 100-150 słów) + sekcja "Źródła:"
2. DOKŁADNOŚĆ: Używaj TYLKO informacji z dostarczonych fragmentów
3. CYTOWANIE: Zawsze podaj konkretne źródła (dokument, strona, ID chunka)
4. PRZEJRZYSTOŚĆ: Jeśli informacje są niepełne, wskaż czego brakuje
5. SPECJALIZACJA: 
   - Procedury → wypunktuj kroki (maks. 7)
   - Definicje → podaj źródło i kontekst
   - Prędkości/terminy → cytuj dokładne wartości
6. JĘZYK: Używaj jasnego, technicznego polskiego

NIGDY nie wymyślaj faktów spoza dostarczonych źródeł."""

    USER_TEMPLATE = """PYTANIE: {question}

DOSTĘPNE ŹRÓDŁA:
{context}

INSTRUKCJA:
Odpowiedz zwięźle na pytanie używając TYLKO informacji z powyższych źródeł. 
Zakończ sekcją "Źródła:" z listą cytowanych dokumentów."""

    EXTRACTIVE_INTRO = "Na podstawie przeszukanych dokumentów:"


class TextProcessor:
    """Enhanced text processing utilities."""
    
    # Improved regex patterns
    WORD_PATTERN = re.compile(r"\p{L}+", re.UNICODE)
    SENTENCE_PATTERN = re.compile(
        r"(?<=[\.\?!…])\s+(?=[\p{Lu}ĄĆĘŁŃÓŚŹŻ])", 
        re.UNICODE
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
                break
        
        if result:
            return result.strip()
        
        # Fall back to word boundary
        return text[:limit].rsplit(" ", 1)[0] if " " in text[:limit] else text[:limit]
    
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


class ExtractiveSearch:
    """Enhanced extractive search for specialized queries."""
    
    def __init__(self, config: RAGConfig):
        self.config = config
    
    def calculate_overlap_score(self, query_tokens: List[str], text_tokens: List[str]) -> float:
        """Calculate semantic overlap between query and text."""
        if not query_tokens or not text_tokens:
            return 0.0
        
        query_set = set(query_tokens)
        intersection = sum(1 for token in text_tokens if token in query_set)
        
        # Improved scoring with length normalization
        base_score = intersection / len(text_tokens)
        length_bonus = min(1.0, len(text_tokens) / 20)  # Prefer longer sentences
        
        return base_score * (1 + length_bonus * 0.2)
    
    def extract_best_sentences(
        self, 
        question: str, 
        hits: List[Dict[str, Any]]
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Extract most relevant sentences from hits."""
        query_tokens = TextProcessor.tokenize(question)
        candidates = []
        
        for hit in hits:
            text = hit.get("text", "")
            sentences = TextProcessor.split_sentences(text)
            
            for sentence in sentences:
                if len(sentence) < 20:  # Skip very short sentences
                    continue
                
                sentence_tokens = TextProcessor.tokenize(sentence)
                score = self.calculate_overlap_score(query_tokens, sentence_tokens)
                
                if score > self.config.context_overlap_threshold:
                    candidates.append((score, sentence.strip(), hit))
        
        # Sort by score and deduplicate
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        seen = set()
        results = []
        for score, sentence, hit in candidates:
            # Simple deduplication by first 100 chars
            key = sentence.lower()[:100]
            if key not in seen:
                seen.add(key)
                results.append((sentence, hit))
                if len(results) >= self.config.max_extractive_bullets:
                    break
        
        return results
    
    def extract_speed_info(self, hits: List[Dict[str, Any]]) -> List[Tuple[str, Dict[str, Any]]]:
        """Extract speed-related information."""
        # Enhanced speed pattern with more context
        speed_pattern = re.compile(
            r"(prędkość\w*|dozwolon\w*|dopuszczaln\w*|maksymaln\w*|ograniczon\w*|vmax|v\s*=?)"
            r"[^\.\n]{0,200}?"
            r"(\d{1,3}(?:[.,]\d{1,2})?(?:\s*[-–]\s*\d{1,3}(?:[.,]\d{1,2})?)?)\s*"
            r"(?:km\s*/?\s*h|km\s+na\s+g(?:odz)?\.?)",
            re.IGNORECASE | re.UNICODE
        )
        
        results = []
        for hit in hits:
            text = hit.get("text", "").replace("\n", " ")
            for match in speed_pattern.finditer(text):
                start = max(0, match.start() - 100)
                end = min(len(text), match.end() + 100)
                context = TextProcessor.clean_text(text[start:end])
                results.append((context, hit))
                if len(results) >= 6:
                    break
            if len(results) >= 6:
                break
        
        return results
    
    def extract_deadline_info(self, hits: List[Dict[str, Any]]) -> List[Tuple[str, Dict[str, Any]]]:
        """Extract deadline-related information."""
        deadline_pattern = re.compile(
            r"(termin\w*|najpóźniej|w\s+ciągu|do\s+dnia|nie\s+później\s+niż|w\s+terminie)"
            r"[^\.\n]{0,200}?"
            r"(\d{1,2}\.\d{1,2}\.\d{4}|\d+\s*(?:dni|godz(?:in)?|tyg(?:odni)?|mies(?:ięcy)?))",
            re.IGNORECASE | re.UNICODE
        )
        
        results = []
        for hit in hits:
            text = hit.get("text", "").replace("\n", " ")
            for match in deadline_pattern.finditer(text):
                start = max(0, match.start() - 100)
                end = min(len(text), match.end() + 100)
                context = TextProcessor.clean_text(text[start:end])
                results.append((context, hit))
                if len(results) >= 6:
                    break
            if len(results) >= 6:
                break
        
        return results


class DocumentRetriever:
    """Handle document retrieval from the retriever service."""
    
    def __init__(self, config: RAGConfig):
        self.config = config
    
    def retrieve_documents(self, question: str) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for the question."""
        try:
            with httpx.Client(timeout=120) as client:
                response = client.get(
                    self.config.retriever_url, 
                    params={"q": question}
                )
                response.raise_for_status()
                data = response.json()
                return data.get("hits", [])[:self.config.top_k]
                
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
        self.client = OpenAI(api_key=config.openai_api_key)
    
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
            clean_text = TextProcessor.shorten_text(text, self.config.chars_per_hit)
            
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
            
            # Try with max_completion_tokens first (newer models)
            try:
                response = self.client.chat.completions.create(
                    model=self.config.openai_model,
                    messages=messages,
                    max_completion_tokens=self.config.max_completion_tokens,
                    temperature=0.1  # Low temperature for consistency
                )
            except BadRequestError as e:
                if "max_completion_tokens" in str(e).lower():
                    # Fall back to max_tokens for older models
                    response = self.client.chat.completions.create(
                        model=self.config.openai_model,
                        messages=messages,
                        max_tokens=self.config.max_completion_tokens,
                        temperature=0.1
                    )
                else:
                    raise
            
            if response.choices:
                return response.choices[0].message.content.strip()
            
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
    
    def has_citations_in_answer(self, answer: str, hits: List[Dict[str, Any]]) -> bool:
        """Check if answer contains citations to the provided hits."""
        if not answer or not hits:
            return False
        
        chunk_ids = [str(hit.get("chunk_id", "")) for hit in hits if hit.get("chunk_id")]
        return any(chunk_id and chunk_id in answer for chunk_id in chunk_ids)


class RAGAnswerer:
    """Main RAG answering system."""
    
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self.retriever = DocumentRetriever(self.config)
        self.generator = AnswerGenerator(self.config)
        self.extractive = ExtractiveSearch(self.config)
        
        # Validate configuration
        if not self.config.openai_api_key:
            logger.warning("OpenAI API key not set - LLM generation will be disabled")
    
    def ask(self, question: str) -> Dict[str, Any]:
        """Answer a question using RAG."""
        logger.info(f"Processing question: {question[:100]}...")
        
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
            
            # 3. Try LLM generation first
            llm_answer = self.generator.generate_llm_answer(question, hits)
            
            # 4. Evaluate if we need extractive fallback
            need_extractive = (
                not llm_answer or 
                len(llm_answer.strip()) < self.config.min_answer_length or
                not self.generator.has_citations_in_answer(llm_answer, hits)
            )
            
            used_hits = []
            final_answer = ""
            method = ""
            
            if need_extractive:
                logger.info("Using extractive search approach")
                
                # Build extractive answer based on question type
                bullets = []
                used_chunk_ids = set()
                
                # General sentence extraction
                sentences = self.extractive.extract_best_sentences(question, hits)
                for sentence, hit in sentences:
                    citation = self.generator.format_citation(hit)
                    bullets.append(f"• {sentence} ({citation})")
                    
                    chunk_id = hit.get("chunk_id")
                    if chunk_id and chunk_id not in used_chunk_ids:
                        used_chunk_ids.add(chunk_id)
                        used_hits.append(hit)
                
                # Specialized extraction based on question type
                if question_type == QuestionType.SPEED:
                    speed_info = self.extractive.extract_speed_info(hits)
                    for info, hit in speed_info:
                        citation = self.generator.format_citation(hit)
                        bullets.append(f"• {info} ({citation})")
                        
                        chunk_id = hit.get("chunk_id")
                        if chunk_id and chunk_id not in used_chunk_ids:
                            used_chunk_ids.add(chunk_id)
                            used_hits.append(hit)
                
                elif question_type == QuestionType.DEADLINE:
                    deadline_info = self.extractive.extract_deadline_info(hits)
                    for info, hit in deadline_info:
                        citation = self.generator.format_citation(hit)
                        bullets.append(f"• {info} ({citation})")
                        
                        chunk_id = hit.get("chunk_id")
                        if chunk_id and chunk_id not in used_chunk_ids:
                            used_chunk_ids.add(chunk_id)
                            used_hits.append(hit)
                
                # Limit bullets and create final answer
                bullets = bullets[:self.config.max_extractive_bullets]
                
                if bullets:
                    final_answer = f"{PromptTemplates.EXTRACTIVE_INTRO}\n" + "\n".join(bullets)
                    method = "extractive"
                else:
                    final_answer = "Nie udało się znaleźć odpowiedzi w dostarczonych dokumentach."
                    method = "no_match"
            
            else:
                logger.info("Using LLM-generated answer")
                final_answer = llm_answer
                method = "llm"
                used_hits = hits[:4]  # Show top hits for LLM answers
            
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
            
            return {
                "answer": final_answer,
                "hits": hits,
                "method": method,
                "question_type": question_type.value
            }
            
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


def cli():
    """Command line interface."""
    if len(sys.argv) < 2:
        print('Użycie: python answer_rag_refactored.py "Twoje pytanie"')
        print('   lub: python answer_rag_refactored.py serve')
        sys.exit(1)
    
    if sys.argv[1] == "serve":
        import uvicorn
        logger.info("Starting RAG Answer API server...")
        uvicorn.run(app, host="127.0.0.1", port=8010)
    else:
        question = " ".join(sys.argv[1:]).strip()
        result = rag_answerer.ask(question)
        print(f"\n{result['answer']}")
        
        if logger.isEnabledFor(logging.INFO):
            print(f"\n[Metoda: {result['method']}, Typ: {result.get('question_type', 'N/A')}]")


if __name__ == "__main__":
    cli()