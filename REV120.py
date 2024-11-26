import os
import time
import json
import logging
import numpy as np
import redis
import asyncio
import pytz
import threading
import schedule
import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from contextlib import contextmanager
from dotenv import load_dotenv

# Advanced ML and NLP Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer
from transformers import (
    T5Tokenizer, 
    T5ForConditionalGeneration, 
    AutoModelForSequenceClassification, 
    AutoTokenizer
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

# Telegram and Communication Libraries
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, CallbackContext

# Additional Libraries
import jiwer
import spacy
import textacy
from tenacity import retry, wait_fixed, stop_after_attempt
from spacy_langdetect import LanguageDetector

# Advanced Configuration and Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables securely
load_dotenv()
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.1')
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))

@dataclass
class ConversationEntry:
    message: str
    response: str
    embedding: np.ndarray
    timestamp: float
    context_used: List[Dict[str, Any]] = field(default_factory=list)
    sentiment: float = 0.0
    topic: str = ''
    language: str = ''
    intent: str = ''
    complexity_score: float = 0.0

class AdvancedContextualMemory:
    def __init__(self, redis_client):
        self.redis_client = redis_client
        self.short_term_memory = defaultdict(list)
        self.nlp = spacy.load('xx_ent_wiki_sm')  # Advanced NLP processing
        
        # Add language detection
        if 'language_detector' not in self.nlp.pipe_names:
            self.nlp.add_pipe('language_detector', last=True)
        
        self.MAX_SHORT_TERM_MEMORY = 20  # Increase the number of short-term memories
        self.MAX_MEMORY_SIZE_MB = 100  # Limit memory size to 100 MB
    
    def add_memory(self, chat_id: str, entry: ConversationEntry):
        # Enhanced memory storage with more metadata
        key = f"memory:{chat_id}:{int(time.time())}"
        memory_data = {
            'message': entry.message,
            'response': entry.response,
            'embedding': entry.embedding.tolist(),
            'timestamp': entry.timestamp,
            'context_used': entry.context_used,
            'sentiment': entry.sentiment,
            'topic': entry.topic,
            'language': entry.language,
            'intent': entry.intent,
            'complexity_score': entry.complexity_score
        }
        
        self.redis_client.setex(key, 30 * 24 * 3600, json.dumps(memory_data, ensure_ascii=False))
        
        # Maintain short-term memory with intelligent pruning
        self.short_term_memory[chat_id].append(entry)
        if len(self.short_term_memory[chat_id]) > self.MAX_SHORT_TERM_MEMORY:
            self.short_term_memory[chat_id].pop(0)
    
    def analyze_language_and_intent(self, text: str) -> Dict[str, str]:
        doc = self.nlp(text)
        
        # Language detection
        language = doc._.language['language']  # Use the correct attribute
        
        # Intent classification using spaCy's textcat
        # (Assuming pre-trained intent classification model)
        intent = self.classify_intent(text)
        
        return {
            'language': language,
            'intent': intent
        }
    
    def classify_intent(self, text: str) -> str:
        # Placeholder for advanced intent classification
        # In a real implementation, this would use a trained model
        intents = ['question', 'request', 'statement', 'greeting', 'farewell']
        # Basic heuristic-based intent detection
        if '?' in text:
            return 'question'
        elif any(greeting in text.lower() for greeting in ['hi', 'hello', 'سلام']):
            return 'greeting'
        else:
            return 'statement'
    
    def clear_old_memories(self, days=30):
        # Delete old memories
        current_time = time.time()
        pattern = "memory:*"
        
        for key in self.redis_client.scan_iter(pattern):
            data = json.loads(self.redis_client.get(key))
            if current_time - data['timestamp'] > days * 24 * 3600:
                self.redis_client.delete(key)
    
    @contextmanager
    def manage_resources(self):
        try:
            # Allocate resources
            yield
        finally:
            # Release resources
            self.close_connections()
    
    def close_connections(self):
        self.redis_client.close()

class EnhancedEmbeddingModel:
    def __init__(self):
        self.base_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        self.topic_model = T5ForConditionalGeneration.from_pretrained('t5-base')
        self.topic_tokenizer = T5Tokenizer.from_pretrained('t5-base')
        
        # Sentiment and complexity analysis models
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
        self.sentiment_tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    
    def generate_embeddings(self, text: str) -> np.ndarray:
        base_embedding = self.base_model.encode(text, convert_to_tensor=True)
        topic_embedding = self.base_model.encode(
            self.extract_topic(text), 
            convert_to_tensor=True
        )
        
        return torch.cat([base_embedding, topic_embedding]).cpu().numpy()
    
    def extract_topic(self, text: str) -> str:
        # Advanced topic extraction with caching
        inputs = self.topic_tokenizer.encode(
            f"summarize: {text}", 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        )
        
        outputs = self.topic_model.generate(
            inputs, 
            max_length=50, 
            num_beams=4, 
            no_repeat_ngram_size=2, 
            early_stopping=True
        )
        
        return self.topic_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def analyze_sentiment_and_complexity(self, text: str) -> Dict[float, float]:
        # Sentiment analysis
        inputs = self.sentiment_tokenizer.encode(text, return_tensors='pt')
        outputs = self.sentiment_model(inputs)
        sentiment_raw = torch.nn.functional.softmax(outputs.logits, dim=1)
        sentiment = sentiment_raw.argmax().item() / 4  # Normalize to 0-1
        
        # Complexity score based on text attributes
        complexity = textacy.text_stats.flesch_kincaid_grade_level(text)
        normalized_complexity = min(max(complexity / 12, 0), 1)  # Normalize to 0-1
        
        return {
            'sentiment': sentiment,
            'complexity_score': normalized_complexity
        }

class AdvancedResponseGenerator:
    def __init__(self):
        self.response_cache = {}
        self.diversity_threshold = 0.4
    
    async def generate_response(
        self, 
        prompt: str, 
        context: List[Dict[str, Any]], 
        chat_id: str
    ) -> str:
        # Implement advanced response generation logic
        enhanced_prompt = self._build_contextual_prompt(prompt, context)
        
        try:
            response = await self._generate_llm_response(enhanced_prompt)
            
            # Post-processing for diversity and quality
            final_response = self._refine_response(
                response, 
                context, 
                chat_id
            )
            
            return final_response
        
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return "متأسفانه در تولید پاسخ خطایی رخ داد. لطفاً دوباره تلاش کنید."
    
    def _build_contextual_prompt(
        self, 
        base_prompt: str, 
        context: List[Dict[str, Any]]
    ) -> str:
        # Enhanced prompt engineering with context-aware construction
        prompt_parts = [f"سوال کنونی: {base_prompt}\n\n"]
        
        if context:
            prompt_parts.append("زمینه و سابقه مکالمه:\n")
            for entry in context[-5:]:  # Limit context window
                prompt_parts.append(
                    f"- در زمان {datetime.datetime.fromtimestamp(entry['timestamp']).strftime('%Y-%m-%d %H:%M')}:\n"
                    f"س: {entry.get('message', '')}\n"
                    f"ج: {entry.get('response', '')}\n"
                )
        
        prompt_parts.append("\nپاسخ دقیق و جامع خود را با توجه به زمینه مکالمه ارائه دهید:")
        return "\n".join(prompt_parts)
    
    async def _generate_llm_response(
        self, 
        prompt: str, 
        timeout: int = 45
    ) -> str:
        try:
            process = await asyncio.create_subprocess_exec(
                "ollama",
                "run",
                OLLAMA_MODEL,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=prompt.encode('utf-8')),
                timeout=timeout
            )
            
            if stderr:
                logger.error(f"LLM generation error: {stderr.decode()}")
            
            response = stdout.decode('utf-8').strip()
            return response or "پاسخی یافت نشد."
        
        except asyncio.TimeoutError:
            logger.warning("LLM response generation timed out")
            return "زمان پاسخ‌دهی به پایان رسید. لطفاً دوباره تلاش کنید."
        
        except Exception as e:
            logger.error(f"LLM response generation failed: {e}")
            return "خطا در تولید پاسخ رخ داد."
    
    def _refine_response(
        self, 
        response: str, 
        context: List[Dict[str, Any]], 
        chat_id: str
    ) -> str:
        # Advanced response refinement
        if not context:
            return response
        
        recent_responses = [entry.get('response', '') for entry in context[-5:]]
        
        for recent in recent_responses:
            similarity = 1 - jiwer.wer(response, recent)
            if similarity > self.diversity_threshold:
                # Add contextual variation
                variation_prefixes = [
                    "به بیانی دیگر،", 
                    "از منظری دیگر،", 
                    "با تفصیل بیشتر،"
                ]
                return f"{np.random.choice(variation_prefixes)} {response}"
        
        return response

class RAGConfig:
    MAX_CONTEXT_WINDOW = 10
    EMBEDDING_MODEL = 'paraphrase-multilingual-mpnet-base-v2'
    REDIS_EXPIRE_TIME = 30 * 24 * 3600  # 30 days

# Main RAG System
class AdvancedHybridRAGSystem:
    def __init__(self):
        # Comprehensive initialization with enhanced error handling
        try:
            self.redis_client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                decode_responses=True
            )
            
            self.embedding_model = EnhancedEmbeddingModel()
            self.memory_system = AdvancedContextualMemory(self.redis_client)
            self.response_generator = AdvancedResponseGenerator()
            
            self._setup_maintenance_tasks()
            logger.info("Advanced Hybrid RAG System initialized successfully")
        
        except Exception as e:
            logger.critical(f"System initialization failed: {e}")
            raise
    
    def _setup_maintenance_tasks(self):
        # Scheduled maintenance with enhanced monitoring
        schedule.every(2).hours.do(self._cleanup_old_data)
        schedule.every(12).hours.do(self._optimize_memory)
        
        maintenance_thread = threading.Thread(
            target=self._run_maintenance_scheduler, 
            daemon=True
        )
        maintenance_thread.start()
    
    def _run_maintenance_scheduler(self):
        while True:
            schedule.run_pending()
            time.sleep(60)
    
    async def process_message(
        self, 
        chat_id: str, 
        message: str
    ) -> str:
        try:
            current_time = time.time()
            
            # Advanced message preprocessing
            language_intent = self.memory_system.analyze_language_and_intent(message)
            
            # Generate embeddings with additional analysis
            embedding = self.embedding_model.generate_embeddings(message)
            topic = self.embedding_model.extract_topic(message)
            
            complexity_analysis = self.embedding_model.analyze_sentiment_and_complexity(message)
            
            # Get temporal context
            temporal_context = self._retrieve_context(chat_id, current_time)
            
            # Generate response
            response = await self.response_generator.generate_response(
                message,
                temporal_context,
                chat_id
            )
            
            # Store conversation with rich metadata
            entry = ConversationEntry(
                message=message,
                response=response,
                embedding=embedding,
                timestamp=current_time,
                context_used=temporal_context,
                sentiment=complexity_analysis['sentiment'],
                topic=topic,
                language=language_intent['language'],
                intent=language_intent['intent'],
                complexity_score=complexity_analysis['complexity_score']
            )
            
            self.memory_system.add_memory(chat_id, entry)
            
            return response
        
        except Exception as e:
            logger.error(f"Message processing error: {e}")
            return "متأسفانه در پردازش پیام خطایی رخ داد."
    
    def _retrieve_context(
        self, 
        chat_id: str, 
        current_time: float, 
        window_size: int = 7200
    ) -> List[Dict[str, Any]]:
        # Implement advanced context retrieval
        context_entries = []
        pattern = f"memory:{chat_id}:*"
        
        for key in self.redis_client.scan_iter(pattern):
            memory_data = json.loads(self.redis_client.get(key))
            
            if current_time - memory_data['timestamp'] <= window_size:
                context_entries.append(memory_data)
        
        # Sort and limit context
        return sorted(
            context_entries, 
            key=lambda x: x['timestamp'], 
            reverse=True
        )[:RAGConfig.MAX_CONTEXT_WINDOW]
    
    def _cleanup_old_data(self):
        try:
            current_time = time.time()
            pattern = "memory:*"
            
            for key in self.redis_client.scan_iter(pattern):
                data = json.loads(self.redis_client.get(key))
                if current_time - data['timestamp'] > 60 * 24 * 3600:  # 60 days
                    self.redis_client.delete(key)
        
        except Exception as e:
            logger.error(f"Data cleanup failed: {e}")
    
    def _optimize_memory(self):
        try:
            self.response_generator.response_cache.clear()
        except Exception as e:
            logger.error(f"Memory optimization failed: {e}")
    
    async def __aenter__(self):
        # Connect to resources
        return self
    
    async def __aexit__(self, exc_type, exc, tb):
        # Close connections
        await self.close_connections()
    
    async def close_connections(self):
        self.redis_client.close()

# Telegram Bot Message Handler
async def handle_message(update: Update, context: CallbackContext):
    try:
        message = update.message.text
        chat_id = str(update.message.chat_id)
        
        # Process message with advanced RAG system
        response = await rag_system.process_message(chat_id, message)
        
        # Format response with timestamp
        current_time = datetime.datetime.now(pytz.timezone('Asia/Tehran'))
        formatted_response = (
            f"{response}\n\n"
            f"🕒 زمان پاسخ: {current_time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        await update.message.reply_text(formatted_response)
    
    except Exception as e:
        logger.error(f"Message handling error: {e}")
        await update.message.reply_text(
            "متأسفانه در پردازش پیام شما خطایی رخ داد. لطفاً دوباره تلاش کنید."
        )

def main():
    try:
        global rag_system
        rag_system = AdvancedHybridRAGSystem()
        
        # Configure Telegram application
        app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
        app.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            handle_message
        ))
        
        # Add error handling and logging
        logger.info("Advanced Hybrid RAG System is starting...")
        app.run_polling(drop_pending_updates=True)
    
    except Exception as e:
        logger.critical(f"Application startup failed: {e}")
        raise

if __name__ == '__main__':
    main()