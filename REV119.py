import os
import time
import json
import logging
import numpy as np
import redis
import subprocess
import torch
import schedule
import datetime
from typing import List, Dict, Any
from dataclasses import dataclass
from collections import defaultdict
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, CallbackContext
import torch.nn as nn
import torch.optim as optim
import jiwer
from transformers import T5Tokenizer, T5ForConditionalGeneration
import asyncio
import pytz

# Load environment variables
load_dotenv()
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')

# Configure timezone for Iran
IRAN_TZ = pytz.timezone('Asia/Tehran')

@dataclass
class ConversationEntry:
    message: str
    response: str
    embedding: np.ndarray
    timestamp: float
    context_used: List[Dict]
    sentiment: float
    topic: str

class ContextualMemory:
    """Enhanced memory system with temporal and semantic understanding"""
    
    def __init__(self, redis_client):
        self.redis_client = redis_client
        self.short_term_memory = defaultdict(list)
        self.clustering_model = KMeans(n_clusters=5)
        self.topic_cache = {}
        
    def add_memory(self, chat_id: str, entry: ConversationEntry):
        """Add memory with temporal weighting"""
        # Store in Redis with TTL
        key = f"memory:{chat_id}:{int(time.time())}"
        self.redis_client.setex(
            key,
            30 * 24 * 3600,  # 30 days TTL
            json.dumps({
                'message': entry.message,
                'response': entry.response,
                'embedding': entry.embedding.tolist(),
                'timestamp': entry.timestamp,
                'context': entry.context_used,
                'sentiment': entry.sentiment,
                'topic': entry.topic
            }, ensure_ascii=False)
        )
        
        # Update short-term memory
        self.short_term_memory[chat_id].append(entry)
        if len(self.short_term_memory[chat_id]) > 10:
            self.short_term_memory[chat_id].pop(0)
            
    def get_temporal_context(self, chat_id: str, current_time: float, window_size: int = 3600) -> List[ConversationEntry]:
        """Retrieve context within temporal window"""
        recent_memories = []
        
        # Get keys for chat_id
        pattern = f"memory:{chat_id}:*"
        for key in self.redis_client.scan_iter(pattern):
            memory_data = json.loads(self.redis_client.get(key))
            if current_time - memory_data['timestamp'] <= window_size:
                recent_memories.append(ConversationEntry(**memory_data))
                
        return sorted(recent_memories, key=lambda x: x.timestamp, reverse=True)

class EnhancedEmbeddingModel:
    """Advanced embedding model with topic clustering and semantic analysis"""
    
    def __init__(self):
        self.base_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        self.topic_model = T5ForConditionalGeneration.from_pretrained('t5-small')
        self.topic_tokenizer = T5Tokenizer.from_pretrained('t5-small')
        self.topic_cache = {}
        
    def generate_embeddings(self, text: str) -> np.ndarray:
        """Generate enhanced embeddings with topic awareness"""
        base_embedding = self.base_model.encode(text, convert_to_tensor=True)
        topic = self.extract_topic(text)
        
        # Combine topic information with base embedding
        topic_embedding = self.base_model.encode(topic, convert_to_tensor=True)
        combined_embedding = torch.cat([base_embedding, topic_embedding])
        
        return combined_embedding.cpu().numpy()
        
    def extract_topic(self, text: str) -> str:
        """Extract topic using T5 model"""
        if text in self.topic_cache:
            return self.topic_cache[text]
            
        inputs = self.topic_tokenizer.encode(
            "summarize: " + text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )
        
        outputs = self.topic_model.generate(
            inputs,
            max_length=50,
            num_beams=4,
            no_repeat_ngram_size=2
        )
        
        topic = self.topic_tokenizer.decode(outputs[0], skip_special_tokens=True)
        self.topic_cache[text] = topic
        return topic

class ResponseGenerator:
    """Enhanced response generation with diversity and context awareness"""
    
    def __init__(self):
        self.response_cache = {}
        self.diversity_threshold = 0.3
        
    async def generate_response(
        self,
        prompt: str,
        context: List[ConversationEntry],
        chat_id: str
    ) -> str:
        """Generate diverse and contextually aware response"""
        
        # Check response cache to avoid repetition
        cache_key = f"{chat_id}:{prompt}"
        if cache_key in self.response_cache:
            cached_response = self.response_cache[cache_key]
            if time.time() - cached_response['timestamp'] < 3600:
                return self._modify_response(cached_response['response'])
        
        try:
            # Prepare enhanced prompt with temporal context
            enhanced_prompt = self._build_enhanced_prompt(prompt, context)
            
            # Generate base response using Llama
            base_response = await self._generate_llama_response(enhanced_prompt)
            
            # Apply diversity enhancement
            final_response = self._ensure_response_diversity(
                base_response,
                context,
                chat_id
            )
            
            # Cache the response
            self.response_cache[cache_key] = {
                'response': final_response,
                'timestamp': time.time()
            }
            
            return final_response
            
        except Exception as e:
            logging.error(f"Response generation failed: {e}")
            return "متأسفانه در تولید پاسخ خطایی رخ داد. لطفاً دوباره تلاش کنید."
            
    def _build_enhanced_prompt(
        self,
        base_prompt: str,
        context: List[ConversationEntry]
    ) -> str:
        """Build enhanced prompt with temporal and topical context"""
        prompt_parts = [f"سوال فعلی: {base_prompt}\n\n"]
        
        if context:
            prompt_parts.append("متن‌های مرتبط با زمینه:\n")
            for entry in context:
                iran_time = datetime.datetime.fromtimestamp(
                    entry.timestamp,
                    tz=IRAN_TZ
                ).strftime("%Y-%m-%d %H:%M")
                
                prompt_parts.append(
                    f"- در تاریخ {iran_time}:\n"
                    f"س: {entry.message}\n"
                    f"ج: {entry.response}\n"
                    f"موضوع: {entry.topic}\n"
                )
        
        prompt_parts.append("\nلطفاً با در نظر گرفتن متن‌های مرتبط و تاریخچه مکالمه پاسخ دهید:")
        return "\n".join(prompt_parts)
        
    async def _generate_llama_response(self, prompt: str, timeout: int = 30) -> str:
        """Generate response using Llama with improved error handling"""
        try:
            process = await asyncio.create_subprocess_exec(
                "powershell",
                "-Command",
                "ollama run llama3.1",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=prompt.encode()),
                timeout=timeout
            )
            
            if stderr:
                logging.error(f"Llama error: {stderr.decode()}")
            
            response = stdout.decode().strip()
            return response if response else "پاسخی یافت نشد."
            
        except asyncio.TimeoutError:
            if process:
                process.kill()
            return "زمان پاسخ‌دهی به پایان رسید. لطفاً دوباره تلاش کنید."
            
        except Exception as e:
            logging.error(f"Llama response generation failed: {e}")
            return "خطا در تولید پاسخ."
            
    def _ensure_response_diversity(
        self,
        response: str,
        context: List[ConversationEntry],
        chat_id: str
    ) -> str:
        """Ensure response diversity by checking against recent responses"""
        if not context:
            return response
            
        recent_responses = [entry.response for entry in context[-5:]]
        
        for recent in recent_responses:
            similarity = 1 - jiwer.wer(response, recent)
            if similarity > self.diversity_threshold:
                # Modify response to increase diversity
                return self._modify_response(response)
                
        return response
        
    def _modify_response(self, response: str) -> str:
        """Modify response to increase diversity while maintaining meaning"""
        # Add variation markers
        variations = [
            "به عبارت دیگر،",
            "به بیان ساده‌تر،",
            "در واقع،",
            "به طور خلاصه،"
        ]
        
        return f"{np.random.choice(variations)} {response}"

class EnhancedHybridRAGSystem:
    """Enhanced Hybrid RAG System with improved context understanding"""
    
    def __init__(self):
        self.redis_client = redis.Redis(
            host='localhost',
            port=6379,
            db=0,
            decode_responses=True
        )
        
        self.embedding_model = EnhancedEmbeddingModel()
        self.memory_system = ContextualMemory(self.redis_client)
        self.response_generator = ResponseGenerator()
        
        self.setup_maintenance_tasks()
        logging.info("Enhanced Hybrid RAG System initialized successfully")
        
    def setup_maintenance_tasks(self):
        """Setup periodic maintenance tasks"""
        schedule.every(1).hours.do(self.cleanup_old_data)
        schedule.every(6).hours.do(self.optimize_memory)
        
        threading.Thread(target=self._run_scheduler, daemon=True).start()
        
    def _run_scheduler(self):
        while True:
            schedule.run_pending()
            time.sleep(60)
            
    async def process_message(
        self,
        chat_id: str,
        message: str
    ) -> str:
        """Process incoming message with enhanced context understanding"""
        try:
            current_time = time.time()
            
            # Generate embeddings
            embedding = self.embedding_model.generate_embeddings(message)
            topic = self.embedding_model.extract_topic(message)
            
            # Get temporal context
            temporal_context = self.memory_system.get_temporal_context(
                chat_id,
                current_time
            )
            
            # Generate response
            response = await self.response_generator.generate_response(
                message,
                temporal_context,
                chat_id
            )
            
            # Store conversation
            entry = ConversationEntry(
                message=message,
                response=response,
                embedding=embedding,
                timestamp=current_time,
                context_used=temporal_context,
                sentiment=0.0,  # You could add sentiment analysis here
                topic=topic
            )
            
            self.memory_system.add_memory(chat_id, entry)
            
            return response
            
        except Exception as e:
            logging.error(f"Message processing failed: {e}")
            return "متأسفانه در پردازش پیام خطایی رخ داد. لطفاً دوباره تلاش کنید."
            
    def cleanup_old_data(self):
        """Cleanup old data and optimize storage"""
        try:
            current_time = time.time()
            pattern = "memory:*"
            
            for key in self.redis_client.scan_iter(pattern):
                data = json.loads(self.redis_client.get(key))
                if current_time - data['timestamp'] > 30 * 24 * 3600:  # 30 days
                    self.redis_client.delete(key)
                    
        except Exception as e:
            logging.error(f"Cleanup failed: {e}")
            
    def optimize_memory(self):
        """Optimize memory usage and clustering"""
        try:
            # Clear response cache
            self.response_generator.response_cache.clear()
            
            # Clear topic cache
            self.embedding_model.topic_cache.clear()
            
        except Exception as e:
            logging.error(f"Memory optimization failed: {e}")

# Message handlers
async def handle_message(update: Update, context: CallbackContext):
    """Handle incoming messages with enhanced context awareness"""
    try:
        message = update.message.text
        chat_id = str(update.message.chat_id)
        
        # Get current time in Iran timezone
        current_time = datetime.datetime.now(IRAN_TZ)
        
        response = await rag_system.process_message(chat_id, message)
        
        # Format response with timestamp
        formatted_response = (
            f"{response}\n\n"
            f"زمان پاسخ: {current_time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        await update.message.reply_text(formatted_response)
        
    except Exception as e:
        logging.error(f"Message handling failed: {e}")
        await update.message.reply_text(
            "متأسفانه در پردازش پیام شما خطایی رخ داد. لطفاً دوباره تلاش کنید."
        )

def main():
    """Application entry point"""
    try:
        global rag_system
        rag_system = EnhancedHybridRAGSystem()
        
        app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
        app.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            handle_message
        ))
        
        logging.info("Enhanced Hybrid RAG System is running")
        app.run_polling()
        
    except Exception as e:
        logging.critical(f"Application startup failed: {e}")
        raise

if __name__ == '__main__':
    main()