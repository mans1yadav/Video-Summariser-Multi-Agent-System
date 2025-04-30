import streamlit as st
from phi.agent import Agent
from phi.model.google import Gemini
import google.generativeai as genai
from pathlib import Path
import tempfile
from dotenv import load_dotenv
import os
from pydub import AudioSegment
import nltk
import logging
import whisper
from typing import Optional, Tuple, List, Dict
import numpy as np
from jiwer import wer
import psutil
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SystemEvaluator:
    def __init__(self):
        if 'evaluator_metrics' not in st.session_state:
            st.session_state.evaluator_metrics = {
                'response_times': [],
                'memory_usage': [],
                'transcription_accuracy': [],
                'query_relevance': [],
                'processing_times': {}
            }
        self.metrics_history = st.session_state.evaluator_metrics

    def track_metrics(self):
        """Display metrics in Streamlit"""
        if st.session_state.evaluator_metrics['response_times']:
            st.sidebar.subheader("System Metrics 📊")
            
            # Response Time
            avg_response = np.mean(st.session_state.evaluator_metrics['response_times'])
            st.sidebar.metric("Avg Response Time", f"{avg_response:.2f}s")
            
            # Memory Usage
            if st.session_state.evaluator_metrics['memory_usage']:
                current_memory = st.session_state.evaluator_metrics['memory_usage'][-1]
                st.sidebar.metric("Memory Usage", f"{current_memory:.1f}MB")
            
            # Query Relevance
            if st.session_state.evaluator_metrics['query_relevance']:
                avg_relevance = np.mean(st.session_state.evaluator_metrics['query_relevance'])
                st.sidebar.metric("Query Relevance", f"{avg_relevance:.2%}")
            
            # Processing Times
            if st.session_state.evaluator_metrics['processing_times']:
                st.sidebar.markdown("#### Processing Times")
                for stage, times in st.session_state.evaluator_metrics['processing_times'].items():
                    avg_time = np.mean(times)
                    st.sidebar.metric(f"{stage}", f"{avg_time:.2f}s")

    def measure_processing_time(self, stage: str):
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                
                if stage not in self.metrics_history['processing_times']:
                    self.metrics_history['processing_times'][stage] = []
                    
                self.metrics_history['processing_times'][stage].append(end_time - start_time)
                return result
            return wrapper
        return decorator

    def calculate_query_relevance(self, query: str, response: str, context: str) -> float:
        try:
            query_words = set(query.lower().split())
            response_words = set(response.lower().split())
            context_words = set(context.lower().split())
            
            query_response_overlap = len(query_words.intersection(response_words))
            context_response_overlap = len(context_words.intersection(response_words))
            
            relevance_score = (0.7 * query_response_overlap / len(query_words) +
                             0.3 * context_response_overlap / len(context_words))
            
            self.metrics_history['query_relevance'].append(relevance_score)
            return relevance_score
        except Exception as e:
            logger.error(f"Error calculating query relevance: {e}")
            return 0.0

    def track_memory_usage(self):
        try:
            process = psutil.Process(os.getpid())
            memory_usage = process.memory_info().rss / 1024 / 1024  # Convert to MB
            self.metrics_history['memory_usage'].append(memory_usage)
            return memory_usage
        except Exception as e:
            logger.error(f"Error tracking memory usage: {e}")
            return 0.0

# Modified VideoProcessor with evaluation
class VideoProcessor:
    def __init__(self, evaluator: SystemEvaluator):
        self.evaluator = evaluator

    @staticmethod
    def extract_audio(video_path: str) -> str:
        try:
            audio = AudioSegment.from_file(video_path)
            audio_path = tempfile.mktemp(suffix='.wav')
            audio.export(audio_path, format="wav")
            return audio_path
        except Exception as e:
            logger.error(f"Audio extraction failed: {str(e)}")
            raise Exception(f"Failed to extract audio: {str(e)}")

    def transcribe(self, audio_path: str) -> str:
        @self.evaluator.measure_processing_time('transcription')
        def _transcribe(path):
            try:
                model = whisper.load_model("base")
                result = model.transcribe(path)
                text = result["text"]
                if not text.strip():
                    raise Exception("No speech detected in audio")
                return text
            except Exception as e:
                logger.error(f"Transcription failed: {str(e)}")
                raise Exception(f"Failed to transcribe audio: {str(e)}")
        
        return _transcribe(audio_path)

# Modified query processing with evaluation
def process_query(query: str, summarizer_agent: Agent, evaluator: SystemEvaluator) -> str:
    start_time = time.time()
    try:
        context = st.session_state.conversation_history[-3:] if len(st.session_state.conversation_history) > 0 else []
        context_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in context])
        content = st.session_state.processed_text

        prompt = f"""As a video content analyst, consider the following:

Content: {content}

Previous conversation:
{context_str}

New question: {query}

Please provide a detailed response that:
1. Directly addresses the new question
2. Considers the context from previous conversation
3. References specific parts of the video content when relevant
"""

        response = summarizer_agent.run(prompt)
        
        # Calculate metrics
        end_time = time.time()
        evaluator.metrics_history['response_times'].append(end_time - start_time)
        evaluator.track_memory_usage()
        evaluator.calculate_query_relevance(query, response.content, content)
        
        return response.content
    except Exception as e:
        logger.error(f"Query processing failed: {str(e)}")
        raise

def main():
    st.set_page_config(page_title="Video Summeriser: Multi-Agent System", page_icon="🎥", layout="wide")
    st.title("Video Summeriser: Multi-Agent System 🎥")

    # Initialize evaluator
    evaluator = SystemEvaluator()
    
    # Initialize session state
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'processed_text' not in st.session_state:
        st.session_state.processed_text = None
    if 'video_processed' not in st.session_state:
        st.session_state.video_processed = False
    
    # Initialize application
    try:
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found")
        genai.configure(api_key=api_key)
        
        summarizer_agent = Agent(
            name="Video Content Analyst",
            model=Gemini(id="gemini-2.0-flash-exp"),
            tools=[],
            markdown=True,
        )
    except Exception as e:
        st.error(f"Failed to initialize application: {str(e)}")
        return

    # Display current metrics
    evaluator.track_metrics()
    
    # Video processing section
    video_file = st.file_uploader("Upload a video file", type=['mp4', 'mov', 'avi'])
    
    if video_file:
        temp_files = []
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
                temp_video.write(video_file.read())
                video_path = temp_video.name
                temp_files.append(video_path)
                st.session_state.video_path = video_path
            
            with st.container():
                st.video(video_path)
            
            if not st.session_state.video_processed:
                if st.button("📥 Process Video"):
                    with st.status("Processing video...") as status:
                        # Initialize video processor with evaluator
                        video_processor = VideoProcessor(evaluator)
                        
                        status.update(label="Extracting audio...")
                        audio_path = video_processor.extract_audio(video_path)
                        temp_files.append(audio_path)
                        
                        status.update(label="Transcribing audio...")
                        transcription = video_processor.transcribe(audio_path)
                        
                        st.session_state.processed_text = transcription
                        st.session_state.video_processed = True
                        status.update(label="Video processed successfully!", state="complete")
                        
                        # Track memory after processing
                        evaluator.track_memory_usage()
            
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
            logger.error(f"Video processing error: {str(e)}")
        finally:
            for file in temp_files:
                if file != st.session_state.video_path:
                    try:
                        Path(file).unlink(missing_ok=True)
                    except Exception as e:
                        logger.error(f"Failed to delete temporary file {file}: {str(e)}")
    
        # Conversation section
        if st.session_state.video_processed:
            st.divider()
            
            chat_col, button_col = st.columns([5, 1])
            
            with button_col:
                if st.button("🔄 New Chat"):
                    video_path = st.session_state.video_path
                    st.session_state.conversation_history = []
                    st.session_state.video_processed = False
                    st.session_state.processed_text = None
                    st.session_state.video_path = video_path
                    st.rerun()
            
            with chat_col:
                st.subheader("Ask questions about the video")
            
            for message in st.session_state.conversation_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            user_query = st.chat_input("Ask a question about the video...")
            
            if user_query:
                with st.chat_message("user"):
                    st.markdown(user_query)
                st.session_state.conversation_history.append({"role": "user", "content": user_query})
                
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            response = process_query(user_query, summarizer_agent, evaluator)
                            st.markdown(response)
                            st.session_state.conversation_history.append({"role": "assistant", "content": response})
                        except Exception as e:
                            st.error(f"Failed to process query: {str(e)}")
        
    else:
        st.info("Upload a video to begin")

if __name__ == "__main__":
    main()