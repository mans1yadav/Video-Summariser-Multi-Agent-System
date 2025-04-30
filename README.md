# Video-Summariser-Multi-Agent-System

Video Summariser: Multi Agent System is an AI-powered platform for automated generation of brief, context-specific summaries of video content. Utilizing a coordinated pipeline of expert agents, the system analyzes video, audio, and text streams to provide both textual and visual summaries, as well as interactive conversational functionality. The tool is intended for educators, companies, content creators, and anyone who requires effectively extracting and analyzing key information from long-form video.

## Key Features
1. Multi-Agent Modular Pipeline: Specialized agents for audio extraction, transcription, conversational AI, summarization, and visual keyframe selection.

2. Multimodal Summarization: Integrates audio, text, and video frame analysis for richer, more accurate summaries.

3. Hybrid Summarization: Combines extractive (TextRank) and abstractive (Gemini) approaches for flexible output.

4. Attention-Based Keyframe Selection: Uses BiLSTM+Attention and EfficientNet to generate a visual storyboard of highlights.

5. Interactive Q&A: Google Gemini-powered conversational agent answers user queries about the summarized content.

## Technology Stack
* Backend: Flask, Python, PyDub, FastAPI, OpenCV
* Frontend: Streamlit
* AI Models: OpenAI Whisper Large-v3, Google Gemini 1.5, EfficientNet-B7, BiLSTM+Attention
