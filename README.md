# Medic.ai
Medic.ai is a smart health assistant that lets users chat, upload medical images, and get instant, AI-generated responses using LLMs, image analysis, and voice support.
## Features

- Chat interface to ask health questions in natural language
- Image upload for visual diagnosis assistance
- Medical-tuned LLM for accurate, real-time answers
- Text-to-speech support for accessibility
- Optional voice input (work in progress)
- Export chat and results as PDF
- Critical symptom alerts and suggestions
- Context-aware search using vector embeddings

## Tech Stack

- Frontend: React.js, Tailwind CSS
- Backend: Node.js, Express
- LLM: OpenAI API (replacing local model due to RAM limits)
- Embeddings: HuggingFace
- Vector DB: Pinecone
- Image analysis: OpenCV (planned)
- Voice: Web Speech API (TTS), Whisper (planned for STT)

## How It Works

1. User inputs a query or uploads an image
2. Text is embedded with HuggingFace and matched using Pinecone
3. Query with context is sent to OpenAI's LLM
4. Response is generated and optionally spoken aloud
5. Critical symptoms are flagged with suggested actions
