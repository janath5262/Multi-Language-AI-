       🚀 Overview


This project is a multi-agent AI system built for assisting visually impaired individuals.
It analyzes real-world images, identifies objects and hazards, extracts embedded text, generates safe navigation steps, and then translates and speaks the results in multiple languages.

The system was built to align with the hackathon theme:

Building AI Agents using LangChain or LangGraph
✔ Multi-Agent Architecture
✔ Reasoning + Tools + Orchestration
✔ Real-world Assistive Use Case       

🔍 1. Image Understanding (Gemini Vision)

Detects objects, environment, spatial orientation (left/right/top/bottom)

Identifies hazards and contextual cues (traffic, obstacles, stairs)

Provides image-grounded descriptions (no hallucinations)

🧭 2. Agent Architecture (LangChain)

Two LangChain agents perform reasoning:

Safety Agent

Evaluates hazards

Provides risk levels (low / medium / high)

Suggests immediate actions

Navigation Agent

Generates 2–6 simple step-by-step navigation instructions

Provides alternative safe options

📝 3. OCR Extraction

Extracts text from images using Tesseract

Includes image preprocessing for improved accuracy
