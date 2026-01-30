"""
Job handlers for GPU worker.

Handlers are organized by job_type/model_type/mode:
- handlers/tts/stabletts/training.py
- handlers/tts/stabletts/inference.py
- handlers/asr/w2v2bert/training.py
- handlers/asr/w2v2bert/inference.py

Each handler module must have a run(job_context, callbacks) function.
"""
