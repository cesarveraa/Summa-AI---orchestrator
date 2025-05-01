import os
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import AsyncGenerator, List, Union
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Configuración de entorno directa (sin .env necesario en Vercel)
AI_API_KEY = "4fcecffe5a3549359895f2f0b920b009"
AI_API_URL = "https://api.aimlapi.com/v1"

if not AI_API_KEY or not AI_API_URL:
    raise RuntimeError("Faltan variables de entorno para la configuración de IA")

app = FastAPI(
    title="Servicio de Asistente Inteligente",
    version="1.0.0",
    description="API para integración con modelos de lenguaje avanzado"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = AsyncOpenAI(
    api_key=AI_API_KEY,
    base_url=AI_API_URL,
    timeout=30.0,
    max_retries=3,
    default_headers={
        "User-Agent": "FastAPI-Assistant/1.0",
        "X-Custom-Request-ID": os.urandom(16).hex()
    }
)

# ---------------------------- SCHEMAS ----------------------------
class GenerateRequest(BaseModel):
    text: List[str] = []
    ui: List[str] = []
    audio_meta: Union[str, None] = None

class GenerateResponse(BaseModel):
    answer: str

class CompleteRequest(BaseModel):
    partial_transcript: str

class SummarizeRequest(BaseModel):
    full_transcript: str
    highlights: List[str] = []

class SummarizeResponse(BaseModel):
    summary: str
    tasks: List[str]
    debug_payload: str

class VisualContext(BaseModel):
    text: List[str] = []
    ui_elements: List[str] = []
    screenshot: Union[str, None] = None

class MultimodalRequest(BaseModel):
    audio_transcript: str
    visual_context: VisualContext
    user_intent: Union[str, None] = None

# ---------------------------- CORE FUNCTION ----------------------------
async def handle_ai_request(messages: list[dict[str, str]], stream: bool = False) -> AsyncGenerator[str, None] | str:
    try:
        if stream:
            async def generate_stream():
                async with client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    stream=True,
                    temperature=0.7,
                    max_tokens=500
                ) as s:
                    async for chunk in s:
                        if content := chunk.choices[0].delta.content:
                            yield f"data: {content}\n\n"
            return generate_stream()

        resp = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.5
        )
        return resp.choices[0].message.content

    except httpx.ConnectError as e:
        raise HTTPException(503, f"Error de conexión con IA: {e}")
    except Exception as e:
        raise HTTPException(500, f"Error interno del servidor: {e}")

# ---------------------------- ENDPOINTS ----------------------------
@app.post("/generate_answer", response_model=GenerateResponse)
async def generate_answer(request: GenerateRequest):
    system_prompt = (
        "Eres un asistente multimodal experto en análisis de contexto visual y auditivo. "
        "Integra información de estas fuentes:\n"
        "1. Texto de pantalla\n"
        "2. Elementos UI\n"
        "3. Contexto de audio\n"
        "Responde de forma clara y concisa."
    )
    user_prompt = (
        f"Texto en pantalla: {', '.join(request.text) or 'Ninguno'}\n"
        f"Elementos UI: {', '.join(request.ui) or 'Ninguno'}\n"
        f"Metadato de audio: {request.audio_meta or 'Ninguno'}\n\n"
        "Genera una respuesta integrada considerando estos tres contextos."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    answer = await handle_ai_request(messages)
    return GenerateResponse(answer=answer)

@app.post("/analyze_screen")
async def analyze_screen(visual: VisualContext):
    messages = [
        {"role": "system", "content": "Eres un experto en análisis de interfaces de usuario."},
        {"role": "user", "content": f"Texto: {', '.join(visual.text)}\nUI: {', '.join(visual.ui_elements)}"}
    ]
    analysis = await handle_ai_request(messages)
    return {"analysis": analysis.split("\n")}

@app.post("/complete_speech")
async def complete_speech(request: CompleteRequest):
    messages = [
        {"role": "system", "content": "Completa el texto proporcionado manteniendo coherencia y estilo."},
        {"role": "user", "content": request.partial_transcript}
    ]
    return StreamingResponse(
        handle_ai_request(messages, stream=True),
        media_type="text/event-stream"
    )

@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest):
    messages = [
        {"role": "system", "content": "Eres un asistente que resume documentos técnicos para reportes de reuniones."},
        {"role": "user", "content": f"Documento:\n{request.full_transcript}\n\nPuntos clave: {', '.join(request.highlights) or 'Ninguno'}"}
    ]
    result = await handle_ai_request(messages)
    if not isinstance(result, str):
        raise HTTPException(500, "Respuesta inesperada del modelo")

    parts = result.strip().split("\n\n", maxsplit=1)
    summary = parts[0].strip()
    tasks = [line.strip() for line in parts[1].split("\n") if line.strip()] if len(parts) > 1 else []
    return SummarizeResponse(summary=summary, tasks=tasks, debug_payload=messages[-1]["content"])

@app.on_event("startup")
async def startup_event():
    try:
        await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"user","content":"ping"}],
            max_tokens=1
        )
        print("✅ Conexión a IA OK")
    except Exception as e:
        print(f"⚠️ No se pudo verificar IA al inicio: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    await client.close()
