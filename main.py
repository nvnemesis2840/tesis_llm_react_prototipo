import traceback

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from schemas import ProcessRequest, ProcessResponse
from ner_bio import normalizar_texto, detectar_entidades, generar_bio_tags
from chunking_clinico import generar_chunks
from prompts import construir_prompt
from llm_providers import generar_salidas_multimodelo
from registrar_excel import registrar_analisis_excel, EXCEL_PATH

try:
    from ner_llm_mistral import procesar_ner_bio_chunks_con_mistral
except Exception:
    procesar_ner_bio_chunks_con_mistral = None


app = FastAPI(
    title="API Tesis LLM Clínico",
    description="NER con Mistral, BIO, chunking clínico, registro Excel y generación multimodelo.",
    version="1.4.0",
)

# CORS para React/Vite en desarrollo local.
# allow_credentials=False evita problemas del navegador cuando se usa allow_origins=["*"].
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "https://llms-ug.netlify.app" 
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Evita que errores internos aparezcan en el navegador como un falso error CORS.
    El detalle real se imprime en la terminal del backend.
    """
    print("\nERROR INTERNO EN BACKEND")
    print(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Error interno del backend. Revisa la terminal donde corre uvicorn para ver el detalle.",
            "error": str(exc),
        },
    )


@app.options("/{full_path:path}")
async def options_handler(full_path: str):
    return {"ok": True}


@app.get("/")
def root():
    return {
        "mensaje": "API activa. Use POST /api/process para procesar un diagnóstico clínico.",
        "modelos_soportados": ["gpt", "gemini", "mistral"],
        "ner": "Mistral si hay API Key; reglas como fallback",
        "registro_excel": str(EXCEL_PATH),
        "flujo_excel": "01_BIO/02_NER/04_Chunks/06_Resultados estructurados",
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "cors": "habilitado",
        "frontend_permitido": "http://localhost:5173",
    }


@app.get("/api/registro/excel")
def descargar_registro_excel():
    if not EXCEL_PATH.exists():
        raise HTTPException(status_code=404, detail="Todavía no existe el archivo de registro Excel.")

    return FileResponse(
        path=str(EXCEL_PATH),
        filename="registro_flujo_tesis_01_a_06.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def ejecutar_ner_bio_chunks(texto: str):
    # Flujo principal: Mistral etiqueta entidades clínicas y de ahí se generan BIO + chunks.
    if procesar_ner_bio_chunks_con_mistral is not None:
        try:
            return procesar_ner_bio_chunks_con_mistral(texto)
        except Exception as exc:
            print(f"NER con Mistral no disponible, usando fallback por reglas: {exc}")

    entidades = detectar_entidades(texto)
    bio_tags = generar_bio_tags(texto, entidades)
    chunks = generar_chunks(texto, entidades)

    return entidades, bio_tags, chunks, "reglas_locales", "fallback_reglas"


@app.post("/api/process", response_model=ProcessResponse)
def process(request: ProcessRequest):
    texto = normalizar_texto(request.texto)

    if not texto:
        raise HTTPException(status_code=400, detail="El diagnóstico clínico no puede estar vacío.")

    modelos = request.modelos or []
    if not modelos:
        raise HTTPException(status_code=400, detail="Debe seleccionar al menos un modelo: GPT, Gemini o Mistral.")

    entidades, bio_tags, chunks, ner_modelo, ner_modo = ejecutar_ner_bio_chunks(texto)

    prompt = construir_prompt(texto, entidades, chunks)
    resultados_modelos = generar_salidas_multimodelo(modelos, prompt, texto, entidades, chunks)

    if not resultados_modelos:
        raise HTTPException(status_code=400, detail="No se recibió ningún modelo válido para procesar.")

    registro = registrar_analisis_excel(
        diagnostico_original=texto,
        modelos=modelos,
        entidades=entidades,
        bio_tags=bio_tags,
        chunks=chunks,
        resultados_modelos=resultados_modelos,
        ner_modelo=ner_modelo,
        ner_modo=ner_modo,
    )

    return {
        "registro_id": registro["registro_id"],
        "excel_path": registro["excel_path"],
        "diagnostico_original": texto,
        "entidades": entidades,
        "bio_tags": bio_tags,
        "chunks": chunks,
        "resultados_modelos": resultados_modelos,
        "ner_modelo": ner_modelo,
        "ner_modo": ner_modo,
    }
