
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any

import requests
from dotenv import load_dotenv

from flujo06 import (
    METODO_GENERACION,
    VERSION_SALIDA,
    normalizar_salida_modelo_06,
    salida_06_demo,
    texto_plano,
    to_json,
)

load_dotenv()

MODEL_CONFIG = {
    "gpt": {
        "nombre": "GPT",
        "proveedor": "OpenAI",
        "env_key": "OPENAI_API_KEY",
        "env_model": "OPENAI_MODEL",
        "default_model": "gpt-4o-mini",
    },
    "gemini": {
        "nombre": "Gemini",
        "proveedor": "Google",
        "env_key": "GEMINI_API_KEY",
        "env_model": "GEMINI_MODEL",
        "default_model": "gemini-1.5-flash",
    },
    "mistral": {
        "nombre": "Mistral",
        "proveedor": "Mistral AI",
        "env_key": "MISTRAL_API_KEY",
        "env_model": "MISTRAL_MODEL",
        "default_model": "mistral-small-latest",
    },
}


def _extraer_json(texto: str) -> Dict[str, Any]:
    try:
        data = json.loads(texto)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        inicio = texto.find("{")
        fin = texto.rfind("}")
        if inicio != -1 and fin != -1 and fin > inicio:
            data = json.loads(texto[inicio: fin + 1])
            return data if isinstance(data, dict) else {}
        raise


def llamar_mistral(prompt: str, model_name: str, api_key: str) -> Tuple[Dict[str, Any], str, Dict[str, Any], Dict[str, Any]]:
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(os.getenv("MISTRAL_TEMPERATURE", "0.2")),
        "response_format": {"type": "json_object"},
    }
    response = requests.post(url, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    raw = response.json()
    content = raw["choices"][0]["message"]["content"]
    return _extraer_json(content), content, raw.get("usage", {}), raw


def llamar_openai(prompt: str, model_name: str, api_key: str) -> Tuple[Dict[str, Any], str, Dict[str, Any], Dict[str, Any]]:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(os.getenv("OPENAI_TEMPERATURE", "0.2")),
        "response_format": {"type": "json_object"},
    }
    response = requests.post(url, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    raw = response.json()
    content = raw["choices"][0]["message"]["content"]
    return _extraer_json(content), content, raw.get("usage", {}), raw


def llamar_gemini(prompt: str, model_name: str, api_key: str) -> Tuple[Dict[str, Any], str, Dict[str, Any], Dict[str, Any]]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": float(os.getenv("GEMINI_TEMPERATURE", "0.2")),
            "responseMimeType": "application/json",
        },
    }
    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()
    raw = response.json()
    content = raw["candidates"][0]["content"]["parts"][0]["text"]
    return _extraer_json(content), content, raw.get("usageMetadata", {}), raw


def generar_salida_por_modelo(
    modelo_id: str,
    prompt: str,
    diagnostico_original: str,
    entidades: List[Dict],
    chunks: List[Dict],
) -> Tuple[Dict[str, Any], str, str, str, Dict[str, Any], Dict[str, Any], str]:
    config = MODEL_CONFIG[modelo_id]
    api_key = os.getenv(config["env_key"], "")
    model_name = os.getenv(config["env_model"], config["default_model"])

    if not api_key:
        salida = salida_06_demo(config["nombre"], diagnostico_original, entidades, chunks)
        return salida, model_name, "demo_sin_api_key", to_json(salida), {}, {}, ""

    try:
        if modelo_id == "mistral":
            salida, raw_text, usage, raw = llamar_mistral(prompt, model_name, api_key)
            return salida, model_name, "api_mistral", raw_text, usage, raw, ""
        if modelo_id == "gpt":
            salida, raw_text, usage, raw = llamar_openai(prompt, model_name, api_key)
            return salida, model_name, "api_openai", raw_text, usage, raw, ""
        if modelo_id == "gemini":
            salida, raw_text, usage, raw = llamar_gemini(prompt, model_name, api_key)
            return salida, model_name, "api_gemini", raw_text, usage, raw, ""
    except Exception as exc:
        print(f"Error al llamar {modelo_id}: {exc}")
        salida = salida_06_demo(config["nombre"], diagnostico_original, entidades, chunks)
        return salida, model_name, "demo_por_error_api", to_json(salida), {}, {}, str(exc)

    salida = salida_06_demo(config["nombre"], diagnostico_original, entidades, chunks)
    return salida, model_name, "demo_modelo_no_soportado", to_json(salida), {}, {}, "modelo_no_soportado"


def _tokens(usage: Dict[str, Any], *keys: str):
    for key in keys:
        if key in usage:
            return usage.get(key)
    return None


def generar_salidas_multimodelo(
    modelos: List[str],
    prompt: str,
    diagnostico_original: str,
    entidades: List[Dict],
    chunks: List[Dict],
) -> List[Dict]:
    resultados = []

    for modelo_id in modelos:
        if modelo_id not in MODEL_CONFIG:
            continue

        timestamp_inicio = datetime.now().isoformat(timespec="seconds")
        salida_raw, modelo_usado, modo, respuesta_raw, usage, raw_response, error = generar_salida_por_modelo(
            modelo_id=modelo_id,
            prompt=prompt,
            diagnostico_original=diagnostico_original,
            entidades=entidades,
            chunks=chunks,
        )
        timestamp_fin = datetime.now().isoformat(timespec="seconds")
        salida = normalizar_salida_modelo_06(salida_raw)

        estado_llm = "error" if error and modo.startswith("api_") else "ok"
        if modo.startswith("demo_por_error"):
            estado_llm = "demo_por_error_api"
        elif modo.startswith("demo"):
            estado_llm = "demo"

        input_tokens = _tokens(usage, "input_tokens", "prompt_tokens", "promptTokenCount", "prompt_eval_count")
        output_tokens = _tokens(usage, "output_tokens", "completion_tokens", "candidatesTokenCount", "eval_count")
        total_tokens = _tokens(usage, "total_tokens", "totalTokenCount")

        explicacion_texto = salida.get("explicacion", "") or texto_plano(salida.get("explicaciones_chunks", []))

        item = {
            "modelo_id": modelo_id,
            "nombre": MODEL_CONFIG[modelo_id]["nombre"],
            "proveedor": MODEL_CONFIG[modelo_id]["proveedor"],
            "diagnostico_simplificado": texto_plano(salida.get("diagnostico_simplificado", "")),
            "explicacion": explicacion_texto,
            "resumen_automatico": texto_plano(salida.get("resumen_automatico", "")),
            "modelo_usado": modelo_usado,
            "modo": modo,

            # Campos estructurados equivalentes al flujo 06.
            "resultado_llm_json": to_json(salida_raw),
            "resultado_modelo_json": to_json(salida_raw),
            "respuesta_llm_raw": respuesta_raw,
            "respuesta_modelo_raw": respuesta_raw,
            "analisis_global_json": to_json(salida.get("analisis_global", {})),
            "analisis_feature_engineering_modelo_json": to_json(salida.get("analisis_feature_engineering_modelo", {})),
            "chunks_analizados_json": to_json(salida.get("chunks_analizados", [])),
            "simplificacion_chunks_usados_json": to_json(salida.get("simplificacion_chunks_usados", [])),
            "simplificacion_chunks_descartados_json": to_json(salida.get("simplificacion_chunks_descartados", [])),
            "explicaciones_chunks_json": to_json(salida.get("explicaciones_chunks", [])),
            "explicacion_chunks_usados_json": to_json(salida.get("explicacion_chunks_usados", [])),
            "explicacion_chunks_descartados_json": to_json(salida.get("explicacion_chunks_descartados", [])),
            "resumen_chunks_usados_json": to_json(salida.get("resumen_chunks_usados", [])),
            "resumen_chunks_descartados_json": to_json(salida.get("resumen_chunks_descartados", [])),
            "evaluacion_interna_json": to_json(salida.get("evaluacion_interna", {})),
            "alertas_semanticas_json": to_json(salida.get("alertas_semanticas", [])),
            "advertencias_clinicas_json": to_json(salida.get("advertencias_clinicas", [])),
            "llm_provider": MODEL_CONFIG[modelo_id]["proveedor"],
            "llm_model": modelo_usado,
            "model_alias": modelo_id,
            "estado_llm": estado_llm,
            "error": error,
            "metodo_generacion": METODO_GENERACION,
            "version_salida": VERSION_SALIDA,
            "temperature": 0.2,
            "max_tokens": None,
            "timestamp_inicio": timestamp_inicio,
            "timestamp_fin": timestamp_fin,
            "elapsed_seconds": None,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "usage_json": to_json(usage),
            "raw_response_json": to_json(raw_response),
        }
        resultados.append(item)

    return resultados
