\
import json
import os
import re
import requests
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv

from ner_bio import normalizar_texto, generar_bio_tags, filtrar_superpuestas
from chunking_clinico import generar_chunks as generar_chunks_fallback

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_NER_MODEL = os.getenv("MISTRAL_NER_MODEL", os.getenv("MISTRAL_MODEL", "mistral-small-latest"))

ETIQUETAS_PERMITIDAS = {"ESTUDIO", "ANATOMIA", "ALTERACION", "HALLAZGO", "NEGACION", "SINTOMA", "CONDICION", "PROCESO"}


def _extraer_json(texto: str) -> Dict:
    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        inicio = texto.find("{")
        fin = texto.rfind("}")
        if inicio != -1 and fin != -1 and fin > inicio:
            return json.loads(texto[inicio: fin + 1])
        raise


def _limpiar_tipo(tipo: str) -> str:
    tipo = (tipo or "").strip().upper()
    if tipo in ETIQUETAS_PERMITIDAS:
        return tipo
    equivalencias = {
        "DIAGNOSTICO": "CONDICION", "DIAGNÓSTICO": "CONDICION", "ENFERMEDAD": "CONDICION",
        "PATOLOGIA": "PROCESO", "PATOLOGÍA": "PROCESO",
        "LOCALIZACION": "ANATOMIA", "LOCALIZACIÓN": "ANATOMIA", "ORGANO": "ANATOMIA", "ÓRGANO": "ANATOMIA",
    }
    return equivalencias.get(tipo, "HALLAZGO")


def _buscar_span(texto_original: str, entidad_texto: str, ocupados: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    entidad_texto = normalizar_texto(entidad_texto)
    if not entidad_texto:
        return None
    patron = re.escape(entidad_texto)
    for match in re.finditer(patron, texto_original, flags=re.IGNORECASE):
        inicio, fin = match.start(), match.end()
        superpuesto = any(not (fin <= oi or inicio >= of) for oi, of in ocupados)
        if not superpuesto:
            return inicio, fin
    palabras = [re.escape(p) for p in entidad_texto.split()]
    if palabras:
        patron_flexible = r"\s+".join(palabras)
        for match in re.finditer(patron_flexible, texto_original, flags=re.IGNORECASE):
            inicio, fin = match.start(), match.end()
            superpuesto = any(not (fin <= oi or inicio >= of) for oi, of in ocupados)
            if not superpuesto:
                return inicio, fin
    return None


def _normalizar_entidades_llm(texto: str, entidades_raw: List[Dict]) -> List[Dict]:
    entidades = []
    ocupados = []
    entidades_raw = sorted(entidades_raw or [], key=lambda e: -len(str(e.get("texto", ""))))
    for item in entidades_raw:
        texto_entidad = normalizar_texto(str(item.get("texto", "")))
        tipo = _limpiar_tipo(str(item.get("tipo", "")))
        span = _buscar_span(texto, texto_entidad, ocupados)
        if not span:
            continue
        inicio, fin = span
        entidades.append({"texto": texto[inicio:fin], "tipo": tipo, "inicio": inicio, "fin": fin})
        ocupados.append((inicio, fin))
    return filtrar_superpuestas(entidades)


def _normalizar_chunks_llm(chunks_raw: List[Dict], entidades: List[Dict], texto: str) -> List[Dict]:
    chunks = []
    for idx, item in enumerate(chunks_raw or [], start=1):
        chunk_texto = normalizar_texto(str(item.get("texto", "")))
        if not chunk_texto:
            continue
        entidades_chunk = []
        tipos = set()
        for entidad in entidades:
            if entidad["texto"].lower() in chunk_texto.lower():
                entidades_chunk.append(entidad["texto"])
                tipos.add(entidad["tipo"])
        for tipo in item.get("tipos", []) or []:
            tipos.add(_limpiar_tipo(str(tipo)))
        chunks.append({"id": idx, "texto": chunk_texto, "tipos": sorted(tipos), "entidades": entidades_chunk})
    if not chunks:
        return generar_chunks_fallback(texto, entidades)
    return chunks


def construir_prompt_ner_bio_chunks(texto: str) -> str:
    return f"""
Eres un anotador de NER clínico en español para un proyecto de tesis.
Debes etiquetar entidades clínicas presentes de forma textual en el diagnóstico.
No inventes entidades. Usa fragmentos exactos del texto original.

Etiquetas permitidas:
- ESTUDIO: exámenes, reportes, estudios o técnicas de imagen.
- ANATOMIA: órganos, regiones, estructuras o localizaciones anatómicas.
- ALTERACION: lesiones, cambios patológicos, anomalías o hallazgos anormales.
- HALLAZGO: observaciones clínicas o radiológicas.
- NEGACION: expresiones de ausencia como "sin evidencia de".
- SINTOMA: síntomas o manifestaciones clínicas.
- CONDICION: antecedentes, enfermedades o condiciones del paciente.
- PROCESO: procesos patológicos generales, por ejemplo diseminación metastásica.

Diagnóstico:
{texto}

Devuelve exclusivamente JSON válido con esta estructura:
{{
  "entidades": [
    {{"texto": "fragmento exacto del diagnóstico", "tipo": "ETIQUETA"}}
  ],
  "chunks": [
    {{
      "texto": "fragmento clínico coherente del diagnóstico",
      "tipos": ["ETIQUETA"],
      "entidades": ["fragmento exacto del diagnóstico"]
    }}
  ]
}}

Reglas:
1. Prefiere entidades clínicas completas antes que palabras sueltas.
2. Evita duplicar entidades contenidas dentro de una entidad mayor.
3. Si aparece "nódulos pulmonares bilaterales", etiquétalo como ALTERACION.
4. Si aparece "engrosamiento pleural focal", etiquétalo como ALTERACION.
5. Si aparece "adenopatías hiliares", etiquétalo como ALTERACION.
6. Si aparece "carcinoma pulmonar avanzado", etiquétalo como CONDICION.
7. Si aparece "diseminación metastásica pulmonar", etiquétalo como PROCESO.
8. Genera chunks clínicos con sentido completo, no solo por comas.
""".strip()


def llamar_mistral_ner(texto: str) -> Optional[Dict]:
    if not MISTRAL_API_KEY:
        return None
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MISTRAL_NER_MODEL,
        "messages": [{"role": "user", "content": construir_prompt_ner_bio_chunks(texto)}],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    response = requests.post(url, headers=headers, json=payload, timeout=90)
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    return _extraer_json(content)


def procesar_ner_bio_chunks_con_mistral(texto: str):
    texto = normalizar_texto(texto)
    salida = llamar_mistral_ner(texto)
    if not salida:
        raise RuntimeError("MISTRAL_API_KEY no configurada para NER con LLM.")
    entidades = _normalizar_entidades_llm(texto, salida.get("entidades", []))
    bio_tags = generar_bio_tags(texto, entidades)
    chunks = generar_chunks_fallback(texto, entidades)
    return entidades, bio_tags, chunks, MISTRAL_NER_MODEL, "ner_bio_chunks_mistral"
