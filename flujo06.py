
from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Dict, List


METODO_GENERACION = "generacion_multimodelo_ner_explicable_web"
VERSION_SALIDA = "v6_web_flujo_01_a_06_incremental"
METODO_CHUNKING = "chunking_clinico_desde_ner_bio_web"


def limpiar_texto(texto: Any) -> str:
    if texto is None:
        return ""
    return re.sub(r"\s+", " ", str(texto)).strip()


def to_json(valor: Any) -> str:
    return json.dumps(valor, ensure_ascii=False)


def valor_excel(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value)


def texto_plano(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        partes = []
        for k, v in value.items():
            if isinstance(v, (dict, list)):
                v = json.dumps(v, ensure_ascii=False)
            partes.append(f"{k}: {v}")
        return "\n".join(partes)
    if isinstance(value, list):
        partes = []
        for item in value:
            partes.append(texto_plano(item))
        return "\n".join([p for p in partes if p])
    return str(value)


def normalizar_entidades_06(entidades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    salida = []
    for idx, ent in enumerate(entidades or [], start=1):
        texto = limpiar_texto(ent.get("texto", ""))
        label = limpiar_texto(ent.get("label", ent.get("tipo", ""))).upper() or "HALLAZGO"
        inicio = int(ent.get("inicio", ent.get("start_char", 0)) or 0)
        fin = int(ent.get("fin", ent.get("end_char", inicio + len(texto))) or inicio + len(texto))
        salida.append({
            "texto": texto,
            "label": label,
            "start_char": inicio,
            "end_char": fin,
            "negado": bool(ent.get("negado", False)),
            "metodo_ner": ent.get("metodo_ner", "ner_clinico_mistral_web"),
            "entidad_id": idx,
        })
    return salida


def _score_chunk(texto: str, labels: List[str]) -> Dict[str, Any]:
    palabras = re.findall(r"\w+", texto, flags=re.UNICODE)
    total_palabras = len(palabras)
    largas = [p for p in palabras if len(p) >= 10]
    nominalizaciones = [p for p in palabras if p.lower().endswith(("ción", "sión", "miento", "mientos", "dad", "tud"))]
    base = 2.5 + min(total_palabras, 18) * 0.22 + len(largas) * 0.35 + len(nominalizaciones) * 0.35
    if any(label in {"ALTERACION", "PROCESO", "CONDICION", "DIAGNOSTICO_CLINICO", "HALLAZGO_CLINICO"} for label in labels):
        base += 0.6
    score = round(max(1.0, min(10.0, base)), 2)
    if score >= 7:
        nivel = "alta"
    elif score >= 4:
        nivel = "media"
    else:
        nivel = "baja"
    return {
        "score_complejidad_np_normalizado_10": score,
        "nivel_complejidad_np": nivel,
        "numero_np": max(1, min(3, total_palabras // 3 or 1)),
        "cantidad_nominalizaciones": len(nominalizaciones),
        "nominalizaciones": nominalizaciones,
    }


def normalizar_chunks_06(chunks: List[Dict[str, Any]], entidades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    entidades_06 = normalizar_entidades_06(entidades)
    salida = []
    for idx, ch in enumerate(chunks or [], start=1):
        texto = limpiar_texto(ch.get("chunk", ch.get("texto", "")))
        labels = ch.get("labels_chunk", ch.get("tipos", [])) or []
        labels = [limpiar_texto(x).upper() for x in labels if limpiar_texto(x)] or ["SIN_ETIQUETA"]
        inicio = int(ch.get("inicio", ch.get("start_char", 0)) or 0)
        fin = int(ch.get("fin", ch.get("end_char", inicio + len(texto))) or inicio + len(texto))
        entidades_chunk = []
        texto_lower = texto.lower()
        for ent in entidades_06:
            if ent["texto"].lower() in texto_lower or (ent["start_char"] >= inicio and ent["end_char"] <= fin):
                entidades_chunk.append({
                    "texto": ent["texto"],
                    "label": ent["label"],
                    "start_char": ent["start_char"],
                    "end_char": ent["end_char"],
                    "negado": ent.get("negado", False),
                })
        if not entidades_chunk and ch.get("entidades"):
            for et in ch.get("entidades", []):
                entidades_chunk.append({"texto": limpiar_texto(et), "label": labels[0], "start_char": inicio, "end_char": fin, "negado": False})

        fe = _score_chunk(texto, labels)
        score = fe["score_complejidad_np_normalizado_10"]
        nivel = fe["nivel_complejidad_np"]
        negado = any(l == "NEGACION" for l in labels) or texto.lower().startswith(("sin ", "no "))

        salida.append({
            "chunk_id": int(ch.get("chunk_id", ch.get("id", idx)) or idx),
            "chunk": texto,
            "tipo": "chunk_por_entidad_clinica_web",
            "metodo": METODO_CHUNKING,
            "start_char": inicio,
            "end_char": fin,
            "longitud_caracteres": len(texto),
            "longitud_palabras": len(re.findall(r"\w+", texto, flags=re.UNICODE)),
            "total_entidades": len(entidades_chunk),
            "labels_chunk": labels,
            "labels_chunk_texto": ", ".join(labels),
            "negado": negado,
            "entidades_chunk_json": entidades_chunk,
            "categoria": labels[0],
            "categoria_principal": labels[0],
            "feature_engineering_np": {
                "fuente_metodologica": "Rúbrica computacional simplificada para ejecución web: longitud, términos largos, nominalizaciones y tipo clínico detectado.",
                "score_complejidad_np_normalizado_10": score,
                "nivel_complejidad_np": nivel,
                "nominalizaciones": fe["nominalizaciones"],
                "cantidad_nominalizaciones": fe["cantidad_nominalizaciones"],
                "nota_ner": "Las entidades NER guían el chunking y la explicación, no modifican el diagnóstico original.",
            },
            "nivel_complejidad": nivel,
            "score_complejidad": score,
            "es_termino_medico": labels[0] != "SIN_ETIQUETA",
            "requiere_explicacion": bool(score >= 4 or any(l in {"ALTERACION", "PROCESO", "CONDICION"} for l in labels)),
        })
    return salida


def calcular_feature_global_desde_chunks(chunks_06: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not chunks_06:
        return {"total_chunks": 0, "promedio_score_np": None, "niveles": {}, "requiere_explicacion": 0}
    scores = []
    niveles = {}
    requiere = 0
    for ch in chunks_06:
        score = ch.get("score_complejidad")
        if isinstance(score, (int, float)):
            scores.append(float(score))
        nivel = ch.get("nivel_complejidad") or "sin_nivel"
        niveles[nivel] = niveles.get(nivel, 0) + 1
        if ch.get("requiere_explicacion"):
            requiere += 1
    return {
        "total_chunks": len(chunks_06),
        "promedio_score_np": round(sum(scores) / len(scores), 2) if scores else None,
        "niveles": niveles,
        "requiere_explicacion": requiere,
    }


def construir_entrada_06(diagnostico_original: str, entidades: List[Dict[str, Any]], chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    entidades_06 = normalizar_entidades_06(entidades)
    chunks_06 = normalizar_chunks_06(chunks, entidades)
    return {
        "diagnostico_original": limpiar_texto(diagnostico_original),
        "entidades": [{"texto": e["texto"], "label": e["label"], "negado": e.get("negado", False)} for e in entidades_06],
        "chunks": [
            {
                "id": ch["chunk_id"],
                "chunk": ch["chunk"],
                "labels": ch["labels_chunk_texto"],
                "negado": ch["negado"],
                "score_np": ch["score_complejidad"],
                "nivel_np": ch["nivel_complejidad"],
                "requiere_exp": ch["requiere_explicacion"],
            }
            for ch in chunks_06
        ],
        "feature_np_global": calcular_feature_global_desde_chunks(chunks_06),
    }


def construir_prompt_06(diagnostico_original: str, entidades: List[Dict[str, Any]], chunks: List[Dict[str, Any]]) -> str:
    entrada = construir_entrada_06(diagnostico_original, entidades, chunks)
    formato = {
        "analisis_global": {
            "nivel_complejidad_global": "baja|media|alta",
            "justificacion": "breve"
        },
        "analisis_feature_engineering_modelo": {
            "uso_de_rubrica_np": "breve",
            "chunks_mas_complejos": [1]
        },
        "chunks_analizados": [
            {"chunk_id": 1, "decision": "usar|descartar|usar_parcialmente", "motivo": "breve"}
        ],
        "diagnostico_simplificado": "texto claro del diagnóstico",
        "simplificacion_chunks_usados": [
            {"chunk_id": 1, "uso": "simplificado|mantenido|fusionado|preservado_por_negacion"}
        ],
        "simplificacion_chunks_descartados": [
            {"chunk_id": 1, "motivo": "breve"}
        ],
        "explicaciones_chunks": [
            {"chunk_id": 1, "explicacion_simple": "máximo una oración", "terminos_clave": ["término"]}
        ],
        "explicacion_chunks_usados": [
            {"chunk_id": 1, "justificacion": "breve"}
        ],
        "explicacion_chunks_descartados": [
            {"chunk_id": 1, "motivo": "breve"}
        ],
        "resumen_automatico": "resumen breve",
        "resumen_chunks_usados": [
            {"chunk_id": 1, "motivo_inclusion": "breve"}
        ],
        "resumen_chunks_descartados": [
            {"chunk_id": 1, "motivo": "breve"}
        ],
        "evaluacion_interna": {
            "conserva_sentido_medico": True,
            "respeta_negaciones": True,
            "agrega_informacion_no_presente": False,
            "omite_hallazgos_relevantes": False,
            "observacion_general": "breve"
        },
        "alertas_semanticas": [],
        "advertencias_clinicas": []
    }
    return (
        "Eres un asistente experto en procesamiento de lenguaje clínico en español.\n"
        "Transforma diagnósticos clínicos en textos comprensibles sin alterar el sentido médico.\n\n"
        "Reglas:\n"
        "- No inventes información clínica.\n"
        "- Respeta negaciones como sin, no se evidencia o ausencia de.\n"
        "- Conserva hallazgos, anatomía, alteraciones, estudios y condiciones importantes.\n"
        "- No des recomendaciones médicas personalizadas.\n"
        "- Responde SOLO con JSON válido.\n"
        "- Sé compacto: explicaciones de máximo 1 oración por chunk.\n"
        "- No calcules F1, precisión, recall, TP, FP ni FN.\n\n"
        "Realiza tres tareas: simplificación, explicación por chunks y resumen.\n"
        "Usa los chunks por id. No repitas el texto completo de cada chunk en las listas de usados/descartados.\n"
        "diagnostico_simplificado y resumen_automatico no deben quedar vacíos.\n"
        "explicaciones_chunks debe incluir al menos un elemento por cada chunk relevante.\n\n"
        "JSON de entrada:\n"
        f"{json.dumps(entrada, ensure_ascii=False, indent=2)}\n\n"
        "Devuelve SOLO un JSON válido con esta estructura compacta:\n"
        f"{json.dumps(formato, ensure_ascii=False, indent=2)}"
    )


def _lista(data: Dict[str, Any], *keys: str) -> List[Any]:
    for key in keys:
        value = data.get(key)
        if isinstance(value, list):
            return value
    return []


def _dict(data: Dict[str, Any], *keys: str) -> Dict[str, Any]:
    for key in keys:
        value = data.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _texto(data: Dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = data.get(key)
        if value:
            return texto_plano(value)
    return ""


def normalizar_salida_modelo_06(data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        data = {}
    for wrapper in ["resultado", "salida", "response", "respuesta"]:
        if isinstance(data.get(wrapper), dict):
            data = data[wrapper]
            break

    explicaciones = _lista(data, "explicaciones_chunks", "explicacion_chunks", "explicaciones")
    explicacion_texto = _texto(data, "explicacion")
    if not explicacion_texto and explicaciones:
        explicacion_texto = "\n".join(
            [f"Chunk {e.get('chunk_id', '')}: {e.get('explicacion_simple', e.get('explicacion', ''))}" for e in explicaciones if isinstance(e, dict)]
        )

    salida = {
        "analisis_global": _dict(data, "analisis_global"),
        "analisis_feature_engineering_modelo": _dict(data, "analisis_feature_engineering_modelo", "analisis_feature_engineering"),
        "chunks_analizados": _lista(data, "chunks_analizados", "analisis_chunks", "chunks_procesados"),
        "diagnostico_simplificado": _texto(data, "diagnostico_simplificado", "simplificacion", "texto_simplificado"),
        "simplificacion_chunks_usados": _lista(data, "simplificacion_chunks_usados", "chunks_usados_simplificacion"),
        "simplificacion_chunks_descartados": _lista(data, "simplificacion_chunks_descartados", "chunks_descartados_simplificacion"),
        "explicaciones_chunks": explicaciones,
        "explicacion": explicacion_texto,
        "explicacion_chunks_usados": _lista(data, "explicacion_chunks_usados", "chunks_usados_explicacion"),
        "explicacion_chunks_descartados": _lista(data, "explicacion_chunks_descartados", "chunks_descartados_explicacion"),
        "resumen_automatico": _texto(data, "resumen_automatico", "resumen", "summary"),
        "resumen_chunks_usados": _lista(data, "resumen_chunks_usados", "chunks_usados_resumen"),
        "resumen_chunks_descartados": _lista(data, "resumen_chunks_descartados", "chunks_descartados_resumen"),
        "evaluacion_interna": _dict(data, "evaluacion_interna", "control_semantico") or {
            "conserva_sentido_medico": None,
            "respeta_negaciones": None,
            "agrega_informacion_no_presente": None,
            "omite_hallazgos_relevantes": None,
            "observacion_general": "",
        },
        "alertas_semanticas": _lista(data, "alertas_semanticas", "alertas"),
        "advertencias_clinicas": _lista(data, "advertencias_clinicas", "advertencias"),
    }
    return salida


def salida_06_demo(modelo_nombre: str, diagnostico_original: str, entidades: List[Dict[str, Any]], chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    chunks_06 = normalizar_chunks_06(chunks, entidades)
    usados = [{"chunk_id": ch["chunk_id"], "uso": "simplificado"} for ch in chunks_06]
    explicaciones = [
        {
            "chunk_id": ch["chunk_id"],
            "explicacion_simple": f"{ch['chunk']} es un elemento clínico identificado como {ch['labels_chunk_texto']}.",
            "terminos_clave": ch.get("entidades_chunk_json", []),
        }
        for ch in chunks_06
    ]
    return {
        "analisis_global": {
            "nivel_complejidad_global": calcular_feature_global_desde_chunks(chunks_06).get("niveles", {}),
            "justificacion": "Salida demo generada porque no se configuró API Key o hubo error de proveedor.",
        },
        "analisis_feature_engineering_modelo": {
            "uso_de_rubrica_np": "Se usó la estructura de chunks y complejidad calculada en el flujo web.",
            "chunks_mas_complejos": [ch["chunk_id"] for ch in chunks_06 if ch.get("requiere_explicacion")][:3],
        },
        "chunks_analizados": [{"chunk_id": ch["chunk_id"], "decision": "usar", "motivo": "Chunk clínico detectado por NER."} for ch in chunks_06],
        "diagnostico_simplificado": f"[{modelo_nombre}] El diagnóstico fue transformado a lenguaje más claro conservando los hallazgos detectados por NER.",
        "simplificacion_chunks_usados": usados,
        "simplificacion_chunks_descartados": [],
        "explicaciones_chunks": explicaciones,
        "explicacion_chunks_usados": [{"chunk_id": ch["chunk_id"], "justificacion": "Chunk relevante para explicación."} for ch in chunks_06],
        "explicacion_chunks_descartados": [],
        "resumen_automatico": f"[{modelo_nombre}] Resumen automático basado en {len(chunks_06)} chunks clínicos detectados.",
        "resumen_chunks_usados": [{"chunk_id": ch["chunk_id"], "motivo_inclusion": "Hallazgo clínico relevante."} for ch in chunks_06],
        "resumen_chunks_descartados": [],
        "evaluacion_interna": {
            "conserva_sentido_medico": True,
            "respeta_negaciones": True,
            "agrega_informacion_no_presente": False,
            "omite_hallazgos_relevantes": False,
            "observacion_general": "Salida demo, no usar como resultado final de tesis.",
        },
        "alertas_semanticas": [],
        "advertencias_clinicas": [],
    }
