
import re
from typing import Any, Dict, List, Tuple


LABEL_MAP = {
    "ESTUDIO": "ESTUDIO_CLINICO",
    "ESTUDIO_CLINICO": "ESTUDIO_CLINICO",
    "HALLAZGO": "HALLAZGO_CLINICO",
    "HALLAZGO_CLINICO": "HALLAZGO_CLINICO",
    "ALTERACION": "ALTERACION",
    "ANATOMIA": "ANATOMIA",
    "NEGACION": "NEGACION",
    "SINTOMA": "SINTOMA",
    "CONDICION": "CONDICION",
    "PROCESO": "PROCESO",
}


PRIORIDAD_CATEGORIA = [
    "ALTERACION",
    "HALLAZGO_CLINICO",
    "HALLAZGO_CLINICO_NEGADO",
    "PROCESO",
    "CONDICION",
    "SINTOMA",
    "ANATOMIA",
    "ESTUDIO_CLINICO",
    "NEGACION",
]


NOMINALIZACIONES_CLINICAS = {
    "rectificación", "disminución", "densidad", "formación", "presencia",
    "alteración", "consolidación", "inflamación", "obstrucción", "destrucción",
    "acentuación", "diseminación", "exposición", "hiperinsuflación", "lordosis",
}


def _normalizar_label(label: str) -> str:
    return LABEL_MAP.get(str(label or "").strip().upper(), str(label or "").strip().upper() or "HALLAZGO_CLINICO")


def _normalizar_entidades(entidades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalizadas = []
    for e in entidades or []:
        texto = str(e.get("texto") or e.get("text") or "").strip()
        if not texto:
            continue
        inicio = int(e.get("inicio", e.get("start_char", 0)) or 0)
        fin = int(e.get("fin", e.get("end_char", inicio + len(texto))) or inicio + len(texto))
        label = _normalizar_label(e.get("tipo", e.get("label", "HALLAZGO_CLINICO")))

        normalizadas.append({
            "texto": texto,
            "label": label,
            "start_char": inicio,
            "end_char": fin,
            "negado": bool(e.get("negado", False)),
        })

    # Evita duplicados exactos.
    vistos = set()
    unicas = []
    for e in sorted(normalizadas, key=lambda x: (x["start_char"], x["end_char"])):
        key = (e["texto"].lower(), e["label"], e["start_char"], e["end_char"])
        if key not in vistos:
            vistos.add(key)
            unicas.append(e)
    return unicas


def _token_count(texto: str) -> int:
    return len(re.findall(r"\b\w+\b", texto, flags=re.UNICODE))


def _detectar_negacion(texto: str, start: int, end: int, entidades: List[Dict[str, Any]]) -> bool:
    fragmento = texto[start:end].lower()
    if re.search(r"\b(sin|no se observa|no se observan|no evidencia|sin evidencia|ausencia de|no muestran)\b", fragmento):
        return True

    for e in entidades:
        if e["start_char"] >= start and e["end_char"] <= end and e.get("negado"):
            return True

    return False


def _expandir_a_clausula(texto: str, start: int, end: int) -> Tuple[int, int]:
    """
    Expande una entidad a su cláusula clínica cercana usando puntuación.
    No se queda en la entidad sola, porque eso vuelve el chunk igual al NER.
    """
    n = len(texto)

    # Inicio: después de coma, punto o punto y coma anterior.
    s = start
    while s > 0 and texto[s - 1] not in ".;,":
        s -= 1
    while s < n and texto[s].isspace():
        s += 1

    # Fin: antes de coma, punto o punto y coma siguiente.
    e = end
    while e < n and texto[e] not in ".;,":
        e += 1

    # Regla: si el fragmento inicial es un encabezado de estudio con "muestra",
    # se conserva junto con el primer hallazgo posterior.
    frag = texto[s:e].lower()
    if ("examen radiológico" in frag or "estudio radiológico" in frag or "radiografía" in frag) and "muestra" in frag:
        # seguir hasta la próxima coma después del primer hallazgo
        if e < n and texto[e] == ",":
            siguiente_fin = e + 1
            while siguiente_fin < n and texto[siguiente_fin] not in ".;,":
                siguiente_fin += 1
            if siguiente_fin > e + 1:
                e = siguiente_fin

    # Regla: si empieza con "sin", "no", o contiene "sin" antes de una entidad,
    # mantener la cláusula completa de negación.
    return s, e


def _fusionar_intervalos(intervalos: List[Tuple[int, int]], texto: str) -> List[Tuple[int, int]]:
    if not intervalos:
        return []

    intervalos = sorted(intervalos)
    fusionados = [intervalos[0]]

    for start, end in intervalos[1:]:
        last_start, last_end = fusionados[-1]

        separador = texto[last_end:start]
        distancia = start - last_end

        # Fusiona si están muy cerca dentro de la misma relación clínica:
        # ejemplo: "cuerpos vertebrales con altura conservada".
        debe_fusionar = False

        if distancia <= 6 and re.search(r"\b(con|y|ni|sin|de|del|la|el)\b|^[\s,]+$", separador, flags=re.IGNORECASE):
            debe_fusionar = True

        # No fusionar si el corte viene por coma y la cláusula anterior ya tiene sentido,
        # salvo que haya una negación que dependa de la entidad anterior.
        if "," in separador and not re.search(r"\bsin\b", separador, flags=re.IGNORECASE):
            debe_fusionar = False

        if debe_fusionar:
            fusionados[-1] = (last_start, max(last_end, end))
        else:
            fusionados.append((start, end))

    return fusionados


def _intervalos_por_clausula(texto: str, entidades: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    intervalos = []
    entidades = sorted(entidades, key=lambda e: (e["start_char"], e["end_char"]))

    for e in entidades:
        start, end = _expandir_a_clausula(texto, e["start_char"], e["end_char"])
        intervalos.append((start, end))

    intervalos = _fusionar_intervalos(intervalos, texto)

    # Limpieza: quitar intervalos contenidos dentro de otros.
    limpios = []
    for s, e in intervalos:
        contenido = any(s >= os and e <= oe and (s, e) != (os, oe) for os, oe in intervalos)
        if not contenido:
            limpios.append((s, e))

    # Evitar duplicados.
    out = []
    vistos = set()
    for s, e in limpios:
        key = (s, e)
        if key not in vistos:
            vistos.add(key)
            out.append((s, e))
    return sorted(out)


def _entidades_en_intervalo(entidades: List[Dict[str, Any]], start: int, end: int) -> List[Dict[str, Any]]:
    return [
        e for e in entidades
        if e["start_char"] >= start and e["end_char"] <= end
    ]


def _categoria_principal(labels: List[str], negado: bool) -> str:
    labels_set = set(labels)
    if negado and ("HALLAZGO_CLINICO" in labels_set or "ALTERACION" in labels_set):
        return "HALLAZGO_CLINICO_NEGADO"
    for label in PRIORIDAD_CATEGORIA:
        if label in labels_set:
            return label
    return labels[0] if labels else "SIN_ETIQUETA"


def _feature_engineering_web(chunk_texto: str, labels: List[str], total_chunks: int, total_palabras_doc: int) -> Dict[str, Any]:
    palabras = re.findall(r"\b\w+\b", chunk_texto.lower(), flags=re.UNICODE)
    total_palabras = max(len(palabras), 1)
    nominalizaciones = sorted({p for p in palabras if p in NOMINALIZACIONES_CLINICAS or p.endswith(("ción", "sión", "dad", "miento"))})

    terminos_largos = [p for p in palabras if len(p) >= 10]
    tiene_clinico_alto = any(l in labels for l in ["ALTERACION", "PROCESO", "CONDICION", "HALLAZGO_CLINICO"])
    tiene_negacion = "NEGACION" in labels

    score = 2.0
    score += min(total_palabras / 3.0, 2.5)
    score += min(len(terminos_largos) * 0.55, 2.0)
    score += min(len(nominalizaciones) * 0.65, 2.0)
    if tiene_clinico_alto:
        score += 0.8
    if tiene_negacion:
        score += 0.4

    score = round(min(score, 10.0), 2)

    if score < 4:
        nivel = "baja"
    elif score < 7:
        nivel = "media"
    else:
        nivel = "alta"

    return {
        "fuente_metodologica": "Rúbrica computacional simplificada para ejecución web, alineada al flujo 04: longitud, términos largos, nominalizaciones y tipo clínico detectado.",
        "total_chunks_documento": total_chunks,
        "total_palabras_documento": total_palabras_doc,
        "longitud_palabras_chunk": total_palabras,
        "terminos_largos": terminos_largos,
        "cantidad_terminos_largos": len(terminos_largos),
        "score_complejidad_np_normalizado_10": score,
        "nivel_complejidad_np": nivel,
        "nominalizaciones": nominalizaciones,
        "cantidad_nominalizaciones": len(nominalizaciones),
        "nota_ner": "Las entidades NER guían el chunking y la explicación, no modifican el diagnóstico original.",
    }


def generar_chunks(texto: str, entidades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Chunking web alineado con el flujo 04 -> 06.

    Antes:
        1 entidad NER = 1 chunk.
    Problema:
        El chunk era casi idéntico al NER.

    Ahora:
        cláusula clínica + entidades NER relacionadas = chunk.

    Esto se parece más al Excel original:
        chunk_por_clausula_con_entidades_ner
        chunking_clinico_desde_ner_bio_feature_np
    """
    texto = str(texto or "")
    entidades_norm = _normalizar_entidades(entidades)
    total_palabras_doc = _token_count(texto)

    if not entidades_norm:
        return [{
            "chunk_id": 1,
            "id": 1,
            "chunk": texto.strip(),
            "texto": texto.strip(),
            "tipo": "chunk_por_clausula_sin_entidades_ner_web",
            "metodo": "chunking_clinico_desde_ner_bio_feature_np_web",
            "start_char": 0,
            "end_char": len(texto.strip()),
            "longitud_caracteres": len(texto.strip()),
            "longitud_palabras": _token_count(texto),
            "total_entidades": 0,
            "labels_chunk": ["SIN_ETIQUETA"],
            "tipos": ["SIN_ETIQUETA"],
            "labels_chunk_texto": "SIN_ETIQUETA",
            "negado": False,
            "entidades_chunk_json": [],
            "entidades": [],
            "categoria": "SIN_ETIQUETA",
            "categoria_principal": "SIN_ETIQUETA",
            "feature_engineering_np": _feature_engineering_web(texto, ["SIN_ETIQUETA"], 1, total_palabras_doc),
            "nivel_complejidad": "baja",
            "score_complejidad": 0,
            "es_termino_medico": False,
            "requiere_explicacion": False,
        }]

    intervalos = _intervalos_por_clausula(texto, entidades_norm)
    total_chunks = len(intervalos)
    chunks = []

    for idx, (start, end) in enumerate(intervalos, start=1):
        chunk_texto = texto[start:end].strip(" ,;.")
        # Ajustar start/end si strip quitó espacios/puntuación.
        while start < end and texto[start] in " ,;.":
            start += 1
        while end > start and texto[end - 1] in " ,;.":
            end -= 1
        chunk_texto = texto[start:end].strip()

        ents = _entidades_en_intervalo(entidades_norm, start, end)

        # Si el intervalo no capturó entidades por ajuste de bordes, usar cercanas.
        if not ents:
            ents = [
                e for e in entidades_norm
                if not (e["end_char"] <= start or e["start_char"] >= end)
            ]

        labels = sorted(set(e["label"] for e in ents)) or ["SIN_ETIQUETA"]
        negado = _detectar_negacion(texto, start, end, ents)

        entidades_chunk_json = []
        for e in ents:
            ent_negada = bool(e.get("negado", False))
            if negado and e["start_char"] >= start:
                # Si hay "sin" antes de esta entidad dentro del mismo chunk,
                # se marca negada solo cuando aparece después de la negación.
                sub = texto[start:e["start_char"]].lower()
                if re.search(r"\bsin\b|\bno\b", sub):
                    ent_negada = True

            entidades_chunk_json.append({
                "texto": e["texto"],
                "label": e["label"],
                "start_char": e["start_char"],
                "end_char": e["end_char"],
                "negado": ent_negada,
            })

        categoria = _categoria_principal(labels, negado)
        feature = _feature_engineering_web(chunk_texto, labels, total_chunks, total_palabras_doc)
        score = feature["score_complejidad_np_normalizado_10"]
        nivel = feature["nivel_complejidad_np"]

        requiere_explicacion = bool(
            score >= 4.0
            or any(l in labels for l in ["ALTERACION", "HALLAZGO_CLINICO", "CONDICION", "PROCESO"])
            or negado
        )

        chunk = {
            "chunk_id": idx,
            "id": idx,
            "chunk": chunk_texto,
            "texto": chunk_texto,
            "tipo": "chunk_por_clausula_con_entidades_ner",
            "metodo": "chunking_clinico_desde_ner_bio_feature_np_web",
            "start_char": start,
            "end_char": end,
            "longitud_caracteres": end - start,
            "longitud_palabras": _token_count(chunk_texto),
            "total_entidades": len(entidades_chunk_json),
            "labels_chunk": labels,
            "tipos": labels,
            "labels_chunk_texto": ", ".join(labels),
            "negado": negado,
            "entidades_chunk_json": entidades_chunk_json,
            "entidades": [e["texto"] for e in entidades_chunk_json],
            "categoria": categoria,
            "categoria_principal": categoria,
            "feature_engineering_np": feature,
            "nivel_complejidad": nivel,
            "score_complejidad": score,
            "es_termino_medico": bool(ents),
            "requiere_explicacion": requiere_explicacion,
            "base_calculo_complejidad": {
                "criterio_usado": "Rúbrica computacional simplificada para ejecución web alineada al flujo 04.",
                "formula": "score_final = score_complejidad_np_normalizado_10",
                "score_np_10": score,
                "rasgos_linguisticos_np": {
                    "longitud_palabras_chunk": _token_count(chunk_texto),
                    "terminos_largos": feature["terminos_largos"],
                    "nominalizaciones": feature["nominalizaciones"],
                    "cantidad_nominalizaciones": feature["cantidad_nominalizaciones"],
                    "nivel_complejidad_np": nivel,
                },
                "nota_ner": "Las entidades NER guían la segmentación clínica, pero no alteran el diagnóstico original.",
            },
        }

        chunks.append(chunk)

    return chunks
