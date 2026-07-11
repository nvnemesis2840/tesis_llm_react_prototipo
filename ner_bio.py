\
import re
import unicodedata
from typing import List, Dict


PATRONES_CLINICOS = {
    "ESTUDIO": [
        r"\bexamen radiol[oó]gico\b",
        r"\bradiograf[ií]a\b",
        r"\btomograf[ií]a\b",
        r"\bresonancia magn[eé]tica\b",
        r"\becograf[ií]a\b",
    ],
    "NEGACION": [
        r"\bsin evidencia de\b",
        r"\bno se observa\b",
        r"\bno se evidencian\b",
        r"\bsin signos de\b",
        r"\bausencia de\b",
    ],
    "ANATOMIA": [
        r"\bpulmon(?:ar|ares)\b",
        r"\bpleural\b",
        r"\bhiliares\b",
        r"\bcervical\b",
        r"\blumbar\b",
        r"\bdorsal\b",
        r"\bventricular\b",
        r"\bcoraz[oó]n\b",
        r"\bcolumna\b",
        r"\bv[ée]rtebras?\b",
        r"\bdisco intervertebral\b",
        r"\bregi[oó]n cervical\b",
        r"\bregi[oó]n lumbar\b",
    ],
    "ALTERACION": [
        r"\bm[úu]ltiples n[oó]dulos pulmonares bilaterales\b",
        r"\bn[oó]dulos pulmonares\b",
        r"\bengrosamiento pleural focal\b",
        r"\badenopat[ií]as hiliares\b",
        r"\bdiseminaci[oó]n metast[aá]sica pulmonar\b",
        r"\bcarcinoma pulmonar avanzado\b",
        r"\bfractura\b",
        r"\bluxaci[oó]n\b",
        r"\bhipertrofia\b",
        r"\bcardiomegalia\b",
        r"\bosteofitos marginales\b",
        r"\bhernia discal\b",
        r"\bdisminuci[oó]n\b",
        r"\blesi[oó]n\b",
        r"\balteraci[oó]n\b",
    ],
    "HALLAZGO": [
        r"\bdistribuci[oó]n perif[eé]rica\b",
        r"\blordosis cervical\b",
        r"\bsignos de\b",
        r"\bpresencia de\b",
        r"\bhallazgo\b",
        r"\bdensidad [óo]sea\b",
    ],
    "SINTOMA": [
        r"\bs[ií]ntomas respiratorios progresivos\b",
        r"\bs[ií]ntomas respiratorios\b",
    ],
    "CONDICION": [
        r"\bpaciente oncol[oó]gico\b",
    ],
}


PRIORIDAD_TIPO = {
    "ESTUDIO": 8,
    "ALTERACION": 7,
    "HALLAZGO": 6,
    "CONDICION": 5,
    "SINTOMA": 5,
    "NEGACION": 4,
    "ANATOMIA": 3,
}


def normalizar_texto(texto: str) -> str:
    return re.sub(r"\s+", " ", texto.strip())


def quitar_acentos(texto: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )


def tokenizar_con_posiciones(texto: str):
    return [
        {"token": m.group(0), "inicio": m.start(), "fin": m.end()}
        for m in re.finditer(r"\w+|[^\w\s]", texto, flags=re.UNICODE)
    ]


def filtrar_superpuestas(entidades: List[Dict]) -> List[Dict]:
    ordenadas = sorted(
        entidades,
        key=lambda e: (
            -(e["fin"] - e["inicio"]),
            -PRIORIDAD_TIPO.get(e["tipo"], 0),
            e["inicio"],
        ),
    )
    seleccionadas = []
    for entidad in ordenadas:
        se_superpone = any(
            not (entidad["fin"] <= otra["inicio"] or entidad["inicio"] >= otra["fin"])
            for otra in seleccionadas
        )
        if not se_superpone:
            seleccionadas.append(entidad)
    return sorted(seleccionadas, key=lambda e: e["inicio"])


def detectar_entidades(texto: str) -> List[Dict]:
    entidades = []
    texto_norm = normalizar_texto(texto)
    for tipo, patrones in PATRONES_CLINICOS.items():
        for patron in patrones:
            for m in re.finditer(patron, texto_norm, flags=re.IGNORECASE):
                entidades.append({"texto": m.group(0), "tipo": tipo, "inicio": m.start(), "fin": m.end()})
    vistos = set()
    unicas = []
    for e in entidades:
        key = (e["inicio"], e["fin"], e["tipo"])
        if key not in vistos:
            vistos.add(key)
            unicas.append(e)
    return filtrar_superpuestas(unicas)


def generar_bio_tags(texto: str, entidades: List[Dict]) -> List[Dict]:
    bio = []
    entidades_ordenadas = sorted(entidades, key=lambda e: (e["inicio"], e["fin"]))
    for tok in tokenizar_con_posiciones(texto):
        etiqueta = "O"
        for e in entidades_ordenadas:
            dentro = tok["inicio"] >= e["inicio"] and tok["fin"] <= e["fin"]
            if dentro:
                etiqueta = f"B-{e['tipo']}" if tok["inicio"] == e["inicio"] else f"I-{e['tipo']}"
                break
        bio.append({"token": tok["token"], "bio": etiqueta})
    return bio
