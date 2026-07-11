
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional


class ProcessRequest(BaseModel):
    texto: str
    modelos: List[str] = Field(default_factory=list)


class EntidadClinica(BaseModel):
    texto: str
    tipo: str
    inicio: int
    fin: int


class BioToken(BaseModel):
    token: str
    bio: str


class ChunkClinico(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: int
    texto: str
    tipos: List[str]
    entidades: List[str] = Field(default_factory=list)


class ResultadoModelo(BaseModel):
    model_config = ConfigDict(extra="allow")
    modelo_id: str
    nombre: str
    proveedor: str
    diagnostico_simplificado: str
    explicacion: str
    resumen_automatico: str
    modelo_usado: str
    modo: str


class ProcessResponse(BaseModel):
    registro_id: str
    excel_path: str
    diagnostico_original: str
    entidades: List[EntidadClinica]
    bio_tags: List[BioToken]
    chunks: List[ChunkClinico]
    resultados_modelos: List[ResultadoModelo]
    ner_modelo: str
    ner_modo: str
