"""Baixe explicitamente o dataset público original; usa somente a stdlib.

Uso: python scripts/baixar_dados.py
Ou:  python scripts/baixar_dados.py --arquivo-zip caminho/bike-sharing.zip
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile


FONTE_URL = "https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset"
DOWNLOAD_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
COLUNAS = [
    "instant", "dteday", "season", "yr", "mnth", "hr", "holiday", "weekday",
    "workingday", "weathersit", "temp", "atemp", "hum", "windspeed", "casual",
    "registered", "cnt",
]


def _arquivo_unico(arquivo_zip: ZipFile, nome: str) -> bytes:
    # Lê apenas os arquivos nomeados em memória. Não usa extract/extractall,
    # portanto caminhos internos do ZIP nunca são gravados no sistema.
    candidatos = [
        item for item in arquivo_zip.infolist()
        if not item.is_dir() and item.filename.replace("\\", "/").split("/")[-1].lower() == nome.lower()
    ]
    if len(candidatos) != 1:
        raise ValueError(f"O ZIP deve conter exatamente um arquivo '{nome}'.")
    if candidatos[0].file_size > 20_000_000:
        raise ValueError(f"Arquivo '{nome}' maior que o limite esperado de 20 MB.")
    return arquivo_zip.read(candidatos[0])


def preparar_dados(conteudo_zip: bytes, destino: Path, origem_download: str) -> dict:
    """Valide o conteúdo conhecido, grave bytes originais e registre SHA-256."""
    with ZipFile(io.BytesIO(conteudo_zip)) as arquivo_zip:
        conteudo_csv = _arquivo_unico(arquivo_zip, "hour.csv")
        leia_me = _arquivo_unico(arquivo_zip, "Readme.txt")
    leitor = csv.reader(io.StringIO(conteudo_csv.decode("utf-8-sig")))
    cabecalho = next(leitor, [])
    if cabecalho != COLUNAS:
        raise ValueError("O hour.csv recebido não contém as 17 colunas esperadas.")
    registros = list(leitor)
    if len(registros) != 17379 or any(len(linha) != len(COLUNAS) for linha in registros):
        raise ValueError("O hour.csv recebido não tem o formato esperado de 17.379 linhas × 17 colunas.")

    manifesto = {
        "dataset": "Bike Sharing — hour.csv",
        "autor": "Hadi Fanaee-T",
        "ano_publicacao": 2013,
        "fonte_url": FONTE_URL,
        "download_url": DOWNLOAD_URL,
        "origem_arquivo_utilizado": origem_download,
        "doi": "10.24432/C5W894",
        "licenca": "CC BY 4.0",
        "licenca_url": "https://creativecommons.org/licenses/by/4.0/",
        "obtido_em_utc": datetime.now(timezone.utc).isoformat(),
        "arquivo_original": "hour.csv",
        "arquivo_local": "dataset.csv",
        "registros": len(registros),
        "numero_colunas": len(cabecalho),
        "colunas": cabecalho,
        "sha256_zip": hashlib.sha256(conteudo_zip).hexdigest(),
        "sha256_csv": hashlib.sha256(conteudo_csv).hexdigest(),
        "sha256_readme": hashlib.sha256(leia_me).hexdigest(),
        "transformacoes_no_csv": "Nenhuma: bytes originais preservados; apenas renomeado para dataset.csv.",
    }
    destino.mkdir(parents=True, exist_ok=True)
    for nome, conteudo in {
        "dataset.csv": conteudo_csv,
        "Readme_original.txt": leia_me,
        "metadados.json": (json.dumps(manifesto, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    }.items():
        temporario = destino / f".{nome}.tmp"
        temporario.write_bytes(conteudo)
        temporario.replace(destino / nome)
    return manifesto


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arquivo-zip", type=Path, help="Use um ZIP oficial já baixado.")
    parser.add_argument("--destino", type=Path, default=Path(__file__).resolve().parents[1] / "dados")
    argumentos = parser.parse_args()
    try:
        if argumentos.arquivo_zip:
            conteudo_zip = argumentos.arquivo_zip.read_bytes()
            origem = "ZIP local informado explicitamente; proveniência declarada: URL oficial de download"
        else:
            requisicao = Request(DOWNLOAD_URL, headers={"User-Agent": "LaboratorioEstatisticoAcademico/1.0"})
            with urlopen(requisicao, timeout=60) as resposta:
                conteudo_zip = resposta.read(25_000_001)
            origem = DOWNLOAD_URL
        if len(conteudo_zip) > 25_000_000:
            raise ValueError("O ZIP excede o limite esperado de 25 MB.")
        manifesto = preparar_dados(conteudo_zip, argumentos.destino, origem)
    except (OSError, ValueError, BadZipFile, UnicodeError) as erro:
        parser.exit(1, f"Erro ao preparar o dataset: {erro}\n")
    print(f"Dados preparados em: {argumentos.destino.resolve()}")
    print(f"{manifesto['registros']:,} registros; {manifesto['numero_colunas']} colunas originais.")
    print(f"SHA-256 do CSV: {manifesto['sha256_csv']}")


if __name__ == "__main__":
    main()
