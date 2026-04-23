"""Extract data from BigQuery using basedosdados.

This module handles all BigQuery interactions. No other module should query
BigQuery directly -- use cached files from data/raw/ instead.
"""
import os
import pandas as pd


def extract_chamados(billing_project_id: str, cache_path: str = "data/raw/chamados_2023_2024.parquet") -> pd.DataFrame:
    """Extract chamados from BigQuery (2023-2024) with caching."""
    if os.path.exists(cache_path):
        print(f"Loading cached: {cache_path}")
        return pd.read_parquet(cache_path)

    import basedosdados as bd

    query = """
    SELECT *
    FROM `datario.adm_central_atendimento_1746.chamado`
    WHERE data_particao >= '2023-01-01'
      AND data_particao <= '2024-12-31'
    """
    print("Querying BigQuery for chamados...")
    df = bd.read_sql(query, billing_project_id=billing_project_id)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_parquet(cache_path, index=False)
    print(f"Saved {len(df)} rows to {cache_path}")
    return df


def extract_auxiliary_table(table_name: str, billing_project_id: str, cache_path: str) -> pd.DataFrame:
    """Extract an auxiliary table from BigQuery with caching."""
    if os.path.exists(cache_path):
        print(f"Loading cached: {cache_path}")
        return pd.read_parquet(cache_path)

    import basedosdados as bd

    query = f"SELECT * FROM `datario.dados_mestres.{table_name}`"
    print(f"Querying BigQuery for {table_name}...")
    df = bd.read_sql(query, billing_project_id=billing_project_id)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_parquet(cache_path, index=False)
    print(f"Saved {len(df)} rows to {cache_path}")
    return df


def extract_all_auxiliary(billing_project_id: str) -> dict:
    """Extract all auxiliary tables."""
    tables = {
        "bairro": "data/raw/bairros.parquet",
        "area_planejamento": "data/raw/areas_planejamento.parquet",
        "regiao_administrativa": "data/raw/regioes_admin.parquet",
        "subprefeitura": "data/raw/subprefeituras.parquet",
    }
    results = {}
    for table, path in tables.items():
        results[table] = extract_auxiliary_table(table, billing_project_id, path)
    return results
