import basedosdados as bd

BILLING_PROJECT = "desafio-rio-494000"

query = """
SELECT COUNT(*) as total
FROM `datario.adm_central_atendimento_1746.chamado`
WHERE data_particao >= '2023-01-01'
LIMIT 1
"""

print("Testando conexão com BigQuery...")
df = bd.read_sql(query, billing_project_id=BILLING_PROJECT)
print("Sucesso!")
print(df)
