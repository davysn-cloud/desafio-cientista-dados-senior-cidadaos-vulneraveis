# API Reference

## Open-Meteo Historical Weather API

### Endpoint
`https://archive-api.open-meteo.com/v1/archive`

### Parameters for Rio de Janeiro
```python
params = {
    "latitude": -22.9068,
    "longitude": -43.1729,
    "start_date": "2023-01-01",
    "end_date": "2024-12-31",
    "daily": ",".join([
        "temperature_2m_max",
        "temperature_2m_min",
        "temperature_2m_mean",
        "precipitation_sum",
        "rain_sum",
        "windspeed_10m_max",
        "weathercode"
    ]),
    "timezone": "America/Sao_Paulo"
}
```

### Response Format
```json
{
  "daily": {
    "time": ["2023-01-01", "2023-01-02", ...],
    "temperature_2m_max": [32.1, 30.5, ...],
    "precipitation_sum": [0.0, 5.2, ...]
  }
}
```

### Rate Limits
- Free tier: 10,000 requests/day
- No API key required
- Cache response to disk immediately after fetching

### Documentation
https://open-meteo.com/en/docs

---

## Public Holiday API (Nager.Date)

### Endpoint
`https://date.nager.at/api/v3/PublicHolidays/{year}/{countryCode}`

### Usage
```python
import requests

holidays_2023 = requests.get('https://date.nager.at/api/v3/PublicHolidays/2023/BR').json()
holidays_2024 = requests.get('https://date.nager.at/api/v3/PublicHolidays/2024/BR').json()
```

### Response Format
```json
[
  {
    "date": "2023-01-01",
    "localName": "Confraternizacao Universal",
    "name": "New Year's Day",
    "countryCode": "BR",
    "fixed": true,
    "global": true,
    "types": ["Public"]
  }
]
```

### Rate Limits
- Free, no API key required
- Be respectful: cache after first call

---

## BigQuery (via basedosdados)

### Setup
```python
import basedosdados as bd

# First time: will prompt for GCP project ID and authentication
df = bd.read_sql("SELECT 1", billing_project_id="your-project-id")
```

### Key Tables
```sql
-- Main table (ALWAYS use partition filter!)
SELECT * FROM `datario.adm_central_atendimento_1746.chamado`
WHERE data_particao >= '2023-01-01' AND data_particao <= '2024-12-31'

-- Auxiliary tables (small, no filter needed)
SELECT * FROM `datario.dados_mestres.bairro`
SELECT * FROM `datario.dados_mestres.area_planejamento`
SELECT * FROM `datario.dados_mestres.regiao_administrativa`
SELECT * FROM `datario.dados_mestres.subprefeitura`
```

### Cost Control
- Always test with LIMIT 100 first
- Use partition filters (data_particao)
- Save results to local files immediately
- Never re-query if cached file exists
