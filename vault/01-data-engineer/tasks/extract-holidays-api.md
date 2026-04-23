# Task: Extract Holiday Data (Public Holiday API)

## Objective
Fetch Brazilian public holidays for 2023 and 2024 from Nager.Date API.

## Steps
1. Call `https://date.nager.at/api/v3/PublicHolidays/2023/BR`
2. Call `https://date.nager.at/api/v3/PublicHolidays/2024/BR`
3. Combine results into single DataFrame
4. Save to `data/raw/holidays_br_2023_2024.csv`
5. Verify: >10 holidays per year, dates are valid
