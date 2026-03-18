"""OpenCorporates integration: look up company information.

Uses the free OpenCorporates REST API (no API key needed for basic search).
"""

import logging

import httpx

logger = logging.getLogger(__name__)

_API_BASE = "https://api.opencorporates.com/v0.4"


async def search_company(company_name: str) -> list[dict]:
    """Search for a company on OpenCorporates.

    Args:
        company_name: Company name to search for.

    Returns:
        List of dicts with keys: name, jurisdiction, registered_address,
        officers, url. Empty list on error.
    """
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{_API_BASE}/companies/search",
                params={"q": company_name, "per_page": 5},
            )
            resp.raise_for_status()

        data = resp.json()
        companies_data = data.get("results", {}).get("companies", [])

        results = []
        for entry in companies_data:
            company = entry.get("company", {})
            results.append(
                {
                    "name": company.get("name", ""),
                    "jurisdiction": company.get("jurisdiction_code", ""),
                    "registered_address": company.get("registered_address_in_full", ""),
                    "officers": [],  # officers require a separate API call
                    "url": company.get("opencorporates_url", ""),
                    "company_number": company.get("company_number", ""),
                    "status": company.get("current_status", ""),
                    "incorporation_date": company.get("incorporation_date", ""),
                }
            )

        logger.info(
            f"OpenCorporates: found {len(results)} companies for '{company_name}'"
        )
        return results

    except httpx.TimeoutException:
        logger.warning(f"OpenCorporates timed out for '{company_name}'")
        return []
    except Exception as e:
        logger.warning(f"OpenCorporates search failed for '{company_name}': {e}")
        return []
