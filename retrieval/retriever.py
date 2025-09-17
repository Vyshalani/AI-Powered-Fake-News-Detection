import requests
from bs4 import BeautifulSoup
from urllib.parse import quote

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/115.0 Safari/537.36"
    )
}

TRUSTED_SITES = {
    "The Namibian": ("https://www.namibian.com.na/?s=", "h2.entry-title a"),
    "Republikein": ("https://www.republikein.com.na/search?query=", "h4.article-title a"),
    "Kosmos 94.1": ("https://kosmos.com.na/?s=", "h2.entry-title a"),
    "Namibian Sun": ("https://www.namibiansun.com/search?query=", "h4.article-title a"),
}


def fetch_from_site(site_name, base_url, selector, query, num_results=5):
    url = base_url + quote(query)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        print(f"[ERROR] {site_name} fetch failed: {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")

    # DEBUG: print some of the raw HTML so we can see the structure
    print(f"\n[DEBUG] First 500 chars of {site_name} HTML:\n", resp.text[:500])

    results = []
    for a in soup.select(selector):
        title = a.get_text(strip=True)
        href = a.get("href")

        if href and not href.startswith("http"):
            if "republikein.com.na" in base_url:
                href = "https://www.republikein.com.na" + href
            elif "namibiansun.com" in base_url:
                href = "https://www.namibiansun.com" + href
            elif "namibian.com.na" in base_url:
                href = "https://www.namibian.com.na" + href
            elif "kosmos.com.na" in base_url:
                href = "https://kosmos.com.na" + href

        if title and href:
            results.append(f"{title} ({href})")

    print(f"[INFO] Retrieved {len(results)} results from {site_name} using selector '{selector}'")
    return results[:num_results]


# 🔹 Add this function back
def fetch_evidence(claim_text, num_results=5):
    evidence = []
    for site, (url, selector) in TRUSTED_SITES.items():
        snippets = fetch_from_site(site, url, selector, claim_text, num_results)
        evidence.extend(snippets)
    return evidence[:num_results]
