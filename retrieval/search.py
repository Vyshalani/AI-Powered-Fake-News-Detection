from retrieval.search import fetch_from_site, TRUSTED_SITES

def fetch_evidence(claim_text, num_results=5):
    evidence = []
    for site, (url, selector) in TRUSTED_SITES.items():
        snippets = fetch_from_site(site, url, selector, claim_text, num_results)
        evidence.extend(snippets)
    return evidence[:num_results]