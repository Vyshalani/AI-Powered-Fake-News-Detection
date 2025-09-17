from retrieval.search import search_claim

claim = "Namibia wins AFCON 2025"
urls = search_claim(claim, num_results=5)

print("Top URLs from trusted Namibian news sources:")
for i, url in enumerate(urls):
    print(f"{i+1}: {url}")
