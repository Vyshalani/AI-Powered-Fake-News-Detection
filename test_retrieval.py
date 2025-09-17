from retrieval.retriever import fetch_evidence

claim = "Namibia wins AFCON 2025"
evidence = fetch_evidence(claim, num_results=3)

print("\n=== Evidence Results ===")
for ev in evidence:
    print(ev)
