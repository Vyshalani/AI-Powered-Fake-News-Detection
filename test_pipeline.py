from pipeline.detect import detect_claim

claim = "Namcor arrests CEO for corruption"
verdict, confidence, evidence, similarity = detect_claim(claim)

print("\n" + "="*50)
print(f"Claim: {claim}")
print(f"Verdict: {verdict} | Confidence: {confidence:.2f}")
if similarity is not None:
    print(f"Evidence similarity: {similarity:.2f}")

print("\nDEBUG: Evidence length =", len(evidence))   # 👈 new debug line
if evidence:
    print(f"\nRetrieved {len(evidence)} evidence docs:")
    for i, e in enumerate(evidence, 1):
        print(f"\n--- Evidence {i} ---\n{e[:300]}")
else:
    print("\nNo evidence retrieved.")
