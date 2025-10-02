from pipeline.detect import detect_claim

claim = "Unverified reports claim gangs are kidnapping people daily for massive ransoms. Witnesses describe the crimes as shocking, unstoppable, and beyond imagination. Social media slogans repeat: “Kidnapped everywhere, kidnapped everywhere, kidnapped everywhere.” No police statistics or official data confirm this. Still, believers call it the worst kidnapping wave ever. Commentators insist Namibia is powerless to stop it. Critics call the rumors fabricated and exaggerated. Despite this, many share the story widely. Families describe themselves as terrified beyond measure. The story is being hailed as massive, magical, and unprecedented."
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
