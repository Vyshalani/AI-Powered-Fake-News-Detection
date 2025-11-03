import pipeline.detect as detect

# Test with a simple claim
test_claim = "Namibia wins AFCON 2025"
print(f"Testing with claim: '{test_claim}'")

try:
    result = detect.detect_claim(test_claim)
    print(f"✅ SUCCESS: {result}")
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()