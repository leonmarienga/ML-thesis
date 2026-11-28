require(authenticated, "Evidence is not verified; run the bootstrap in a fresh kernel")
print(json.dumps(claims, indent=2, sort_keys=True, allow_nan=False))
