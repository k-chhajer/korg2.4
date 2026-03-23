Return valid JSON only, matching this exact schema:

{
  "challenge_type": "factual|logical|missing_consideration|alternative_approach",
  "target": "exact claim snippet",
  "correction": "compact correction"
}

Constraints:
- target <= 20 tokens
- correction <= 30 tokens
- no markdown
- no extra keys
