import os
import json
import requests


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_T2T_MODEL_NAME = os.getenv("OPENROUTER_T2T_MODEL_NAME")


SYSTEM_PROMPT = """
SYSTEM:
You are a professional outfit recommendation assistant. Your job is to create up to 5 outfit sets (each a list of item ids) that best satisfy the user's intent, chosen occasions, and the provided selected items and catalog. Be pragmatic and conservative: only choose items that are present in the provided `selected_items` or `catalog` arrays (do not invent new item ids or items). Consider style compatibility (color, pattern, silhouette), formality, seasonality, and the user's query and occasions. Avoid duplicate outfits and avoid returning multiple near-identical sets.

IMPORTANT RULES:
- Output **only valid JSON** and **nothing else** (no surrounding text, no backticks).
- The top-level JSON MUST be an array (possibly empty) with at most 5 objects.
- Each object must contain exactly these keys: `"items"`, `"compatibility"`, `"description"`.
  - `"items"`: an array of item ids (strings). Each id must exist in `selected_items` or `catalog`.
  - `"compatibility"`: a float between 0.00 and 1.00 inclusive that represents how confident you are about this set (round to two decimal places).
  - `"description"`: a concise 12–30 word explanation of why these items work together for the user's query and occasions.
- Sort the returned outfits from most to least relevant (highest compatibility first).
- Return **at most 5** outfits. If none are suitable, return an empty array `[]`.
- Do not include any extra keys, confidence fields per item, or metadata — only the specified keys.
- Avoid near-duplicate outfits: if two outfits differ only by color shade or a single trivial swap, keep the higher-scoring one.
- If the user provided a query, prioritize satisfying it. If missing, optimize for occasions and selected items.
- If a requested combination contradicts (e.g., formal occasion but items are strictly sporty), still propose options but lower compatibility scores and explain gaps in the description.
"""

USER_PROMPT = """
USER INPUT:
- query: "{{user_query}}"   # may be empty string ""
- occasions: {{["Casual","Work"]}}   # array of chosen occasion strings (may be empty)
- selected_items: {{ [{{"id":"s1","caption":"black tee"}},{{"id":"p2","caption":"navy chinos"}}] }} 
  # array with minimal metadata; these are items user explicitly selected and should be considered primary.
- catalog: {{ [{{"id":"s1","caption":"black tee"}},{{"id":"p2","caption":"navy chinos"}},{{"id":"s3","caption":"white oxford shirt"}},{{"id":"p4","caption":"gray trousers"}}] }}
  # full catalog to draw additional items from

TASK:
Using the inputs above, return a JSON array (max 5 entries) of outfit suggestions that best match the user's intent and occasions. Each suggestion must use item ids drawn from `selected_items` or `catalog`. Prioritize including selected_items when they are relevant; you may add complementary catalog items to complete a full outfit.

OUTPUT FORMAT (exact JSON only):
[
  {{
    "items": ["s1","p2","shoe3"],
    "compatibility": 0.92,
    "description": "Short 12–30 word explanation why these pieces form a compatible outfit for the user's query and occasions."
  }},
  ...
]

EXAMPLE (for clarity only — DO NOT output this in real responses):
[
  {{
    "items": ["s1","p2"],
    "compatibility": 0.95,
    "description": "Black tee with navy chinos — simple, smart-casual combo ideal for a relaxed work meeting or coffee date."
  }},
  {{
    "items": ["s3","p4"],
    "compatibility": 0.78,
    "description": "White oxford and gray trousers — more formal option suitable for office or evening events; slightly less casual."
  }}
]

Constraints & heuristics:
- Use at least 2 items per outfit when possible (top + bottom). If only one complementary item is available, still return it but with lower compatibility.
- Prefer diversity among returned outfits (different styles, formality levels, color contrasts), unless the user's query asks otherwise.
- Round `compatibility` to two decimal places (0.00–1.00).
- If the catalog does not contain suitable complementary items, mention the limitation succinctly in the description and give the best possible set using available items.

Remember: output **only** the required JSON array and nothing else.

--------------
query: {query}
occasions: {occasions}
selected_items: {selected_items}
catalog: {catalog}

"""


def get_response(system_prompt, user_prompt):

    try:
        resp = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            data=json.dumps({
                "model": OPENROUTER_T2T_MODEL_NAME,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            })
        )

        resp = resp.json()
        resp = resp["choices"][0]["message"]["content"]
        resp = resp.replace("```json", "").replace("```", "")
        resp = json.loads(resp)
        print(resp)
    except Exception as e:
        print(resp)
        print(repr(e))
        resp = []
    return resp


def get_llm_response(query, occasions, items, catalog):
    resp = get_response(
        SYSTEM_PROMPT,
        USER_PROMPT.format(
            query=query,
            occasions=occasions,
            selected_items=items,
            catalog=catalog
        )
    )
    return resp