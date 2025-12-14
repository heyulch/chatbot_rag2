import re
import unicodedata

PERSON_PATTERNS = {
  "heyul": r"\b(jorge|heyul|chavez|arias)\b",
  "maria":  r"\b(maria|elena|quispe|huaman)\b",
  "carlos":   r"\b(carlos|andres|mendoza|rios)\b",
}


def normalize(text: str) -> str:
    text = text.lower()
    text = ''.join(c for c in unicodedata.normalize('NFD', text)
                   if unicodedata.category(c) != 'Mn')
    text = re.sub(r"[^a-z0-9\s]", " ", text)  
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_people(query: str) -> list[str]:
    q = normalize(query)
    found = []
    for cv_id, pat in PERSON_PATTERNS.items():
        if re.search(pat, q):
            found.append(cv_id)
    return found

def route(query: str) -> dict:
    people = detect_people(query)
    if len(people) == 0:
        return {"mode": "single", "cv_ids": ["heyul"]}       
    if len(people) == 1:
        return {"mode": "single", "cv_ids": people}
    return {"mode": "multi", "cv_ids": people}