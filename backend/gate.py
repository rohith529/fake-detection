"""Input gate for Veritas Desk.

Classifies incoming text before sending it to LLM verification.
Filters out inputs that are not verifiable factual claims:
- too_short (< 5 words)
- question (inquiries, interrogatives)
- prediction (future forecasts, predictions)
- opinion (editorials, first-person subjective views)
- claim (verifiable factual statements)
"""
import re
from typing import Dict, TypedDict


class GateResult(TypedDict):
    type: str
    reason: str


# Common question starter words (interrogatives and auxiliaries)
QUESTION_STARTERS = {
    "will", "is", "are", "was", "were", "does", "do", "did",
    "can", "could", "should", "would", "how", "what", "when",
    "where", "who", "whom", "whose", "why", "which", "shall",
    "may", "might", "has", "have", "had"
}

# Clear first-person opinion prefixes
OPINION_PREFIXES = [
    r"^editorial\b",
    r"^opinion\b",
    r"^op-ed\b",
    r"^commentary\b",
    r"^in my view\b",
    r"^in my opinion\b",
    r"^to my mind\b",
    r"^from my perspective\b",
    r"^i think\b",
    r"^i believe\b",
    r"^i feel\b",
    r"^i suspect\b",
    r"^i guess\b",
    r"^we believe\b",
    r"^in our view\b",
    r"^my view is\b",
    r"^my opinion is\b",
    r"^my take is\b",
]

# Opinion markers anywhere in the text
OPINION_INFIX_PATTERNS = [
    r"\bin my opinion\b",
    r"\bin my view\b",
    r"\bin our view\b",
    r"\bi personally (?:think|believe|feel)\b",
    r"\bi strongly (?:believe|feel|oppose|support)\b",
    r"\bmy personal opinion\b",
    r"\bmy personal view\b",
    r"\bto the best of my belief\b",
]

# Patterns where 'will' is a noun or part of a past-tense/legal context
WILL_NOUN_PATTERNS = [
    r"\b(?:a|the|his|her|their|my|our|its|last|free|good|political|iron|strong|execut\w+|probate)\s+will\b",
    r"\bwill\s+and\s+testament\b",
    r"\bwill\s+of\s+(?:the|his|her|their|god|all)\b",
]

# Case-sensitive proper noun pattern (e.g., "Will Smith", "Will Ferrell")
WILL_PROPER_NAME_PATTERN = r"\bWill\s+[A-Z][a-z]+\b"

# Past tense indicator words that signal completed past events
PAST_TENSE_WORDS = {
    "was", "were", "had", "did", "ruled", "won", "announced", "signed",
    "bequeathed", "died", "passed", "occurred", "held", "found", "reported",
    "stated", "declared", "confirmed", "decided", "concluded", "killed",
    "asked", "said", "inquired", "noted", "tweeted", "yesterday", "earlier"
}

# Future prediction markers
FUTURE_MARKERS = [
    r"\btomorrow\b",
    r"\bnext (?:week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    r"\b(?:weather\s+)?forecast(?:s|ed|ing)?\b",
    r"\bexpected to\b",
    r"\bprojected to\b",
    r"\bpredicted to\b",
    r"\bis going to\b",
    r"\bare going to\b",
    r"\bgoing to\b",
    r"\bin the coming (?:days|weeks|months|years)\b",
    r"\bby (?:202[7-9]|20[3-9]\d)\b",
]


def _strip_quotes(text: str) -> str:
    """Strip outermost quotation marks if the entire text is quoted."""
    s = text.strip()
    quote_pairs = [('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’")]
    for q_start, q_end in quote_pairs:
        if s.startswith(q_start) and s.endswith(q_end) and len(s) >= 2:
            return s[1:-1].strip()
    return s


def _has_quoted_question_inside_claim(text: str) -> bool:
    """Check if '?' appears solely inside quotes within an otherwise declarative statement."""
    # Match content inside double or single quotes
    without_quotes = re.sub(r'["“‘\'][^"”’\']*["”’\']', '', text).strip()
    # If the remaining text outside quotes does NOT contain '?' and has substantial content
    if "?" not in without_quotes and len(without_quotes.split()) >= 3:
        # Check if the outer sentence does not end with '?'
        if not text.strip().endswith("?"):
            return True
    return False


def classify_input(text: str) -> GateResult:
    """Classify input text into one of: too_short, question, prediction, opinion, claim."""
    if not text:
        return {
            "type": "too_short",
            "reason": "Text is empty or missing."
        }

    raw = text.strip()
    words = raw.split()

    # Rule 1: Too short (< 5 words)
    if len(words) < 5:
        return {
            "type": "too_short",
            "reason": "This text is too short (fewer than 5 words) to evaluate as a factual claim. Please provide a full headline or statement."
        }

    # Normalize outer quotes for whole-sentence evaluation
    unquoted = _strip_quotes(raw)
    unquoted_lower = unquoted.lower()
    first_word = re.sub(r'^[^\w]+|[^\w]+$', '', words[0]).lower()

    # Rule 2: Question detection
    # Case A: Contains '?' (unless it is purely inside a quoted statement of a larger declarative claim)
    if "?" in unquoted:
        if not _has_quoted_question_inside_claim(unquoted):
            return {
                "type": "question",
                "reason": "This is a question, not a factual statement. Rephrase it as a claim (for example, instead of 'Will it rain today in X?', write 'It rained in X today')."
            }

    # Case B: Starts with an interrogative/auxiliary without being a proper noun or declarative sentence
    if first_word in QUESTION_STARTERS:
        # Check for proper noun edge case like "Will Smith won an Oscar"
        is_proper_name = False
        if first_word == "will" and len(words) >= 2:
            second_word = words[1]
            if second_word[0].isupper() and second_word.lower() not in {"it", "the", "there", "we", "they", "he", "she", "you", "this", "that"}:
                is_proper_name = True

        if not is_proper_name:
            # Check if it ends with a period and has a declarative structure (e.g., "Did not attend the meeting" or "Does not apply")
            # But standard auxiliary starters ("Is X doing Y", "Will X do Y", "Did X do Y") are questions
            if not (first_word in {"does", "did", "do"} and len(words) >= 2 and words[1].lower() == "not"):
                return {
                    "type": "question",
                    "reason": f"This appears to be a question starting with '{words[0]}'. Rephrase it as a factual claim."
                }

    # Rule 3: Opinion detection
    # Starts with opinion prefix
    for pattern in OPINION_PREFIXES:
        if re.search(pattern, unquoted_lower):
            return {
                "type": "opinion",
                "reason": "This text appears to be an opinion, editorial, or personal commentary rather than an objective factual claim."
            }

    # Contains clear first-person opinion markers
    for pattern in OPINION_INFIX_PATTERNS:
        if re.search(pattern, unquoted_lower):
            return {
                "type": "opinion",
                "reason": "This text contains personal opinion markers ('in my opinion', 'I believe', etc.). Veritas Desk evaluates verifiable factual claims."
            }

    # Check if sentence has embedded quotes within an outer declarative reporting clause
    without_inner_quotes = re.sub(r'["“‘\'][^"”’\']*["”’\']', '', unquoted).strip()
    has_outer_reporting_clause = (
        len(without_inner_quotes.split()) >= 3 and
        any(w.lower() in PAST_TENSE_WORDS for w in without_inner_quotes.split())
    )

    # Rule 4: Prediction detection
    target_for_prediction = without_inner_quotes.lower() if has_outer_reporting_clause else unquoted_lower

    # Check for explicit future markers
    for pattern in FUTURE_MARKERS:
        if re.search(pattern, target_for_prediction):
            # Verify it's not a past progressive (e.g., "was going to", "were going to")
            if "was going to" not in target_for_prediction and "were going to" not in target_for_prediction:
                return {
                    "type": "prediction",
                    "reason": "This text appears to be a future prediction or forecast. Veritas Desk verifies claims about established past or present events."
                }

    # Check for auxiliary "will" as a future marker
    if re.search(r"\bwill\b", target_for_prediction):
        # Edge case: Check if "will" is used as a noun, proper name, or in a legal/past context
        is_noun_or_name = False
        if re.search(WILL_PROPER_NAME_PATTERN, unquoted):
            is_noun_or_name = True
        else:
            for pattern in WILL_NOUN_PATTERNS:
                if re.search(pattern, target_for_prediction):
                    is_noun_or_name = True
                    break

        # Check if the sentence has dominant past-tense verbs and references a past will/testament
        has_past_tense = any(w.lower() in PAST_TENSE_WORDS for w in words)

        if not is_noun_or_name:
            # If not a noun or proper name, check if it functions as a future auxiliary verb (e.g. "will rain", "will arrive")
            # If the context is strictly past ("announced that his will was executed"), it's safe
            # Otherwise, modal "will" signifies future action / prediction
            if not (has_past_tense and ("will was" in target_for_prediction or "will had" in target_for_prediction)):
                return {
                    "type": "prediction",
                    "reason": "This statement describes a future event or expectation using 'will'. Veritas Desk verifies claims about past or present events."
                }

    # If all checks pass, it is classified as a claim
    return {
        "type": "claim",
        "reason": "This input appears to be a verifiable factual claim."
    }


# ── Broader Daily Activity Categories ─────────────────────────────────────────

CATEGORIES = {
    "weather": {
        "id": "weather",
        "label": "Weather & Meteorology",
        "icon": "🌦️",
        "keywords": {
            "rain", "rainfall", "raining", "rainy", "downpour", "shower", "precipitation",
            "storm", "thunderstorm", "cyclone", "hurricane", "typhoon", "tornado", "blizzard",
            "snow", "snowfall", "hail", "hailstorm", "flood", "floods", "flooding", "waterlogging",
            "heatwave", "temperature", "coldwave", "celsius", "fahrenheit", "degrees", "monsoon",
            "drought", "imd", "meteorological", "weather", "forecast", "climate"
        },
        "time_sensitive": True
    },
    "sports": {
        "id": "sports",
        "label": "Sports & Athletics",
        "icon": "🏏",
        "keywords": {
            "cricket", "football", "soccer", "tennis", "hockey", "badminton", "basketball",
            "kabaddi", "chess", "match", "tournament", "championship", "cup", "world cup",
            "odi", "t20", "test match", "ipl", "fifa", "bcci", "icc", "olympics", "medal",
            "gold medal", "silver medal", "bronze medal", "wicket", "runs", "century", "half-century",
            "goal", "score", "scores", "scored", "defeated", "won against", "lost to", "draw",
            "trophy", "striker", "batsman", "bowler", "captain", "coach", "athlete"
        },
        "time_sensitive": True
    },
    "entertainment": {
        "id": "entertainment",
        "label": "Entertainment & Culture",
        "icon": "🎬",
        "keywords": {
            "actor", "actress", "cinema", "movie", "film", "bollywood", "hollywood",
            "tollywood", "kollywood", "box office", "crore club", "trailer", "teaser",
            "director", "producer", "singer", "song", "album", "concert", "oscar",
            "grammy", "national award", "filmfare", "celebrity", "star", "acting",
            "marriage", "wedding", "divorce", "dating", "paparazzi"
        },
        "time_sensitive": False
    },
    "politics": {
        "id": "politics",
        "label": "Politics & Governance",
        "icon": "🏛️",
        "keywords": {
            "minister", "prime minister", "chief minister", "president", "governor",
            "government", "parliament", "lok sabha", "rajya sabha", "assembly", "mla",
            "mp", "election", "elections", "voting", "voter", "poll", "polls", "bjp",
            "congress", "aap", "tmc", "dmk", "party", "cabinet", "policy", "scheme",
            "bill", "ordinance", "supreme court", "high court", "judiciary", "verdict",
            "bench", "constitution", "constitutional", "treaty", "sanctions", "diplomat"
        },
        "time_sensitive": False
    },
    "crime_disaster": {
        "id": "crime_disaster",
        "label": "Incidents & Disasters",
        "icon": "🚨",
        "keywords": {
            "earthquake", "tsunami", "landslide", "avalanche", "wildfire", "fire",
            "accident", "crash", "derailment", "collision", "explosion", "blast",
            "arrest", "arrested", "police", "cbi", "ed", "investigation", "theft",
            "murder", "killed", "injured", "death toll", "casualties", "collapse", "collapsed"
        },
        "time_sensitive": True
    },
    "health_science": {
        "id": "health_science",
        "label": "Health & Science",
        "icon": "🔬",
        "keywords": {
            "doctor", "hospital", "patient", "medical", "medicine", "cure", "cures",
            "cancer", "diabetes", "vaccine", "vaccination", "virus", "infection",
            "disease", "epidemic", "pandemic", "who", "fda", "icmr", "nasa", "isro",
            "spacecraft", "satellite", "moon", "mars", "space", "astronomy", "research",
            "scientist", "study"
        },
        "time_sensitive": False
    },
    "business": {
        "id": "business",
        "label": "Business & Economy",
        "icon": "📈",
        "keywords": {
            "bank", "banking", "rbi", "repo rate", "interest rate", "inflation",
            "gdp", "economy", "economic", "market", "stock", "sensex", "nifty",
            "shares", "rupee", "dollar", "crypto", "bitcoin", "revenue", "profit",
            "loss", "acquisition", "merger", "startup", "layoffs"
        },
        "time_sensitive": False
    }
}


def classify_category(text: str) -> dict:
    """Classify the claim into a daily activity category with domain metadata.
    
    Returns:
        dict with id, label, icon, time_sensitive, and matched_keywords.
    """
    text_lower = (text or "").lower()
    words = set(re.findall(r"[a-z0-9]+", text_lower))
    
    best_cat = None
    max_score = 0
    matched_kws = []

    for cat_id, cat_info in CATEGORIES.items():
        score = 0
        hits = []
        for kw in cat_info["keywords"]:
            if " " in kw:
                if kw in text_lower:
                    score += 2
                    hits.append(kw)
            else:
                if kw in words:
                    score += 1
                    hits.append(kw)
        if score > max_score:
            max_score = score
            best_cat = cat_info
            matched_kws = hits

    if best_cat and max_score >= 1:
        return {
            "id": best_cat["id"],
            "label": best_cat["label"],
            "icon": best_cat["icon"],
            "time_sensitive": best_cat["time_sensitive"],
            "matched_keywords": matched_kws
        }

    return {
        "id": "general",
        "label": "General News",
        "icon": "📰",
        "time_sensitive": False,
        "matched_keywords": []
    }
