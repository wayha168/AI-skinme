"""Chat service: retrieval + optional LLM for skin assistant replies."""
import ast
import os
import re
from typing import Any, Optional

from skin_assistant.infrastructure import KnowledgeRepository


def _gemini_api_key() -> Optional[str]:
    """Google AI API key for Gemini (GEMINI_API_KEY or GOOGLE_API_KEY)."""
    return (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip() or None


def _llm_enabled() -> bool:
    """Gemini is used when a key is set and GOOGLE_API_ENABLED is not false (Spring-style toggle)."""
    if os.environ.get("GOOGLE_API_ENABLED", "").strip().lower() in ("0", "false", "no"):
        return False
    return bool(_gemini_api_key())

CONCERN_TYPES = {
    "moistur": "moisturiser",
    "dry": "moisturiser",
    "hydrat": "moisturiser",
    "acne": "moisturiser",
    "oil": "moisturiser",
    "sensitive": "moisturiser",
    "redness": "moisturiser",
    "aging": "moisturiser",
    "wrinkle": "moisturiser",
    "pigment": "moisturiser",
    "spf": "moisturiser",
    "sun": "moisturiser",
}


def _detect_concern_type(text: str) -> Optional[str]:
    t = text.lower()
    for key, ptype in CONCERN_TYPES.items():
        if key in t:
            return ptype
    return None


def _is_recommendation_request(text: str) -> bool:
    """True when user is asking for product recommendations — reply must use DB/synced products only."""
    msg = text.strip().lower()
    return any(
        w in msg
        for w in (
            "recommend",
            "recommendation",
            "suggest",
            "suggestion",
            "products for",
            "product for",
            "for my skin",
            "for dry",
            "for oily",
            "for acne",
            "for sensitive",
        )
    )


# Follow-up questions when user mentions a skin problem vaguely (like a real person would ask)
CONCERN_FOLLOWUPS = {
    "acne": (
        "I'd like to recommend the right things for you. Can you tell me a bit more?\n\n"
        "• **What type of acne** do you have — blackheads, whiteheads, cystic, or hormonal breakouts?\n"
        "• **How severe** is it — mild, moderate, or more persistent?\n"
        "• **What's your skin type** — oily, combination, or sensitive?\n\n"
        "Once I know this, I can suggest ingredients and products that fit your situation."
    ),
    "dry": (
        "To help you better, it would help to know:\n\n"
        "• Is your skin **mostly dry** (tight or flaky), or also **sensitive or reactive**?\n"
        "• **Which areas** are most affected — face, body, or both?\n"
        "• Do you have any **other concerns** (e.g. redness, dullness)?\n\n"
        "Then I can suggest ingredients and products that suit you."
    ),
    "oil": (
        "Quick questions so I can tailor my suggestions:\n\n"
        "• Is the oiliness **all over** or mainly in the **T-zone** (forehead, nose)?\n"
        "• Do you also have **breakouts** or **enlarged pores**?\n"
        "• Is your skin **sensitive** or quite tolerant to products?\n\n"
        "With that, I can recommend the right products for you."
    ),
    "sensitive": (
        "I'd like to suggest things that won't irritate you. Can you share:\n\n"
        "• **What tends to trigger** your skin — certain products, weather, or stress?\n"
        "• Do you get **redness**, **itching**, or both?\n"
        "• Any **other concerns** (dryness, breakouts) alongside sensitivity?\n\n"
        "Then I can point you to gentle ingredients and products."
    ),
    "redness": (
        "To give you useful advice:\n\n"
        "• Is the redness **ongoing** or does it **flare up** sometimes?\n"
        "• Do you have **sensitive** or **reactive** skin, or rosacea?\n"
        "• **Which areas** are affected — cheeks, nose, or all over?\n\n"
        "I can then suggest calming ingredients and products."
    ),
    "aging": (
        "To recommend the right products:\n\n"
        "• What bothers you most — **fine lines**, **deeper wrinkles**, **dullness**, or **loss of firmness**?\n"
        "• Do you have **sensitive** skin or use **retinoids** already?\n"
        "• Any **other concerns** (dryness, pigmentation)?\n\n"
        "Then I can suggest ingredients and products that fit your goals."
    ),
    "wrinkle": (
        "To recommend the right products:\n\n"
        "• What bothers you most — **fine lines**, **deeper wrinkles**, **dullness**, or **loss of firmness**?\n"
        "• Do you have **sensitive** skin or use **retinoids** already?\n"
        "• Any **other concerns** (dryness, pigmentation)?\n\n"
        "Then I can suggest ingredients and products that fit your goals."
    ),
}


def _get_concern_followup(user_message: str, conversation_history: Optional[list[dict]] = None) -> Optional[str]:
    """
    If the user mentions a skin problem vaguely (e.g. 'I have acne' without asking for products),
    return a friendly follow-up question to understand their problem type. Otherwise return None.
    """
    history = conversation_history or []
    # Continue the thread: do not repeat vague follow-ups after the assistant has already replied
    if history and any(m.get("role") == "assistant" for m in history):
        return None
    msg = user_message.strip().lower()
    # Photo analysis already encodes the condition — use retrieval/LLM with DB instead of vague follow-up
    if "from my skin photo the condition appears to be " in msg:
        return None
    if any(
        w in msg
        for w in (
            "recommend",
            "suggest",
            "product",
            "products",
            "what can i use",
            "what should i use",
            "give me",
            "find me",
            "best for",
            "good for",
            "help with",
            "to treat",
            "to fix",
        )
    ):
        return None
    concern_words = ["acne", "dry", "oil", "oily", "sensitive", "redness", "aging", "wrinkle", "pigment", "hydrat"]
    has_concern = any(c in msg for c in concern_words)
    is_short = len(msg) < 100
    is_statement = any(
        msg.startswith(p)
        for p in (
            "i have",
            "i got",
            "my skin",
            "i suffer",
            "suffering",
            "dealing with",
            "struggling",
            "help",
            "problem",
            "issues with",
            "concern",
        )
    ) or not msg.endswith("?")
    if has_concern and is_short and is_statement:
        if "acne" in msg:
            return CONCERN_FOLLOWUPS["acne"]
        if "dry" in msg or "hydrat" in msg:
            return CONCERN_FOLLOWUPS["dry"]
        if "oil" in msg:
            return CONCERN_FOLLOWUPS["oil"]
        if "sensitive" in msg:
            return CONCERN_FOLLOWUPS["sensitive"]
        if "redness" in msg:
            return CONCERN_FOLLOWUPS["redness"]
        if "aging" in msg or "wrinkle" in msg:
            return CONCERN_FOLLOWUPS.get("aging") or CONCERN_FOLLOWUPS.get("wrinkle", CONCERN_FOLLOWUPS["acne"])
        return (
            "I'd like to help. Can you tell me a bit more?\n\n"
            "• **What exactly** is bothering you — breakouts, dryness, redness, or something else?\n"
            "• **Where** — face, body, or both?\n"
            "• **What's your skin type** if you know it — oily, dry, combination, or sensitive?\n\n"
            "Then I can suggest ingredients and products that fit."
        )
    return None


def _stringify_ingredient_field(val: Any) -> str:
    """Turn CSV list/tuple/string reprs into readable comma-separated text (fixes ugly [' ', 'Acne', ...] output)."""
    if val is None:
        return ""
    if isinstance(val, float) and val != val:  # NaN
        return ""
    if isinstance(val, (list, tuple, set)):
        bits = [str(x).strip() for x in val if str(x).strip()]
        return ", ".join(bits)
    s = str(val).strip()
    if not s:
        return ""
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                bits = [str(x).strip() for x in parsed if str(x).strip()]
                return ", ".join(bits)
        except (ValueError, SyntaxError):
            pass
    return s


def _format_ingredient(ing: dict) -> str:
    name = ing.get("name", "Unknown")
    what = _stringify_ingredient_field(ing.get("what_is_it", ""))
    what_do = _stringify_ingredient_field(ing.get("what_does_it_do", ""))
    good_for = _stringify_ingredient_field(ing.get("who_is_it_good_for", ""))
    avoid = _stringify_ingredient_field(ing.get("who_should_avoid", ""))
    parts = [f"**{name}**"]
    if what:
        parts.append(f"What it is: {str(what)[:400]}{'...' if len(str(what)) > 400 else ''}")
    if what_do:
        parts.append(f"Benefits: {str(what_do)[:400]}{'...' if len(str(what_do)) > 400 else ''}")
    if good_for:
        parts.append(f"Good for: {good_for}")
    if avoid:
        parts.append(f"Avoid if: {avoid}")
    return "\n".join(parts)


def _format_product(prod: dict) -> str:
    name = prod.get("product_name", "Unknown")
    ptype = prod.get("product_type", "")
    price = prod.get("price", "")
    url = prod.get("product_url", "")
    line = f"• **{name}**"
    if ptype:
        line += f" ({ptype})"
    if price:
        line += f" — {price}"
    if url:
        line += f" — [Link]({url})"
    return line


def _is_short_affirmation(text: str) -> bool:
    """True if the message is a short yes/ok style reply."""
    t = text.strip().lower()
    return t in ("yes", "yeah", "yep", "yea", "ok", "okay", "sure", "yup", "please", "go ahead")


CONCERN_WORDS = [
    "dry", "acne", "oil", "oily", "sensitive", "redness", "burn", "burned", "itch",
    "skin", "face", "moistur", "wrinkle", "aging", "pigment", "dull", "flaky", "tight",
]

# Specific problem types — when present we use this message for product fit; otherwise prefer last concern from history
SPECIFIC_CONCERN_WORDS = [
    "dry", "acne", "oil", "oily", "sensitive", "redness", "burn", "burned", "itch",
    "wrinkle", "aging", "pigment", "dull", "flaky", "tight", "breakout", "rash",
]


def _message_has_concern(text: str) -> bool:
    """True if the message clearly describes a skin concern (so we use it for product search)."""
    if not text or len(text.strip()) < 4:
        return False
    t = text.strip().lower()
    return any(c in t for c in CONCERN_WORDS)


def _message_has_specific_concern(text: str) -> bool:
    """True if the message names a specific skin problem (dry, acne, burn, etc.) so we use it for recommendations."""
    if not text or len(text.strip()) < 4:
        return False
    t = text.strip().lower()
    return any(c in t for c in SPECIFIC_CONCERN_WORDS)


def _last_user_concern(conversation_history: list[dict]) -> Optional[str]:
    """Return the last user message that mentions a skin concern, for context after 'yes' / 'ok' or generic 'product for my problem'."""
    for m in reversed(conversation_history):
        if m.get("role") != "user":
            continue
        content = (m.get("content") or "").strip()
        if len(content) > 8 and _message_has_concern(content):
            return content
    return None


def _search_text_for_knowledge(
    user_message: str,
    conversation_history: Optional[list[dict]] = None,
) -> str:
    """
    Build a concise string for ingredient/product search against local data (CSV) and optional MySQL.
    Strips the skin-photo prefix so queries match concerns and ingredients; falls back like retrieval mode.
    """
    history = conversation_history or []
    msg = (user_message or "").strip()
    if not msg:
        return msg
    low = msg.lower()
    marker = "from my skin photo the condition appears to be "
    if marker in low:
        try:
            start = low.index(marker) + len(marker)
            after = msg[start:].strip()
        except ValueError:
            after = msg
        if "." in after:
            cond_part, rest = after.split(".", 1)
            rest = rest.strip()
            merged = f"{cond_part.strip()} {rest}".strip() if rest else cond_part.strip()
            return merged
        return after
    if _message_has_specific_concern(msg):
        return msg
    prev = _last_user_concern(history)
    if prev:
        return prev
    return msg


def _parse_budget(text: str) -> Optional[float]:
    """Parse max price/budget from message (e.g. '15$', 'under 15', 'budget 20', 'max 10 dollars'). Returns None if not found."""
    if not text or not text.strip():
        return None
    # Match numbers with optional $ or "dollars" / "dollar"
    # e.g. 15$, $15, under 15, budget 20, max 10 dollars, under $15
    patterns = [
        r"(?:under|below|max|budget|within)\s*\$?\s*(\d+(?:\.\d+)?)",
        r"\$?\s*(\d+(?:\.\d+)?)\s*(?:dollars?|\$|usd)?",
        r"(\d+(?:\.\d+)?)\s*\$",
    ]
    for pat in patterns:
        m = re.search(pat, text.strip(), re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except (ValueError, IndexError):
                pass
    return None


def _filter_products_by_price(products: list[dict], max_price: float) -> list[dict]:
    """Keep only products with price <= max_price. Price may be string or number."""
    out = []
    for p in products:
        raw = p.get("price")
        if raw is None or raw == "":
            continue
        try:
            price = float(raw) if not isinstance(raw, (int, float)) else raw
            if price <= max_price:
                out.append(p)
        except (ValueError, TypeError):
            continue
    return out


# Terms that must match product name/type text when we have an identified skin concern (budget + recommendations)
_CONCERN_MATCH_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "acne": ("acne", "blemish", "breakout", "bha", "salicylic", "benzoyl", "peroxide", "clarif", "pore"),
    "dry": ("dry", "dehydrat", "hydrat", "moistur", "ceramide", "barrier", "hyaluronic", "glycerin", "humect"),
    "oil": ("oil", "oily", "sebum", "matte", "shine", "t-zone"),
    "oily": ("oil", "oily", "sebum", "matte", "shine", "t-zone"),
    "sensitive": ("sensitive", "calm", "soothe", "gentle", "barrier", "redness", "irritat"),
    "redness": ("redness", "rosacea", "calm", "soothe", "anti-red", "couperose"),
    "wrinkle": ("wrinkle", "line", "firm", "retinol", "peptide", "aging", "anti-age"),
    "aging": ("aging", "wrinkle", "line", "firm", "retinol", "peptide", "mature"),
    "pigment": ("pigment", "bright", "dark spot", "niacinamide", "vitamin c", "uneven", "tone"),
    "dull": ("dull", "bright", "radiance", "glow", "vitamin c"),
    "itch": ("itch", "eczema", "dermatitis", "calm", "oat"),
    "rash": ("rash", "irritat", "calm", "soothe"),
    "burn": ("burn", "soothe", "repair", "barrier"),
    "tight": ("tight", "dry", "hydrat", "moistur"),
    "flaky": ("flaky", "dry", "exfoliat", "hydrat"),
    "breakout": ("breakout", "acne", "blemish", "salicylic"),
}


def _concern_match_terms(search_query: str) -> list[str]:
    """Substrings product name/type should match for relevance to the user's problem."""
    if not search_query or not search_query.strip():
        return []
    t = search_query.lower()
    terms: list[str] = []
    for w in SPECIFIC_CONCERN_WORDS:
        if w in t:
            terms.extend(_CONCERN_MATCH_EXPANSIONS.get(w, (w,)))
    # Extra skin terms (e.g. rosacea, eczema) if present in query
    for extra in ("rosacea", "eczema", "melasma", "hyperpigmentation", "blackhead", "whitehead", "cystic"):
        if extra in t:
            terms.append(extra)
    # Dedupe, preserve order
    seen = set()
    out: list[str] = []
    for x in terms:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _product_blob(p: dict) -> str:
    parts = [
        str(p.get("product_name") or ""),
        str(p.get("product_type") or ""),
        str(p.get("product_url") or ""),
    ]
    return " ".join(parts).lower()


def _filter_products_for_user_problem(products: list[dict], search_query: str) -> list[dict]:
    """Prefer products whose name/type relate to the skin concern; if none match by title, keep search results."""
    terms = _concern_match_terms(search_query)
    if not terms:
        return products
    out = []
    for p in products:
        blob = _product_blob(p)
        if any(term in blob for term in terms):
            out.append(p)
    return out if out else products


class ChatService:
    """Produces assistant replies from user message and optional history."""

    def __init__(self, repo: Optional[KnowledgeRepository] = None):
        self._repo = repo or KnowledgeRepository()

    def reply_with_retrieval(
        self,
        user_message: str,
        use_database: bool = False,
        conversation_history: Optional[list[dict]] = None,
    ) -> str:
        msg = user_message.strip().lower()
        history = conversation_history or []

        # Short affirmation after we asked a follow-up: use previous user message as context
        if _is_short_affirmation(user_message):
            prev_concern = _last_user_concern(history)
            if prev_concern:
                concern_type = _detect_concern_type(prev_concern)
                use_db = use_database or _is_recommendation_request(prev_concern)
                ing_hits = self._repo.search_ingredients(prev_concern, 6)
                prod_hits = self._repo.search_skinme_products_for_chat_context(
                    prev_concern, ing_hits, product_type=concern_type, top_k=24, use_database=use_db
                )
                prod_hits = _filter_products_for_user_problem(prod_hits, prev_concern)
                max_price = _parse_budget(user_message)
                if max_price is not None:
                    prod_hits = _filter_products_by_price(prod_hits, max_price)[:8]
                    if not prod_hits:
                        prod_hits = self._repo.search_skinme_products_for_chat_context(
                            prev_concern, ing_hits, product_type=concern_type, top_k=20, use_database=use_db
                        )
                        prod_hits = sorted(
                            prod_hits,
                            key=lambda p: float(p.get("price") or 999) if p.get("price") else 999,
                        )[:5]
                else:
                    prod_hits = prod_hits[:5]
                parts = []
                if ing_hits and max_price is None:
                    parts.append("**Ingredients that may help:**\n" + "\n\n".join(_format_ingredient(h) for h in ing_hits[:3]))
                if prod_hits:
                    if max_price is not None:
                        parts.append(f"Here are products **within your ${max_price:.0f} budget**:\n" + "\n".join(_format_product(p) for p in prod_hits))
                    else:
                        parts.append("**Product suggestions:**\n" + "\n".join(_format_product(p) for p in prod_hits))
                if parts:
                    return "Based on what you told me, here are some suggestions:\n\n" + "\n\n".join(parts)
                return (
                    "I’d like to suggest the right products. Can you tell me a bit more? "
                    "For example: Is your skin mostly dry or sensitive? Which area is most affected — face or body?"
                )
            return (
                "Could you tell me a bit more so I can recommend the right products? "
                "For example: What’s your main skin concern — dryness, acne, sensitivity, or something else?"
            )

        # Products with [ingredient]
        for start in ("products with ", "products containing ", "product with ", "containing "):
            if start in msg:
                idx = msg.find(start)
                name = user_message[idx + len(start) :].strip().rstrip("?.!").strip()
                if name:
                    prods = self._repo.get_products_containing_ingredient(name, 6, use_database=use_database)
                    if prods:
                        ing = self._repo.get_ingredient_by_name(name)
                        out = (_format_ingredient(ing) + "\n\n") if ing else ""
                        out += "**Products containing it:**\n" + "\n".join(_format_product(p) for p in prods)
                        return out
                    ing = self._repo.get_ingredient_by_name(name)
                    if ing:
                        prods = self._repo.get_products_containing_ingredient(
                            ing.get("name", name), 6, use_database=use_database
                        )
                        if prods:
                            return _format_ingredient(ing) + "\n\n**Products:**\n" + "\n".join(_format_product(p) for p in prods)
                break

        # What is [ingredient]
        for start in ("what is ", "tell me about ", "ingredient ", "what does ", "explain "):
            if msg.startswith(start):
                name = user_message[len(start) :].strip().rstrip("?.!").strip()
                if name:
                    ing = self._repo.get_ingredient_by_name(name)
                    if ing:
                        out = _format_ingredient(ing)
                        prods = self._repo.get_products_containing_ingredient(
                            ing.get("name", name), 3, use_database=use_database
                        )
                        if prods:
                            out += "\n\n**Products containing it:**\n" + "\n".join(_format_product(p) for p in prods)
                        return out
                    hits = self._repo.search_ingredients(name, 3)
                    if hits:
                        return "\n\n".join(_format_ingredient(h) for h in hits)
                break

        # Vague skin problem — ask what type of problem (like a real person would)
        followup = _get_concern_followup(user_message, history)
        if followup:
            return followup

        # Products for concern — ingredients from all datasets; SkinMe products matched to those ingredients + concern
        concern_type = _detect_concern_type(user_message)
        has_product_intent = concern_type or any(
            w in msg
            for w in (
                "product",
                "recommend",
                "suggest",
                "for my",
                "for dry",
                "for oily",
                "for sensitive",
                "for acne",
                "moisturiser",
                "moisturizer",
                "budget",
                "under",
            )
        )
        if has_product_intent:
            max_price = _parse_budget(user_message)
            search_query = _search_text_for_knowledge(user_message, history)
            concern_type = _detect_concern_type(search_query)
            use_db = use_database or _is_recommendation_request(user_message) or _is_recommendation_request(search_query)
            ing_hits = self._repo.search_ingredients(search_query, 6)
            prod_hits = self._repo.search_skinme_products_for_chat_context(
                search_query, ing_hits, product_type=concern_type, top_k=24, use_database=use_db
            )
            prod_hits = _filter_products_for_user_problem(prod_hits, search_query)
            if max_price is not None:
                within_budget = _filter_products_by_price(prod_hits, max_price)
                if within_budget:
                    prod_hits = within_budget[:8]
                    budget_header = f"Here are products **within your ${max_price:.0f} budget**:\n"
                else:
                    def _price_val(p):
                        try:
                            return float(p.get("price") or 999)
                        except (ValueError, TypeError):
                            return 999
                    prod_hits = sorted(prod_hits, key=_price_val)[:5]
                    budget_header = f"We don't have products under ${max_price:.0f}. Here are the **closest in price**:\n"
                if not prod_hits:
                    return (
                        f"No products matched your skin concern within ${max_price:.0f}. "
                        "Try a slightly higher budget or rephrase your concern (e.g. acne, dry skin, oily skin)."
                    )
                return budget_header + "\n".join(_format_product(p) for p in prod_hits)
            prod_hits = prod_hits[:5]
            parts = []
            if prod_hits:
                if ing_hits:
                    parts.append("**Ingredients that may help:**\n" + "\n\n".join(_format_ingredient(h) for h in ing_hits[:3]))
                parts.append("**Product suggestions:**\n" + "\n".join(_format_product(p) for p in prod_hits))
            if parts:
                return "\n\n".join(parts)

        # Greeting — check before ingredient search so "Hi" doesn't match random ingredients
        if any(w in msg for w in ("hi", "hello", "hey")) and len(msg) < 20:
            return (
                "Hi! I'm your Skin Assistant. You can ask me:\n"
                "• **What is [ingredient]?** — e.g. \"What is niacinamide?\"\n"
                "• **Recommend products for [concern]** — e.g. \"Products for dry skin\" or \"I have acne\"\n"
                "• **Products with [ingredient]** — I'll suggest products that contain it.\n"
                "Ask me anything about ingredients or product recommendations."
            )

        # General ingredient search (only if message looks like a real query, not a greeting or affirmation)
        ing_hits = self._repo.search_ingredients(user_message, 5)
        if ing_hits:
            return "**Matching ingredients:**\n\n" + "\n\n".join(_format_ingredient(h) for h in ing_hits)

        return (
            "I'm not sure how to answer that. Try asking:\n"
            "• **What is niacinamide?**\n"
            "• **Recommend products for dry skin**\n"
            "• **Products with hyaluronic acid**"
        )

    def reply_with_llm(
        self, user_message: str, conversation_history: list[dict], use_database: bool = False
    ) -> str:
        api_key = _gemini_api_key()
        if not api_key:
            return self.reply_with_retrieval(
                user_message, use_database=use_database, conversation_history=conversation_history
            )
        try:
            from google import genai as google_genai
            from google.genai import types as genai_types
        except Exception:
            return self.reply_with_retrieval(
                user_message, use_database=use_database, conversation_history=conversation_history
            )

        search_query = _search_text_for_knowledge(user_message, conversation_history)
        concern_type = _detect_concern_type(search_query)
        use_db = use_database or _is_recommendation_request(user_message) or _is_recommendation_request(search_query)
        ing_hits = self._repo.search_ingredients(search_query, 6)
        prod_hits = self._repo.search_skinme_products_for_chat_context(
            search_query, ing_hits, product_type=concern_type, top_k=24, use_database=use_db
        )
        prod_hits = _filter_products_for_user_problem(prod_hits, search_query)
        max_price_llm = _parse_budget(user_message)
        budget_note = ""
        if max_price_llm is not None and prod_hits:
            within = _filter_products_by_price(prod_hits, max_price_llm)
            if within:
                prod_hits = within[:8]
                budget_note = f"User asked for products within ${max_price_llm:.0f}."
            else:
                def _pv_llm(p: dict) -> float:
                    try:
                        return float(p.get("price") or 999)
                    except (ValueError, TypeError):
                        return 999.0
                priced = [p for p in prod_hits if p.get("price") not in (None, "")]
                prod_hits = sorted(priced, key=_pv_llm)[:5] if priced else prod_hits[:5]
                budget_note = (
                    f"User budget ${max_price_llm:.0f}: no items in range for this concern; "
                    f"closest-priced options for the same concern are listed below."
                )
        else:
            prod_hits = prod_hits[:8]
        context_parts = []
        if ing_hits:
            context_parts.append(
                "Relevant ingredients (from knowledge base):\n"
                + "\n".join(
                    f"- {h.get('name')}: {_stringify_ingredient_field(h.get('what_is_it', ''))[:200]} | "
                    f"Benefits: {_stringify_ingredient_field(h.get('what_does_it_do', ''))[:200]} | "
                    f"Good for: {_stringify_ingredient_field(h.get('who_is_it_good_for', ''))[:150]}"
                    for h in ing_hits
                )
            )
        if prod_hits:
            prod_lines = "\n".join(
                f"- {p.get('product_name')} ({p.get('product_type')}) {p.get('price', '')}" for p in prod_hits
            )
            if budget_note:
                prod_lines = budget_note + "\n" + prod_lines
            prod_src = (
                "scraped skinme_products.csv + MySQL (skinme_db)"
                if use_db
                else "scraped skinme_products.csv"
            )
            context_parts.append(f"Relevant products (from {prod_src}):\n" + prod_lines)
        context = "\n\n".join(context_parts) if context_parts else "No specific ingredients or products found."

        system = """You are a friendly Skin Care Assistant. Use the provided context to give accurate, concise answers.

The context is loaded from CSV files under the project data/ folder. **Ingredient explanations** may come from ingredientsList.csv, ingredient_lookup.csv, Paula lists, and other ingredient data. **Product names and prices** must come only from the product list in the context (scraped skinme_products.csv, and MySQL skinme_db when enabled). They are chosen to align with those ingredient matches plus the user’s concern. Do not invent products, prices, or ingredient facts that are not supported by this context. If something is missing from the context, say so briefly instead of guessing.

Continue the conversation naturally: refer to what the user said earlier in the thread when answering follow-up questions, budget changes, or new details. Keep the same skin concern unless they clearly switch topics.

When they ask for products within a budget, only mention products from the context that fit their stated skin problem (do not suggest unrelated categories).

When the user mentions a skin problem but does NOT ask for products and context is thin, you may ask brief follow-up questions. When they ask for help, recommendations, or what to use, use ingredients and products from the context; mention 1–3 products by name and price when listed. Do not invent product names or ingredient facts not supported by the context."""
        history_lines: list[str] = []
        for m in conversation_history[-6:]:
            role = m.get("role")
            c = ((m.get("content") or "")[:500]).strip()
            if not c:
                continue
            if role == "user":
                history_lines.append(f"User: {c}")
            elif role == "assistant":
                history_lines.append(f"Assistant: {c}")
        blocks: list[str] = []
        if history_lines:
            blocks.append("Recent conversation:\n\n" + "\n\n".join(history_lines))
        blocks.append(
            f"Active skin concern / search focus (carry through the thread): {search_query}\n\n"
            f"Context:\n{context}\n\nUser message:\n{user_message}"
        )
        user_block = "\n\n---\n\n".join(blocks)

        model_name = (os.environ.get("GEMINI_MODEL") or "gemini-2.0-flash").strip()
        try:
            client = google_genai.Client(api_key=api_key)
            resp = client.models.generate_content(
                model=model_name,
                contents=user_block,
                config=genai_types.GenerateContentConfig(
                    system_instruction=system,
                    max_output_tokens=500,
                    temperature=0.5,
                ),
            )
            text = (getattr(resp, "text", None) or "").strip()
            if text:
                return text
        except Exception:
            pass
        return self.reply_with_retrieval(
            user_message, use_database=use_database, conversation_history=conversation_history
        )

    def get_reply(
        self,
        user_message: str,
        conversation_history: Optional[list[dict]] = None,
        use_llm: bool = True,
        use_database: bool = False,
    ) -> str:
        # When user mentions a skin problem vaguely, ask what type (like a real person) first
        followup = _get_concern_followup(user_message, conversation_history)
        if followup:
            return followup
        history = conversation_history or []
        if use_llm and _llm_enabled():
            return self.reply_with_llm(user_message, history, use_database=use_database)
        return self.reply_with_retrieval(
            user_message, use_database=use_database, conversation_history=history
        )
