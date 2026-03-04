"""Chat service: retrieval + optional LLM for skin assistant replies."""
import os
from typing import Optional

from skin_assistant.infrastructure import KnowledgeRepository

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


def _get_concern_followup(user_message: str) -> Optional[str]:
    """
    If the user mentions a skin problem vaguely (e.g. 'I have acne' without asking for products),
    return a friendly follow-up question to understand their problem type. Otherwise return None.
    """
    msg = user_message.strip().lower()
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


def _format_ingredient(ing: dict) -> str:
    name = ing.get("name", "Unknown")
    what = ing.get("what_is_it", "")
    what_do = ing.get("what_does_it_do", "")
    good_for = ing.get("who_is_it_good_for", "")
    avoid = ing.get("who_should_avoid", "")
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


class ChatService:
    """Produces assistant replies from user message and optional history."""

    def __init__(self, repo: Optional[KnowledgeRepository] = None):
        self._repo = repo or KnowledgeRepository()

    def reply_with_retrieval(self, user_message: str, use_database: bool = False) -> str:
        msg = user_message.strip().lower()

        # Products with [ingredient]
        for start in ("products with ", "products containing ", "product with ", "containing "):
            if start in msg:
                idx = msg.find(start)
                name = user_message[idx + len(start) :].strip().rstrip("?.!").strip()
                if name:
                    prods = self._repo.get_products_containing_ingredient(name, 6)
                    if prods:
                        ing = self._repo.get_ingredient_by_name(name)
                        out = (_format_ingredient(ing) + "\n\n") if ing else ""
                        out += "**Products containing it:**\n" + "\n".join(_format_product(p) for p in prods)
                        return out
                    ing = self._repo.get_ingredient_by_name(name)
                    if ing:
                        prods = self._repo.get_products_containing_ingredient(ing.get("name", name), 6)
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
                        prods = self._repo.get_products_containing_ingredient(ing.get("name", name), 3)
                        if prods:
                            out += "\n\n**Products containing it:**\n" + "\n".join(_format_product(p) for p in prods)
                        return out
                    hits = self._repo.search_ingredients(name, 3)
                    if hits:
                        return "\n\n".join(_format_ingredient(h) for h in hits)
                break

        # Vague skin problem — ask what type of problem (like a real person would)
        followup = _get_concern_followup(user_message)
        if followup:
            return followup

        # Products for concern
        concern_type = _detect_concern_type(user_message)
        if concern_type or any(
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
            )
        ):
            ing_hits = self._repo.search_ingredients(user_message, 4)
            prod_hits = self._repo.search_products_by_concern(
                user_message, product_type=concern_type, top_k=5, use_database=use_database
            )
            parts = []
            if ing_hits:
                parts.append("**Ingredients that may help:**\n" + "\n\n".join(_format_ingredient(h) for h in ing_hits[:3]))
            if prod_hits:
                parts.append("**Product suggestions:**\n" + "\n".join(_format_product(p) for p in prod_hits))
            if parts:
                return "\n\n".join(parts)

        # General ingredient search
        ing_hits = self._repo.search_ingredients(user_message, 5)
        if ing_hits:
            return "**Matching ingredients:**\n\n" + "\n\n".join(_format_ingredient(h) for h in ing_hits)

        # Greeting
        if any(w in msg for w in ("hi", "hello", "hey")):
            return (
                "Hi! I'm your Skin Assistant. You can ask me:\n"
                "• **What is [ingredient]?** — e.g. \"What is niacinamide?\"\n"
                "• **Recommend products for [concern]** — e.g. \"Products for dry skin\" or \"I have acne\"\n"
                "• **Products with [ingredient]** — I'll suggest products that contain it.\n"
                "Ask me anything about ingredients or product recommendations."
            )

        return (
            "I'm not sure how to answer that. Try asking:\n"
            "• **What is niacinamide?**\n"
            "• **Recommend products for dry skin**\n"
            "• **Products with hyaluronic acid**"
        )

    def reply_with_llm(
        self, user_message: str, conversation_history: list[dict], use_database: bool = False
    ) -> str:
        if not os.environ.get("OPENAI_API_KEY"):
            return self.reply_with_retrieval(user_message, use_database=use_database)
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        except Exception:
            return self.reply_with_retrieval(user_message)

        ing_hits = self._repo.search_ingredients(user_message, 5)
        concern_type = _detect_concern_type(user_message)
        prod_hits = self._repo.search_products_by_concern(
            user_message, product_type=concern_type, top_k=5, use_database=use_database
        )
        context_parts = []
        if ing_hits:
            context_parts.append(
                "Relevant ingredients:\n"
                + "\n".join(
                    f"- {h.get('name')}: {str(h.get('what_is_it', ''))[:200]} | {str(h.get('what_does_it_do', ''))[:200]}"
                    for h in ing_hits
                )
            )
        if prod_hits:
            context_parts.append(
                "Relevant products:\n"
                + "\n".join(
                    f"- {p.get('product_name')} ({p.get('product_type')}) {p.get('price', '')}" for p in prod_hits
                )
            )
        context = "\n\n".join(context_parts) if context_parts else "No specific ingredients or products found."

        system = """You are a friendly Skin Care Assistant. Use the provided context to give accurate, concise answers.

When the user mentions a skin problem (acne, dry skin, oily, sensitive, redness) but does NOT give enough detail, ask follow-up questions like a real person: e.g. for acne ask type (blackheads/cystic/hormonal), severity, and skin type; for dry ask areas and if sensitive; for oily ask T-zone vs all over and breakouts. Only suggest specific products/ingredients after they share more detail or explicitly ask for recommendations.
Be brief. When recommending products, mention 1–3 by name and price. Do not make up product names or ingredients not in the context."""
        messages = [{"role": "system", "content": system}]
        messages.append({"role": "user", "content": f"Context:\n{context}\n\nUser: {user_message}"})
        for m in conversation_history[-6:]:
            messages.insert(-1, {"role": m["role"], "content": (m["content"] or "")[:500]})

        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=500,
                temperature=0.5,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return self.reply_with_retrieval(user_message, use_database=use_database)

    def get_reply(
        self,
        user_message: str,
        conversation_history: Optional[list[dict]] = None,
        use_llm: bool = True,
        use_database: bool = False,
    ) -> str:
        # When user mentions a skin problem vaguely, ask what type (like a real person) first
        followup = _get_concern_followup(user_message)
        if followup:
            return followup
        history = conversation_history or []
        if use_llm and os.environ.get("OPENAI_API_KEY"):
            return self.reply_with_llm(user_message, history, use_database=use_database)
        return self.reply_with_retrieval(user_message, use_database=use_database)
