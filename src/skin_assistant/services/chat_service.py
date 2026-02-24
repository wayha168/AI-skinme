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

    def reply_with_retrieval(self, user_message: str) -> str:
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
            prod_hits = self._repo.search_products_by_concern(user_message, product_type=concern_type, top_k=5)
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

    def reply_with_llm(self, user_message: str, conversation_history: list[dict]) -> str:
        if not os.environ.get("OPENAI_API_KEY"):
            return self.reply_with_retrieval(user_message)
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        except Exception:
            return self.reply_with_retrieval(user_message)

        ing_hits = self._repo.search_ingredients(user_message, 5)
        concern_type = _detect_concern_type(user_message)
        prod_hits = self._repo.search_products_by_concern(user_message, product_type=concern_type, top_k=5)
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
            return self.reply_with_retrieval(user_message)

    def get_reply(
        self,
        user_message: str,
        conversation_history: Optional[list[dict]] = None,
        use_llm: bool = True,
    ) -> str:
        history = conversation_history or []
        if use_llm and os.environ.get("OPENAI_API_KEY"):
            return self.reply_with_llm(user_message, history)
        return self.reply_with_retrieval(user_message)
