"""
AI Skin Assistant chatbot: retrieval from ingredients/products + optional LLM for natural replies.
"""
import os
from skin_knowledge import (
    load_ingredients,
    load_products,
    load_skinme_products,
    search_ingredients,
    search_products_by_concern,
    search_skinme_by_concern,
    search_skinme_db_by_concern,
    get_ingredient_by_name,
    get_products_containing_ingredient,
)

# Lazy-load data
_ingredients_df = None
_products_df = None
_skinme_df = None


def _get_ingredients():
    global _ingredients_df
    if _ingredients_df is None:
        _ingredients_df = load_ingredients()
    return _ingredients_df


def _get_products():
    global _products_df
    if _products_df is None:
        _products_df = load_products()
    return _products_df


def _get_skinme():
    global _skinme_df
    if _skinme_df is None:
        _skinme_df = load_skinme_products()
    return _skinme_df


# Concern keywords -> product type hints
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


def _detect_concern_type(text: str):
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


def _get_concern_followup(user_message: str) -> str | None:
    """
    If the user mentions a skin problem vaguely (e.g. 'I have acne' without asking for products),
    return a friendly follow-up question to understand their problem type. Otherwise return None.
    """
    msg = user_message.strip().lower()
    # User is clearly asking for recommendations or products — don't interrupt with questions
    if any(w in msg for w in (
        "recommend", "suggest", "product", "products", "what can i use", "what should i use",
        "give me", "find me", "best for", "good for", "help with", "to treat", "to fix",
    )):
        return None
    # Short or statement-like message that mentions a concern — ask for more detail
    concern_words = ["acne", "dry", "oil", "oily", "sensitive", "redness", "aging", "wrinkle", "pigment", "hydrat"]
    has_concern = any(c in msg for c in concern_words)
    is_short = len(msg) < 100
    is_statement = any(msg.startswith(p) for p in (
        "i have", "i got", "my skin", "i suffer", "suffering", "dealing with", "struggling",
        "help", "problem", "issues with", "concern",
    )) or not msg.endswith("?")
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
        # Generic follow-up for other concerns
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
        parts.append(f"What it is: {what[:400]}{'...' if len(str(what)) > 400 else ''}")
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


def reply_with_retrieval(user_message: str, use_database: bool = False) -> str:
    """Answer using ingredients and products. If use_database=True, query MySQL (skinme_db) for product recommendations."""
    ingredients_df = _get_ingredients()
    products_df = _get_products()
    msg = user_message.strip().lower()

    # 0) "Products with [ingredient]" / "recommend products containing X"
    for start in ("products with ", "products containing ", "product with ", "containing "):
        if start in msg:
            idx = msg.find(start)
            name = user_message[idx + len(start):].strip()
            # trim trailing question etc
            name = name.rstrip("?.!").strip()
            if name:
                prods = get_products_containing_ingredient(name, products_df, 6)
                if prods:
                    ing = get_ingredient_by_name(name, ingredients_df)
                    out = ""
                    if ing:
                        out = _format_ingredient(ing) + "\n\n"
                    out += "**Products containing it:**\n" + "\n".join(_format_product(p) for p in prods)
                    return out
                # try ingredient name for product search
                ing = get_ingredient_by_name(name, ingredients_df)
                if ing:
                    prods = get_products_containing_ingredient(ing.get("name", name), products_df, 6)
                    if prods:
                        return _format_ingredient(ing) + "\n\n**Products:**\n" + "\n".join(_format_product(p) for p in prods)
            break

    # 1) "What is [ingredient]?" or "Tell me about [ingredient]"
    for start in ("what is ", "tell me about ", "ingredient ", "what does ", "explain "):
        if msg.startswith(start):
            name = user_message[len(start):].strip().rstrip("?.!")
            if name:
                ing = get_ingredient_by_name(name, ingredients_df)
                if ing:
                    out = _format_ingredient(ing)
                    prods = get_products_containing_ingredient(ing.get("name", name), products_df, 3)
                    if prods:
                        out += "\n\n**Products containing it:**\n" + "\n".join(_format_product(p) for p in prods)
                    return out
                hits = search_ingredients(name, ingredients_df, 3)
                if hits:
                    return "\n\n".join(_format_ingredient(h) for h in hits)
            break

    # 1.5) Vague skin problem (e.g. "I have acne") — ask what type of problem, like a real person would
    followup = _get_concern_followup(user_message)
    if followup:
        return followup

    # 2) "Products for [concern]" / "Recommend for dry skin" / "I have acne" — optional DB, then SkinMe CSV
    concern_type = _detect_concern_type(user_message)
    if concern_type or any(w in msg for w in ("product", "recommend", "suggest", "for my", "for dry", "for oily", "for sensitive", "for acne", "moisturiser", "moisturizer")):
        concern = user_message
        ing_hits = search_ingredients(concern, ingredients_df, 4)
        prod_hits = []
        from_db = False
        if use_database:
            prod_hits = search_skinme_db_by_concern(concern, product_type=concern_type, top_k=5)
            from_db = bool(prod_hits)
        if not prod_hits:
            skinme_df = _get_skinme()
            prod_hits = search_skinme_by_concern(concern, skinme_df, product_type=concern_type, top_k=5) if not skinme_df.empty else []
            if not prod_hits:
                prod_hits = search_products_by_concern(concern, products_df, product_type=concern_type, top_k=5)
        parts = []
        if ing_hits:
            parts.append("**Ingredients that may help:**\n" + "\n\n".join(_format_ingredient(h) for h in ing_hits[:3]))
        if prod_hits:
            label = "**Product suggestions (from database):**" if from_db else "**Product suggestions:**"
            parts.append(label + "\n" + "\n".join(_format_product(p) for p in prod_hits))
        if parts:
            return "\n\n".join(parts)

    # 3) General ingredient search
    ing_hits = search_ingredients(user_message, ingredients_df, 5)
    if ing_hits:
        return "**Matching ingredients:**\n\n" + "\n\n".join(_format_ingredient(h) for h in ing_hits)

    # 4) Greeting / fallback
    if any(w in msg for w in ("hi", "hello", "hey")):
        return (
            "Hi! I’m your Skin Assistant. You can ask me:\n"
            "• **What is [ingredient]?** — e.g. “What is niacinamide?”\n"
            "• **Recommend products for [concern]** — e.g. “Products for dry skin” or “I have acne”\n"
            "• **Products with [ingredient]** — I’ll suggest products that contain it.\n"
            "Ask me anything about ingredients or product recommendations."
        )

    return (
        "I’m not sure how to answer that. Try asking:\n"
        "• **What is niacinamide?**\n"
        "• **Recommend products for dry skin**\n"
        "• **Products with hyaluronic acid**"
    )


def reply_with_llm(user_message: str, conversation_history: list[dict], use_database: bool = False) -> str:
    """Use OpenAI to generate a reply, augmented with retrieval. Set OPENAI_API_KEY to use."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return reply_with_retrieval(user_message, use_database=use_database)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    except Exception:
        return reply_with_retrieval(user_message, use_database=use_database)

    ingredients_df = _get_ingredients()
    products_df = _get_products()
    ing_hits = search_ingredients(user_message, ingredients_df, 5)
    concern_type = _detect_concern_type(user_message)
    prod_hits = []
    if use_database:
        prod_hits = search_skinme_db_by_concern(user_message, product_type=concern_type, top_k=5)
    if not prod_hits:
        skinme_df = _get_skinme()
        prod_hits = search_skinme_by_concern(user_message, skinme_df, product_type=concern_type, top_k=5) if not skinme_df.empty else []
    if not prod_hits:
        prod_hits = search_products_by_concern(user_message, products_df, product_type=concern_type, top_k=5)
    context_parts = []
    if ing_hits:
        context_parts.append("Relevant ingredients (name, what_is_it, what_does_it_do):\n" + "\n".join(
            f"- {h.get('name')}: {str(h.get('what_is_it', ''))[:200]} | {str(h.get('what_does_it_do', ''))[:200]}"
            for h in ing_hits
        ))
    if prod_hits:
        context_parts.append("Relevant products (product_name, product_type, price):\n" + "\n".join(
            f"- {p.get('product_name')} ({p.get('product_type')}) {p.get('price', '')}"
            for p in prod_hits
        ))
    context = "\n\n".join(context_parts) if context_parts else "No specific ingredients or products found."

    system = """You are a friendly, professional Skin Care Assistant. You help users understand ingredients and choose skincare products.

IMPORTANT — When the user mentions a skin problem (acne, dry skin, oily, sensitive, redness, etc.) but does NOT give enough detail:
- Ask follow-up questions like a real person would. For example:
  - For acne: ask what type (blackheads, whiteheads, cystic, hormonal), severity (mild/moderate), and skin type (oily/combination/sensitive).
  - For dry skin: ask if it's tight/flaky, which areas (face/body), and if they have sensitivity.
  - For oily: ask if it's T-zone or all over, and if they have breakouts or enlarged pores.
  - For sensitive: ask what triggers them (products, weather) and if they have redness or itching.
- Only after they share more detail (or explicitly ask for product recommendations) should you suggest specific ingredients and products from the context.

Use the provided context (ingredients and products) to give accurate, concise answers. If the context doesn't contain relevant info, say so and give general skincare advice.
Be brief and helpful. When recommending products, mention 1–3 by name and price if available. Do not make up product names or ingredients not in the context."""

    messages = [{"role": "system", "content": system}]
    messages.append({"role": "user", "content": f"Context from our database:\n{context}\n\nUser question: {user_message}"})
    # Optional: prepend recent history for continuity
    for m in conversation_history[-6:]:
        messages.insert(-1, {"role": m["role"], "content": m["content"][:500]})

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=500,
            temperature=0.5,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return reply_with_retrieval(user_message, use_database=use_database)


def get_reply(
    user_message: str,
    conversation_history: list[dict] | None = None,
    use_llm: bool = True,
    use_database: bool = False,
) -> str:
    """Get the assistant reply. Set use_database=True to check products from MySQL (skinme_db)."""
    # When user mentions a skin problem vaguely, ask what type (like a real person) before suggesting products
    followup = _get_concern_followup(user_message)
    if followup:
        return followup
    history = conversation_history or []
    if use_llm and os.environ.get("OPENAI_API_KEY"):
        return reply_with_llm(user_message, history, use_database=use_database)
    return reply_with_retrieval(user_message, use_database=use_database)
