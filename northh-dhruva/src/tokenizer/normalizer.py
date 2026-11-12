import re
import unicodedata


def normalize_text(text: str) -> str:
    """Normalize input text for the Northh Dhruva Tokenizer.

    Steps:
    1. Apply Unicode NFC normalization.
    2. Strip invisible noise characters: zero-width space (\\u200B) and BOM (\\uFEFF).
    3. Collapse multiple horizontal spaces/tabs ([ \\t]+) into a single space ' ' while preserving \\n.

    Args:
        text: Input string to normalize.

    Returns:
        Cleaned, normalized string.
    """
    if not text:
        return ""

    # 1. Unicode NFC normalization
    text = unicodedata.normalize("NFC", text)

    # 2. Strip invisible noise characters (\u200B and \uFEFF)
    text = re.sub(r"[\u200B\uFEFF]", "", text)

    # 3. Collapse multiple horizontal spaces/tabs into a single " " while preserving newlines
    text = re.sub(r"[ \t]+", " ", text)

    return text

