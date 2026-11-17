import re
import unicodedata

# Variantes de grafia vistas nas fontes -> nome canônico (chave já normalizada)
BAIRRO_ALIASES = {
    "alto da xv": "Alto da Rua XV",
    "alto da rua xv": "Alto da Rua XV",
    "santa quiteria": "Santa Quitéria",
    "cidade industrial": "Cidade Industrial de Curitiba",
    "cic": "Cidade Industrial de Curitiba",
    "a verde": "Água Verde",
    "juveve": "Juvevê",
    "barigui": "Bariguí",
}


def strip_accents(text: str):
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def bairro_key(name):
    """Chave de comparação: sem acento, minúscula, espaços colapsados."""
    if not isinstance(name, str):
        return None
    return re.sub(r"\s+", " ", strip_accents(name).lower()).strip() or None
