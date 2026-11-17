from pipeline.text import bairro_key, strip_accents


def test_strip_accents():
    assert strip_accents("Água Verde Mercês") == "Agua Verde Merces"


def test_bairro_key_ignores_case_accents_and_spaces():
    assert bairro_key("  Santa   Quitéria ") == bairro_key("santa quiteria")


def test_bairro_key_handles_non_strings():
    assert bairro_key(None) is None
    assert bairro_key(float("nan")) is None
