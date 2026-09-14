"""Unit (v43): prompt/bigdata/progdoc grounded entries."""
from src.analyzers.action_map import phrase_lemmas, resolve_action_tools

MARKET = {'prompt engineering': 66, 'llm': 544, 'rag': 226,
          'big data': 186, 'техническая документация': 63, 'python': 3016}


def test_prompt_templates_map():
    for phrase in ['анализ и отладки промптов', 'базовых шаблонов промптов',
                   'взаимодействовать с ИИ посредством мультимодальных промптов',
                   'разработка дизайна и структуры промптов']:
        hit = resolve_action_tools(phrase_lemmas(phrase), MARKET)
        assert hit == ('prompt engineering', 'grounded'), (phrase, hit)


def test_unstructured_maps_bigdata():
    hit = resolve_action_tools(phrase_lemmas('неструктурированные данные'), MARKET)
    assert hit == ('big data', 'grounded')


def test_progdoc_maps_techdocs():
    hit = resolve_action_tools(phrase_lemmas('программная документация'), MARKET)
    assert hit == ('техническая документация', 'grounded')
