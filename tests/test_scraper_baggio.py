import scraper

DETAIL_HTML = """
<html><body>
<div>Aluguel</div><div class="x">R$ 3.490,00</div>
<span>IPTU</span><span>R$ 181,52</span>
<span>Condomínio</span><span>R$ 0,00</span>
<span>Aluguel + Taxas</span><div class="jet-listing-dynamic-field__content">R$ 3.921,67</div>
<ul class="elementor-icon-list-items elementor-inline-items">
<li><span class="elementor-icon-list-text">3</span></li>
<li><span class="elementor-icon-list-text">3</span></li>
<li><span class="elementor-icon-list-text">2</span></li>
<li><span class="elementor-icon-list-text">-</span></li>
<li><span class="elementor-icon-list-text">122m</span></li>
</ul>
<p>Área Privativa: 111</p><p>Área Total: 122</p>
<p>Rua Adolfo Matoso dos Santos, 28 – 82540410 – Boa Vista – Curitiba</p><a>ver rota</a>
<p>Imóveis para alugar</p><p>Rua Outra, 1 – 80.240-050 – Água Verde – Curitiba</p>
</body></html>
"""


def test_detail_reads_money_and_total():
    detail = scraper.parse_baggio_detail(DETAIL_HTML)
    assert detail["aluguel"] == 3490.0
    assert detail["condominio_iptu"] == 181.52
    assert detail["custo_total"] == 3921.67


def test_detail_reads_specs_in_icon_order_and_missing_dash():
    detail = scraper.parse_baggio_detail(DETAIL_HTML)
    assert detail["quartos"] == 3
    assert detail["banheiros"] == 2
    assert detail["vagas"] is None  # "-" = ausente, não zero


def test_area_prefers_private_over_total():
    assert scraper.parse_baggio_detail(DETAIL_HTML)["area"] == 111.0


def test_address_block_gives_city_and_normalized_cep():
    detail = scraper.parse_baggio_detail(DETAIL_HTML)
    assert detail["cidade"] == "Curitiba"
    assert detail["bairro"] == "Boa Vista"
    assert detail["cep"] == "82540-410"
    assert detail["endereco"] == "Rua Adolfo Matoso dos Santos, 28"


def test_address_ignores_suggested_listings_block():
    # O primeiro bloco "ver rota" é o do imóvel; os cards sugeridos não podem sobrescrever
    assert scraper.parse_baggio_detail(DETAIL_HTML)["bairro"] != "Água Verde"


def test_is_baggio_house_uses_taxonomy():
    assert scraper.is_baggio_house({"class_list": ["imovel_tipo-casa", "imovel_tipo-sobrado"]})
    assert not scraper.is_baggio_house({"class_list": ["imovel_tipo-casa", "imovel_tipo-casa-comercial"]})
    assert not scraper.is_baggio_house({"class_list": ["imovel_tipo-casa", "imovel_tipo-apartamento"]})
    assert not scraper.is_baggio_house({"class_list": ["imovel_tipo-sobrado"]})


def test_money_and_measure_formats_differ():
    text = "| Aluguel | R$ 8.800,00 | Área Total: | 246.75m²"
    assert scraper.parse_money(text, "Aluguel") == 8800.0
    assert scraper.parse_measure(text, "Área Total") == 246.75


def test_build_urls_cartesian_product():
    site = {"base_url": "https://x.com/a", "locations": ["l1", "l2"], "filters": ["casa"], "page_param": "pg", "pages": 2}
    urls = scraper.build_urls(site)
    assert len(urls) == 4
    assert urls[0] == "https://x.com/a/l1/casa/?pg=1"
