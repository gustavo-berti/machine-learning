import pandas as pd
import requests
import re
import time
import json
from html import unescape
from datetime import date
from pathlib import Path

BASE_DIR = Path(__file__).parent

COLUMNS = [
    "id", "fonte", "url", "data_coleta",
    "tipo", "endereco", "bairro", "cidade", "cep", "latitude", "longitude",
    "area", "quartos", "banheiros", "vagas",
    "aluguel", "condominio_iptu", "custo_total", "mobiliado",
]

NEXT_DATA_PATTERN = re.compile(
    r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', re.S
)
LD_JSON_PATTERN = re.compile(
    r'<script type="application/ld\+json"[^>]*>(.*?)</script>', re.S
)

QUINTOANDAR_FIELDS = {
    "id": "id",
    "tipo": "type",
    "area": "area",
    "quartos": "bedrooms",
    "banheiros": "bathrooms",
    "vagas": "parkingSpots",
    "aluguel": "rentPrice",
    "condominio_iptu": "condoIptu",
    "custo_total": "totalCost",
    "mobiliado": "isFurnished",
}

BAGGIO_API_FIELDS = "id,link,meta,class_list"
# "Rua X, 28 – 82.540-410 – Boa Vista – Curitiba | ver rota" (o CEP às vezes vem sem pontuação)
BAGGIO_ADDRESS_PATTERN = re.compile(r"\| ([^|]+?) [–-] ([\d.\-]{8,10}) [–-] ([^|]+?) [–-] ([^|]+?) \| ver rota")
BAGGIO_SPECS_PATTERN = re.compile(
    r'<ul class="elementor-icon-list-items elementor-inline-items">(.*?)</ul>', re.S
)
BAGGIO_SPEC_ITEM_PATTERN = re.compile(r'<span class="elementor-icon-list-text">(.*?)</span>', re.S)
# Termos da taxonomia imovel_tipo que desqualificam o anúncio como casa residencial
BAGGIO_EXCLUDED_TYPES = {"imovel_tipo-casa-comercial", "imovel_tipo-apartamento"}

CHAVESNAMAO_ID_PATTERN = re.compile(r"/id-(\d+)/")
CHAVESNAMAO_AREA_PATTERN = re.compile(r"(\d+)m2")
CHAVESNAMAO_TYPES = {
    "SingleFamilyResidence": "Casa",
    "House": "Casa",
    "Apartment": "Apartamento",
}


def load_json(file_path: Path):
    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(file_path: Path, data):
    with file_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def build_session(request_config: dict):
    session = requests.Session()
    session.headers.update({
        "user-agent": request_config["user_agent"],
        "accept-language": "pt-BR,pt;q=0.9",
    })
    return session


def fetch_html(session: requests.Session, url: str, request_config: dict):
    for attempt in range(1, request_config["max_retries"] + 1):
        try:
            response = session.get(url, timeout=request_config["timeout_seconds"])
        except requests.RequestException:
            response = None

        if response is not None and response.status_code == 200:
            return response.text

        time.sleep(request_config["delay_seconds"] * attempt)

    return None


def extract_next_data(html: str):
    match = NEXT_DATA_PATTERN.search(html)
    if match is None:
        return None
    return json.loads(match.group(1))


def extract_ld_json(html: str, schema_type: str):
    for block in LD_JSON_PATTERN.findall(html):
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue
        if data.get("@type") == schema_type:
            return data
    return None


def build_urls(site: dict):
    locations = site.get("locations") or [""]
    filters = site.get("filters") or [""]
    page_param = site.get("page_param")
    pages = site.get("pages", 1)

    urls = []
    for location in locations:
        for path_filter in filters:
            parts = [site["base_url"], location, path_filter]
            url = "/".join(part.strip("/") for part in parts if part)

            if page_param is None:
                urls.append(url)
                continue

            for page in range(1, pages + 1):
                urls.append(f"{url}/?{page_param}={page}")

    return urls


def parse_quintoandar(html: str, source_name: str):
    next_data = extract_next_data(html)
    if next_data is None:
        return []

    houses = next_data["props"]["pageProps"]["initialState"].get("houses", {})
    collected_at = date.today().isoformat()

    rows = []
    for house in houses.values():
        # O estado do Next mistura os anúncios com contadores auxiliares
        if not isinstance(house, dict) or "rentPrice" not in house:
            continue

        address = house.get("address") or {}
        row = {column: house.get(key) for column, key in QUINTOANDAR_FIELDS.items()}
        row["fonte"] = source_name
        row["url"] = f"https://www.quintoandar.com.br/imovel/{house['id']}"
        row["data_coleta"] = collected_at
        row["endereco"] = address.get("address")
        row["bairro"] = house.get("neighbourhood") or house.get("regionName")
        row["cidade"] = address.get("city")
        rows.append(row)

    return rows


def parse_chavesnamao_area(item: dict, url: str):
    floor_size = item.get("floorSize") or {}
    unit_text = floor_size.get("unitText")
    if unit_text:
        return pd.to_numeric(re.sub(r"\D", "", unit_text), errors="coerce")

    # Parte dos anúncios só traz a metragem no slug da URL (ex: ...-194m2-RS4500)
    match = CHAVESNAMAO_AREA_PATTERN.search(url)
    return int(match.group(1)) if match else None


def parse_chavesnamao(html: str, source_name: str):
    listing = extract_ld_json(html, "RealEstateListing")
    if listing is None:
        return []

    collected_at = date.today().isoformat()

    rows = []
    for offer in listing["offers"]["itemListElement"]:
        item = offer.get("itemOffered") or {}
        address = item.get("address") or {}
        geo = item.get("geo") or {}
        url = offer["url"]
        id_match = CHAVESNAMAO_ID_PATTERN.search(url)
        schema_type = item.get("@type")

        rows.append({
            "id": id_match.group(1) if id_match else None,
            "fonte": source_name,
            "url": url,
            "data_coleta": collected_at,
            "tipo": CHAVESNAMAO_TYPES.get(schema_type, schema_type),
            "endereco": address.get("streetAddress"),
            "bairro": address.get("addressLocality"),
            "cidade": (address.get("addressRegion") or "").split(",")[0],
            "cep": address.get("postalCode"),
            "latitude": geo.get("latitude"),
            "longitude": geo.get("longitude"),
            "area": parse_chavesnamao_area(item, url),
            "quartos": item.get("numberOfBedrooms"),
            "banheiros": item.get("numberOfBathroomsTotal"),
            "aluguel": offer.get("price"),
        })

    return rows


def html_to_text(html: str):
    """Achata o HTML em texto com '|' no lugar das tags, para casar rótulo e valor."""
    text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", html, flags=re.S)
    text = re.sub(r"<[^>]+>", "|", text)
    text = re.sub(r"[|\s]*\|[|\s]*", " | ", text)
    return re.sub(r"\s+", " ", text)


def parse_money(text: str, label: str):
    # Valores em real: "R$ 8.800,00" -> ponto de milhar, vírgula decimal
    match = re.search(rf"\|\s*{label}\s*\|\s*R\$\s*([\d.]+,\d{{2}})", text)
    if match is None:
        return None
    return float(match.group(1).replace(".", "").replace(",", "."))


def parse_measure(text: str, label: str):
    # Áreas: "Área Privativa: 111" ou "246.75m²" -> ponto decimal; ausente vem como "-"
    match = re.search(rf"{label}:\s*\|?\s*(\d[\d.]*)", text)
    return float(match.group(1)) if match else None


def parse_spec(value: str):
    """Itens da lista de características: '3', '122m', '54.2m' ou '-' (ausente)."""
    match = re.match(r"\d[\d.]*", value.strip())
    return float(match.group()) if match else None


def parse_baggio_specs(html: str):
    """Lista de ícones sem rótulo: quartos, quartos (repetido), banheiros, vagas, área."""
    anchor = html.find("Aluguel + Taxas")
    match = BAGGIO_SPECS_PATTERN.search(html, max(anchor, 0))
    if match is None:
        return []
    return [parse_spec(value) for value in BAGGIO_SPEC_ITEM_PATTERN.findall(match.group(1))]


def parse_baggio_detail(html: str):
    text = html_to_text(html)
    specs = parse_baggio_specs(html) + [None] * 5
    quartos, _, banheiros, vagas, listed_area = specs[:5]

    detail = {
        "area": (
            parse_measure(text, "Área Privativa")
            or parse_measure(text, "Área Total")
            or listed_area
        ),
        "quartos": quartos,
        "banheiros": banheiros,
        "vagas": vagas,
        "aluguel": parse_money(text, "Aluguel"),
        "condominio_iptu": (parse_money(text, "Condomínio") or 0) + (parse_money(text, "IPTU") or 0),
        "custo_total": parse_money(text, r"Aluguel \+ Taxas"),
    }

    address = BAGGIO_ADDRESS_PATTERN.search(text)
    if address is not None:
        endereco, cep, bairro, cidade = address.groups()
        digits = re.sub(r"\D", "", cep)
        detail.update({
            "endereco": unescape(endereco),
            "cep": f"{digits[:5]}-{digits[5:]}" if len(digits) == 8 else None,
            "bairro": unescape(bairro).strip().title(),
            "cidade": unescape(cidade).strip(),
        })
    return detail


def is_baggio_house(item: dict):
    """A taxonomia é a fonte da verdade do tipo; o filtro da API aceita combinações mistas."""
    classes = set(item.get("class_list") or [])
    return "imovel_tipo-casa" in classes and not classes & BAGGIO_EXCLUDED_TYPES


def fetch_json(session: requests.Session, url: str, params: dict, request_config: dict):
    for attempt in range(1, request_config["max_retries"] + 1):
        try:
            response = session.get(url, params=params, timeout=request_config["timeout_seconds"])
        except requests.RequestException:
            response = None

        if response is not None and response.status_code == 200:
            return response.json()

        time.sleep(request_config["delay_seconds"] * attempt)

    return None


def collect_pages(session: requests.Session, site: dict, request_config: dict, parser):
    """Percorre as URLs montadas pelo config e aplica o parser em cada página."""
    urls = build_urls(site)
    print(f"Páginas a visitar: {len(urls)}\n")

    rows = []
    for position, url in enumerate(urls, 1):
        html = fetch_html(session, url, request_config)
        if html is None:
            print(f"  [{position}/{len(urls)}] falhou: {url}")
            continue

        page_rows = parser(html, site["name"])
        rows.extend(page_rows)
        print(f"  [{position}/{len(urls)}] {len(page_rows):3d} anúncios | {url}")

        time.sleep(request_config["delay_seconds"])

    return rows


def collect_chavesnamao(session: requests.Session, site: dict, request_config: dict):
    return collect_pages(session, site, request_config, parse_chavesnamao)


def collect_quintoandar(session: requests.Session, site: dict, request_config: dict):
    return collect_pages(session, site, request_config, parse_quintoandar)


def collect_baggio(session: requests.Session, site: dict, request_config: dict):
    """Duas etapas: a REST API do WordPress lista os imóveis, a página traz os valores."""
    api_url = f"{site['base_url']}/wp-json/wp/v2/imovel"
    params = dict(site["api_filters"], per_page=100, _fields=BAGGIO_API_FIELDS)

    items = []
    page = 1
    while True:
        batch = fetch_json(session, api_url, dict(params, page=page), request_config)
        if not batch:
            break
        items.extend(batch)
        page += 1
        time.sleep(request_config["delay_seconds"])

    houses = [item for item in items if is_baggio_house(item)]
    print(f"Imóveis listados na API: {len(items)} ({len(houses)} casas residenciais)\n")

    city = site.get("city")
    collected_at = date.today().isoformat()
    rows = []
    for position, item in enumerate(houses, 1):
        html = fetch_html(session, item["link"], request_config)
        if html is None:
            print(f"  [{position}/{len(houses)}] falhou: {item['link']}")
            continue

        row = {
            "id": item["meta"].get("codigo") or item["id"],
            "fonte": site["name"],
            "url": item["link"],
            "data_coleta": collected_at,
            "tipo": "Casa",
        }
        row.update(parse_baggio_detail(html))

        # A Baggio atende a região metropolitana; a cidade só aparece na página do imóvel
        if city and row.get("cidade") != city:
            print(f"  [{position}/{len(houses)}] fora de {city}: {row.get('cidade')}")
            time.sleep(request_config["delay_seconds"])
            continue

        rows.append(row)
        print(f"  [{position}/{len(houses)}] {row.get('bairro')} | {row.get('aluguel')}")

        time.sleep(request_config["delay_seconds"])

    return rows


COLLECTORS = {
    "quintoandar": collect_quintoandar,
    "chavesnamao": collect_chavesnamao,
    "baggio": collect_baggio,
}


def scrape_site(session: requests.Session, site: dict, request_config: dict):
    collector = COLLECTORS.get(site["collector"])
    if collector is None:
        print(f"Coletor '{site['collector']}' não implementado.")
        print(f"Coletores disponíveis: {', '.join(COLLECTORS.keys())}")
        return []

    print(f"\n{'='*60}")
    print(f"Coletando de: {site['name']}")
    print(f"{'='*60}\n")

    return collector(session, site, request_config)


def build_dataframe(rows: list):
    df = pd.DataFrame(rows).reindex(columns=COLUMNS)
    return df.drop_duplicates(subset=["fonte", "id"]).reset_index(drop=True)


def scrape():
    config = load_json(BASE_DIR / "config.json")
    request_config = config["request"]
    session = build_session(request_config)

    rows = []
    for site in config["sites"]:
        if not site["enabled"]:
            continue
        rows.extend(scrape_site(session, site, request_config))

    df = build_dataframe(rows)

    output_path = (BASE_DIR / config["output"]).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    # Cópia datada: cada coleta vira um ponto para a comparação temporal de preços
    history_dir = (BASE_DIR / config["history_dir"]).resolve()
    history_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(history_dir / f"{output_path.stem}_{date.today().isoformat()}.csv", index=False)

    summary = {
        "coletado_em": date.today().isoformat(),
        "anuncios_brutos": len(rows),
        "anuncios_unicos": len(df),
        "anuncios_por_fonte": df["fonte"].value_counts().to_dict(),
        "anuncios_por_tipo": df["tipo"].value_counts().to_dict(),
        "anuncios_por_bairro": df["bairro"].value_counts().to_dict(),
        "com_coordenadas": int(df["latitude"].notna().sum()),
    }
    summary_path = output_path.with_suffix(".resumo.json")
    save_json(summary_path, summary)

    print(f"\n{'='*60}")
    print(f"\nRESULTADO DA COLETA:\n")
    print(f"  Anúncios brutos: {summary['anuncios_brutos']}")
    print(f"  Anúncios únicos: {summary['anuncios_unicos']}")
    print(f"  Com coordenadas: {summary['com_coordenadas']}")
    print(f"  Bairros distintos: {len(summary['anuncios_por_bairro'])}")
    print(f"{'='*60}\n")
    print(f"Base salva em: {output_path}")
    print(f"Resumo salvo em: {summary_path}")


def main():
    scrape()


if __name__ == "__main__":
    main()
