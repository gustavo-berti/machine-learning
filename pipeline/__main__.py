"""Executa a pipeline completa: python -m pipeline [--scrape] [--no-geocode]"""
import argparse
import subprocess
import sys

from pipeline import config, eda, history, modeling, preprocess


def main():
    parser = argparse.ArgumentParser(description="Pré-processamento, EDA, modelagem e histórico.")
    parser.add_argument("--scrape", action="store_true", help="roda o scraper antes (leva ~15 min)")
    parser.add_argument("--no-geocode", action="store_true", help="não consulta o Nominatim (usa só cache e centroides)")
    args = parser.parse_args()

    if args.scrape:
        print("== Etapa 1: coleta ==")
        subprocess.run([sys.executable, str(config.ROOT / "scraper" / "scraper.py")], check=True)

    print("== Etapa 2: pré-processamento ==")
    preprocess.run(geocode=not args.no_geocode)
    print("\n== Etapa 3: análise exploratória ==")
    eda.run()
    print("\n== Etapa 4: modelagem ==")
    modeling.run()
    print("\n== Histórico de coletas ==")
    history.run()
    print("\nPronto. Abra o app com: ./venv/bin/streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
