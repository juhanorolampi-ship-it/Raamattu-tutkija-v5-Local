# run_full_diagnostics.py (Versio 8.0 - Autonominen korjaussilmukka)
import logging
import time
import re
from collections import defaultdict

# Tuodaan kaikki tarvittavat funktiot ja sanakirjat
from logic import (
    etsi_merkityksen_mukaan,
    lataa_resurssit,
    arvioi_tulokset,
    ehdota_uutta_strategiaa,
    STRATEGIA_SANAKIRJA,
    STRATEGIA_SIEMENJAE_KARTTA
)

# --- MÄÄRITYKSET ---
SYOTE_TIEDOSTO = 'syote.txt'
TULOS_LOKI = 'diagnostiikka_raportti_autonominen.txt'
HAKUTULOSTEN_MAARA_PER_TEEMA = 15
# Asetetaan raja-arvo, jonka alittuessa korjausprosessi käynnistyy.
# <= 7 tarkoittaa, että arvosanat 1-7 käynnistävät sen.
STRATEGIAEHDOTUS_RAJA = 7

# --- LOKITUSMÄÄRITYKSET ---
logger = logging.getLogger()
logger.setLevel(logging.INFO)

if logger.hasHandlers():
    logger.handlers.clear()

file_handler = logging.FileHandler(TULOS_LOKI, encoding='utf-8', mode='w')
file_handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(file_handler)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S'
))
logger.addHandler(stream_handler)


def log_header(text):
    """Tulostaa siistin otsikon lokitiedostoon."""
    line = "=" * 80
    logger.info("\n%s\n%s\n%s\n", line, text.center(80), line)


def lue_syote_tiedosto(tiedostopolku):
    """Lukee ja jäsentää syötetiedoston vankasti."""
    try:
        with open(tiedostopolku, 'r', encoding='utf-8') as f:
            sisalto = f.read()
    except FileNotFoundError:
        logging.error(f"Syötetiedostoa '{tiedostopolku}' ei löytynyt.")
        return None, None

    sisallysluettelo_match = re.search(
        r"Sisällysluettelo:(.*?)(?=\n\d\.|\Z)", sisalto, re.DOTALL
    )
    sisallysluettelo = sisallysluettelo_match.group(
        1).strip() if sisallysluettelo_match else ""

    hakulauseet = {}
    osiot = re.split(r'\n(?=\d\.\s)', sisalto)

    for osio_teksti in osiot:
        osio_teksti = osio_teksti.strip()
        if not osio_teksti:
            continue

        rivit = osio_teksti.split('\n', 1)
        otsikko = rivit[0].strip()
        kuvaus = rivit[1].strip() if len(rivit) > 1 else ""

        osio_match = re.match(r"^([\d\.]+)", otsikko)
        if osio_match:
            osio_nro = osio_match.group(1).strip('.')
            haku = f"{otsikko}: {kuvaus.replace(' ', ' ')}"
            hakulauseet[osio_nro] = haku

    return hakulauseet, sisallysluettelo


def suorita_diagnostiikka():
    """Ajaa koko diagnostiikkaprosessin sisältäen autonomisen korjaussilmukan."""
    total_start_time = time.time()
    log_header("RAAMATTU-TUTKIJA v6 - DIAGNOSTIIKKA (Autonominen korjaussilmukka)")

    logging.info("Esiladataan kaikki resurssit...")
    lataa_resurssit()
    logging.info("Resurssit ladattu.")

    hakulauseet, _ = lue_syote_tiedosto(SYOTE_TIEDOSTO)
    if not hakulauseet:
        return

    logging.info(f"Löytyi {len(hakulauseet)} osiota käsiteltäväksi.")

    # Luodaan väliaikaiset kopiot strategioista tätä ajokertaa varten
    temp_strategiat = STRATEGIA_SANAKIRJA.copy()
    temp_siemenjakeet = STRATEGIA_SIEMENJAE_KARTTA.copy()

    jae_kartta_tuloksille = defaultdict(list)
    lopulliset_arvosanat = {}

    sorted_osiot = sorted(
        hakulauseet.items(),
        key=lambda item: [int(p) for p in item[0].split('.')]
    )

    for i, (osio_nro, haku) in enumerate(sorted_osiot):
        log_header(f"Käsitellään osio {i+1}/{len(sorted_osiot)}: {osio_nro}")

        # --- VAIHE 1: ALKUPERÄINEN HAKU JA ARVIOINTI ---
        tulokset = etsi_merkityksen_mukaan(
            haku,
            top_k=HAKUTULOSTEN_MAARA_PER_TEEMA,
            custom_strategiat=temp_strategiat,
            custom_siemenjakeet=temp_siemenjakeet
        )
        arvio = arvioi_tulokset(haku, tulokset)

        logger.info("--- 1. Alkuperäinen laadunarviointi ---")
        if arvio["arvosana"] is not None:
            logger.info(f"AI Arvosana: {arvio['arvosana']}/10")
        else:
            logger.info("AI Arvosana: Ei saatu.")
        logger.info(f"AI Perustelu: {arvio['perustelu']}")
        logger.info("-" * 40)

        # --- VAIHE 2: AUTONOMINEN KORJAUSSILMUKKA ---
        if arvio["arvosana"] is not None and arvio["arvosana"] <= STRATEGIAEHDOTUS_RAJA:
            logger.info("Laatuarvio oli heikko. Käynnistetään autonominen korjausprosessi...")

            # 2a: Luo uusi strategia
            ehdotus = ehdota_uutta_strategiaa(haku, tulokset, arvio)

            if "virhe" in ehdotus:
                logger.error(f"Strategiaehdotus epäonnistui: {ehdotus['virhe']}")
                lopulliset_arvosanat[osio_nro] = arvio["arvosana"]
            else:
                avainsanat = ehdotus.get('avainsanat', [])
                selite = ehdotus.get('selite', '')
                logger.info("--- 2a. Uusi strategia luotu ---")
                logger.info(f"AI:n ehdottamat avainsanat: {avainsanat}")
                logger.info(f"AI:n ehdottama selite: {selite}")

                # 2b: Lisää uusi strategia väliaikaisesti muistiin
                for sana in avainsanat:
                    temp_strategiat[sana.lower()] = selite
                logger.info("Uusi strategia lisätty väliaikaisesti tähän ajoon.")
                logger.info("-" * 40)

                # 2c: Aja haku uudelleen uudella strategialla
                logger.info("--- 2c. Suoritetaan haku uudelleen korjatulla strategialla... ---")
                uudet_tulokset = etsi_merkityksen_mukaan(
                    haku,
                    top_k=HAKUTULOSTEN_MAARA_PER_TEEMA,
                    custom_strategiat=temp_strategiat,
                    custom_siemenjakeet=temp_siemenjakeet
                )

                # 2d: Arvioi uudet tulokset ja vertaa
                uusi_arvio = arvioi_tulokset(haku, uudet_tulokset)

                logger.info("--- 2d. Korjatun haun laadunarviointi ---")
                logger.info(f"Alkuperäinen arvosana: {arvio['arvosana']}/10")
                if uusi_arvio["arvosana"] is not None:
                    logger.info(f"UUSI arvosana: {uusi_arvio['arvosana']}/10")
                    if uusi_arvio['arvosana'] > arvio['arvosana']:
                        logger.info("TULOS: Laatu parani merkittävästi! ✅")
                        # Käytetään jatkossa paranneltuja tuloksia
                        tulokset = uudet_tulokset
                        arvio = uusi_arvio
                    else:
                        logger.warning("TULOS: Laatu ei parantunut. ⚠️")
                else:
                    logger.error("Uutta arvosanaa ei saatu.")
                logger.info(f"Uusi perustelu: {uusi_arvio['perustelu']}")
                logger.info("-" * 40)

        # Tallennetaan osion lopullinen arvosana ja jakeet
        lopulliset_arvosanat[osio_nro] = arvio.get("arvosana")
        if tulokset:
            for tulos in tulokset:
                jae_viite_teksti = f"- {tulos['viite']}: \"{tulos['teksti']}\""
                jae_kartta_tuloksille[osio_nro].append(jae_viite_teksti)

    total_end_time = time.time()
    log_header("DIAGNOSTIIKAN YHTEENVETO")
    logging.info(f"Koko diagnostiikan ajo kesti: {(total_end_time - total_start_time):.2f} sekuntia.")

    valid_scores = [s for s in lopulliset_arvosanat.values() if s is not None]
    if valid_scores:
        keskiarvo = sum(valid_scores) / len(valid_scores)
        logging.info(f"AI:n antama lopullinen keskiarvo tulosten laadulle: {keskiarvo:.2f}/10")

    log_header("YKSITYISKOHTAINEN JAEJAOTTELU")
    for osio_nro_sorted, haku_sorted in sorted_osiot:
        logger.info(f"\n--- {osio_nro_sorted} {haku_sorted.split(':')[0]} ---\n")
        logger.info(f"(Lopullinen arvosana tälle osiolle: {lopulliset_arvosanat.get(osio_nro_sorted, 'N/A')}/10)")
        if osio_nro_sorted in jae_kartta_tuloksille:
            for jae in jae_kartta_tuloksille[osio_nro_sorted]:
                logger.info(jae)
        else:
            logger.info("Ei jakeita tähän osioon.")

    logging.info("\nDiagnostiikka valmis.")


if __name__ == '__main__':
    suorita_diagnostiikka()