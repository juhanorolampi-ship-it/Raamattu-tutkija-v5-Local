# run_full_diagnostics.py (Versio 10.0 - Kontekstuaalinen ja tavoitehakuinen oppiminen)
import logging
import time
import re
from collections import defaultdict

from logic import (
    etsi_merkityksen_mukaan,
    lataa_resurssit,
    arvioi_tulokset,
    ehdota_uutta_strategiaa,
    tallenna_uusi_strategia,
    luo_kontekstisidonnainen_avainsana,
    STRATEGIA_SANAKIRJA,
    STRATEGIA_SIEMENJAE_KARTTA
)

# --- MÄÄRITYKSET ---
SYOTE_TIEDOSTO = 'syote.txt'
TULOS_LOKI = 'diagnostiikka_raportti_oppiva_v2.txt'
HAKUTULOSTEN_MAARA_PER_TEEMA = 15
TAVOITEARVOSANA = 8
MAKSIMI_YRITYKSET_PER_OSIO = 3

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
        return None

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
    return hakulauseet


def suorita_diagnostiikka():
    """Ajaa koko diagnostiikkaprosessin sisältäen itseoppivan silmukan."""
    total_start_time = time.time()
    log_header("RAAMATTU-TUTKIJA v8 - DIAGNOSTIIKKA (Tavoitehakuinen oppiminen)")

    logging.info("Esiladataan kaikki resurssit...")
    lataa_resurssit()
    logging.info("Resurssit ladattu.")

    hakulauseet = lue_syote_tiedosto(SYOTE_TIEDOSTO)
    if not hakulauseet:
        return

    logging.info(f"Löytyi {len(hakulauseet)} osiota käsiteltäväksi.")
    
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

        tulokset = etsi_merkityksen_mukaan(
            haku,
            top_k=HAKUTULOSTEN_MAARA_PER_TEEMA,
            custom_strategiat=temp_strategiat,
            custom_siemenjakeet=temp_siemenjakeet
        )
        arvio = arvioi_tulokset(haku, tulokset)

        logger.info("--- 1. Alkuperäinen laadunarviointi ---")
        logger.info(f"AI Arvosana: {arvio.get('arvosana', 'N/A')}/10")
        logger.info(f"AI Perustelu: {arvio.get('perustelu', 'Ei perustelua.')}")
        logger.info("-" * 40)

        if arvio.get("arvosana") is not None and arvio["arvosana"] < TAVOITEARVOSANA:
            yritykset = 0
            edellinen_ehdotus = None
            
            while arvio.get("arvosana") is not None and arvio["arvosana"] < TAVOITEARVOSANA and yritykset < MAKSIMI_YRITYKSET_PER_OSIO:
                yritykset += 1
                log_header(f"Parannusyritys {yritykset}/{MAKSIMI_YRITYKSET_PER_OSIO} osiolle {osio_nro}")

                ehdotus = ehdota_uutta_strategiaa(haku, tulokset, arvio, edellinen_ehdotus)
                edellinen_ehdotus = ehdotus.copy()

                if "virhe" in ehdotus:
                    logging.error("Strategiaehdotus epäonnistui, silmukka keskeytetään.")
                    break
                
                avainsanat = ehdotus.get('avainsanat', [])
                selite = ehdotus.get('selite', '')

                for sana in avainsanat:
                    temp_strategiat[sana.lower()] = selite
                
                uudet_tulokset = etsi_merkityksen_mukaan(
                    haku, top_k=HAKUTULOSTEN_MAARA_PER_TEEMA, custom_strategiat=temp_strategiat
                )
                uusi_arvio = arvioi_tulokset(haku, uudet_tulokset)

                logger.info("--- Parannetun haun laadunarviointi ---")
                logger.info(f"Edellinen arvosana: {arvio.get('arvosana', 'N/A')}/10")
                logger.info(f"UUSI arvosana: {uusi_arvio.get('arvosana', 'N/A')}/10")
                
                if uusi_arvio.get("arvosana") is not None and arvio.get("arvosana") is not None:
                    if uusi_arvio['arvosana'] > arvio['arvosana']:
                        logging.info("TULOS: Laatu parani! ✅")
                        tulokset, arvio = uudet_tulokset, uusi_arvio
                        
                        uniikit_avainsanat = []
                        for sana in avainsanat:
                            if sana.lower() in STRATEGIA_SANAKIRJA:
                                logging.warning(f"Avainsana '{sana}' on jo olemassa. Luodaan kontekstisidonnainen versio.")
                                uusi_sana = luo_kontekstisidonnainen_avainsana(sana, selite)
                                uniikit_avainsanat.append(uusi_sana)
                                logging.info(f"Luotu uusi kontekstisidonnainen avainsana: '{uusi_sana}'")
                            else:
                                uniikit_avainsanat.append(sana)
                        
                        tallenna_uusi_strategia(uniikit_avainsanat, selite)
                    else:
                        logging.warning("TULOS: Laatu ei parantunut tällä yrityksellä. ⚠️")
                        arvio = uusi_arvio
                else:
                    logging.error("Arvosanaa ei saatu, silmukka keskeytetään.")
                    break
        
        lopulliset_arvosanat[osio_nro] = arvio.get("arvosana")
        if tulokset:
            jae_kartta_tuloksille[osio_nro] = [f"- {t['viite']}: \"{t['teksti']}\"" for t in tulokset]

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