# run_full_diagnostics.py (Versio 12.1 - Yksityiskohtainen iterointiloki, PEP-8)
import logging
import re
import time
from collections import defaultdict

from logic import (
    STRATEGIA_SANAKIRJA,
    STRATEGIA_SIEMENJAE_KARTTA,
    arvioi_tulokset,
    ehdota_uutta_strategiaa,
    etsi_merkityksen_mukaan,
    lataa_resurssit,
    luo_kontekstisidonnainen_avainsana,
    tallenna_uusi_strategia
)

# --- MÄÄRITYKSET ---
SYOTE_TIEDOSTO = 'syote.txt'
TULOS_LOKI = 'diagnostiikka_raportti_oppiva_v4_FINAALI.txt'
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
    log_header("RAAMATTU-TUTKIJA v9 - DIAGNOSTIIKKA (Analyyttinen oppiminen)")

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

        logging.info("--- 1. Alkuperäinen laadunarviointi ---")
        logging.info(f"AI Arvosana: {arvio.get('arvosana', 'N/A')}/10")
        logging.info(f"AI Perustelu: {arvio.get('perustelu', 'Ei.')}")
        logging.info("-" * 40)

        arvosana_num = arvio.get("arvosana")
        if arvosana_num is not None and arvosana_num < TAVOITEARVOSANA:
            yritykset = 0
            edellinen_ehdotus = None

            while (arvio.get("arvosana") is not None and
                   arvio["arvosana"] < TAVOITEARVOSANA and
                   yritykset < MAKSIMI_YRITYKSET_PER_OSIO):
                yritykset += 1
                log_header(
                    f"Parannusyritys {yritykset}/{MAKSIMI_YRITYKSET_PER_OSIO} "
                    f"osiolle {osio_nro}"
                )

                ehdotus = ehdota_uutta_strategiaa(haku, tulokset, arvio,
                                                   edellinen_ehdotus)
                edellinen_ehdotus = ehdotus.copy()

                if "virhe" in ehdotus:
                    logging.error("Strategiaehdotus epäonnistui, "
                                  "silmukka keskeytetään.")
                    break

                avainsanat = ehdotus.get('avainsanat', [])
                selite = ehdotus.get('selite', '')
                
                # === UUSI LOKITUS ALKAA ===
                logger.info("--- Luotu strategia ---")
                logger.info(f"  Avainsanat: {avainsanat}")
                logger.info(f"  Selite: {selite}")
                logger.info("---------------------")
                # === UUSI LOKITUS PÄÄTTYY ===
                
                if not avainsanat or not selite:
                    logging.error("Strategiaehdotus oli tyhjä, "
                                  "silmukka keskeytetään.")
                    break

                for sana in avainsanat:
                    temp_strategiat[sana.lower()] = selite

                uudet_tulokset = etsi_merkityksen_mukaan(
                    haku, top_k=HAKUTULOSTEN_MAARA_PER_TEEMA,
                    custom_strategiat=temp_strategiat
                )
                uusi_arvio = arvioi_tulokset(haku, uudet_tulokset)

                logging.info("--- Parannetun haun laadunarviointi ---")
                logging.info(f"Edellinen arvosana: "
                             f"{arvio.get('arvosana', 'N/A')}/10")
                logging.info(f"UUSI arvosana: "
                             f"{uusi_arvio.get('arvosana', 'N/A')}/10")

                vanha_arvosana = arvio.get("arvosana")
                uusi_arvosana = uusi_arvio.get("arvosana")

                if uusi_arvosana is not None and vanha_arvosana is not None:
                    if uusi_arvosana > vanha_arvosana:
                        logging.info("TULOS: Laatu parani! ✅")
                        tulokset, arvio = uudet_tulokset, uusi_arvio

                        uniikit_sanat = []
                        for sana in avainsanat:
                            if sana.lower() in STRATEGIA_SANAKIRJA:
                                msg = (f"Avainsana '{sana}' on jo olemassa. "
                                       "Luodaan kontekstisidonnainen versio.")
                                logging.warning(msg)
                                uusi = luo_kontekstisidonnainen_avainsana(sana,
                                                                          selite)
                                uniikit_sanat.append(uusi)
                                logging.info("Luotu uusi kontekstisidonnainen "
                                             f"avainsana: '{uusi}'")
                            else:
                                uniikit_sanat.append(sana)

                        tallenna_uusi_strategia(uniikit_sanat, selite)
                    else:
                        logging.warning("TULOS: Laatu ei parantunut. ⚠️")
                        arvio = uusi_arvio
                else:
                    logging.error("Arvosanaa ei saatu, "
                                  "silmukka keskeytetään.")
                    break

        lopulliset_arvosanat[osio_nro] = arvio.get("arvosana")
        if tulokset:
            jae_kartta_tuloksille[osio_nro] = [
                f"- {t['viite']}: \"{t['teksti']}\"" for t in tulokset
            ]

    total_end_time = time.time()
    log_header("DIAGNOSTIIKAN YHTEENVETO")
    aika = total_end_time - total_start_time
    logging.info(f"Koko diagnostiikan ajo kesti: {aika:.2f} sekuntia.")

    valid_scores = [s for s in lopulliset_arvosanat.values() if s is not None]
    if valid_scores:
        keskiarvo = sum(valid_scores) / len(valid_scores)
        logging.info("AI:n antama lopullinen keskiarvo tulosten laadulle: "
                     f"{keskiarvo:.2f}/10")

    log_header("YKSITYISKOHTAINEN JAEJAOTTELU")
    for osio_nro_sorted, haku_sorted in sorted_osiot:
        logger.info(f"\n--- {osio_nro_sorted} "
                     f"{haku_sorted.split(':')[0]} ---\n")
        arvosana = lopulliset_arvosanat.get(osio_nro_sorted, 'N/A')
        logger.info(f"(Lopullinen arvosana tälle osiolle: {arvosana}/10)")
        if osio_nro_sorted in jae_kartta_tuloksille:
            for jae in jae_kartta_tuloksille[osio_nro_sorted]:
                logger.info(jae)
        else:
            logger.info("Ei jakeita tähän osioon.")

    logging.info("\nDiagnostiikka valmis.")


if __name__ == '__main__':
    suorita_diagnostiikka()