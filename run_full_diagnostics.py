# run_full_diagnostics.py (Versio 16.0 - Lopullinen, vankka logiikka)
import logging
import re
import time
from collections import defaultdict
import math

from logic import (
    arvioi_tulokset,
    etsi_merkityksen_mukaan,
    lataa_resurssit,
)

# --- MÄÄRITYKSET ---
SYOTE_TIEDOSTO = 'syote.txt'
TULOS_LOKI = 'diagnostiikka_raportti_hybridi.txt'
LOPULLISTEN_HAKUTULOSTEN_MAARA = 15
LAAJAN_HAUN_MAARA = 75
ARVIOINTI_ERAN_KOKO = 10 # Pilkotaan arviointi pienempiin osiin

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

    hakulauseet = {}
    otsikot = {}
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
            haku = f"{otsikko}: {kuvaus.replace(r'\n', ' ')}"
            hakulauseet[osio_nro] = haku
            otsikot[osio_nro] = otsikko
            
    return hakulauseet, otsikot


def suorita_diagnostiikka():
    """Ajaa koko diagnostiikkaprosessin käyttäen uutta kaksivaiheista logiikkaa."""
    total_start_time = time.time()
    log_header("RAAMATTU-TUTKIJA v16 - DIAGNOSTIIKKA (Laaja haku + Pilkottu karsinta)")

    logging.info("Esiladataan kaikki resurssit...")
    lataa_resurssit()
    logging.info("Resurssit ladattu.")

    hakulauseet, otsikot = lue_syote_tiedosto(SYOTE_TIEDOSTO)
    if not hakulauseet:
        return

    logging.info(f"Löytyi {len(hakulauseet)} osiota käsiteltäväksi.")

    jae_kartta_tuloksille = defaultdict(list)
    lopulliset_arvosanat = {}

    sorted_osiot = sorted(
        hakulauseet.items(),
        key=lambda item: [int(p) for p in item[0].split('.')]
    )

    for i, (osio_nro, haku) in enumerate(sorted_osiot):
        log_header(f"Käsitellään osio {i+1}/{len(sorted_osiot)}: {otsikot.get(osio_nro, '')}")

        # VAIHE 1: LAAJA ETSINTÄ
        logging.info(f"Vaihe 1: Suoritetaan laaja haku (haetaan {LAAJAN_HAUN_MAARA} jaetta)...")
        ehdokkaat = etsi_merkityksen_mukaan(haku, top_k=LAAJAN_HAUN_MAARA)
        logging.info(f"Löytyi {len(ehdokkaat)} ehdokasjaetta.")

        if not ehdokkaat:
            logging.warning("Laaja haku ei tuottanut tuloksia. Siirrytään seuraavaan osioon.")
            continue

        # VAIHE 2: PILKOTTU KARSINTA
        logging.info(f"Vaihe 2: Arvioidaan {len(ehdokkaat)} ehdokasta {ARVIOINTI_ERAN_KOKO} jakeen erissä...")
        kaikki_arviot = []
        erien_maara = math.ceil(len(ehdokkaat) / ARVIOINTI_ERAN_KOKO)

        for j in range(erien_maara):
            alku = j * ARVIOINTI_ERAN_KOKO
            loppu = alku + ARVIOINTI_ERAN_KOKO
            era_ehdokkaat = ehdokkaat[alku:loppu]
            
            logging.info(f"  - Arvioidaan erä {j+1}/{erien_maara}...")
            arvio = arvioi_tulokset(haku, era_ehdokkaat)
            era_arviot = arvio.get("jae_arviot", [])

            if len(era_arviot) != len(era_ehdokkaat):
                logging.error(f"  -> VIRHE: Malli palautti {len(era_arviot)} arviota, "
                              f"vaikka sille annettiin {len(era_ehdokkaat)} jaetta. Erä ohitetaan.")
                continue

            kaikki_arviot.extend(era_arviot)
        
        logging.info(f"Arviointi valmis. Saatiin yhteensä {len(kaikki_arviot)} jaearviota.")

        if not kaikki_arviot:
            logging.error("Arviointi epäonnistui, yhtään jaearviota ei saatu.")
            continue
            
        # VAIHE 3: LOPULLISTEN TULOSTEN KOKOAMINEN
        jarjestetyt_arviot = sorted(
            kaikki_arviot,
            key=lambda x: x.get('arvosana', 0),
            reverse=True
        )

        parhaat_arviot = jarjestetyt_arviot[:LOPULLISTEN_HAKUTULOSTEN_MAARA]

        final_tulokset = []
        for par_arvio in parhaat_arviot:
            vastaava_jae = next((item for item in ehdokkaat if item['viite'] == par_arvio['viite']), None)
            if vastaava_jae:
                final_tulokset.append(vastaava_jae)
        
        valid_scores = [a.get('arvosana') for a in parhaat_arviot if a.get('arvosana') is not None]
        keskiarvo = sum(valid_scores) / len(valid_scores) if valid_scores else 0
        lopulliset_arvosanat[osio_nro] = f"{keskiarvo:.2f}"

        logging.info(f"Vaihe 3: Valittu {len(final_tulokset)} parasta jaetta. Lopullinen laatuarvio: {keskiarvo:.2f}/10")
        logging.info("--- Valitut jakeet ja niiden arviot ---")
        for jae_arvio in parhaat_arviot:
             logging.info(f"  - {jae_arvio.get('viite')}: {jae_arvio.get('arvosana')}/10 ({jae_arvio.get('perustelu')})")

        if final_tulokset:
            jae_kartta_tuloksille[osio_nro] = [
                f"- {t['viite']}: \"{t['teksti']}\"" for t in final_tulokset
            ]

    total_end_time = time.time()
    log_header("DIAGNOSTIIKAN YHTEENVETO")
    aika = total_end_time - total_start_time
    logging.info(f"Koko diagnostiikan ajo kesti: {aika:.2f} sekuntia.")

    valid_scores_float = [float(s) for s in lopulliset_arvosanat.values() if isinstance(s, str) and s != 'N/A']
    if valid_scores_float:
        keskiarvo_total = sum(valid_scores_float) / len(valid_scores_float)
        logging.info("AI:n antama lopullinen keskiarvo tulosten laadulle: "
                     f"{keskiarvo_total:.2f}/10")

    log_header("YKSITYISKOHTAINEN JAEJAOTTELU")
    for osio_nro_sorted, haku_sorted in sorted_osiot:
        otsikko_sorted = otsikot.get(osio_nro_sorted, haku_sorted.split(':')[0])
        logger.info(f"\n--- {osio_nro_sorted} {otsikko_sorted} ---\n")
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