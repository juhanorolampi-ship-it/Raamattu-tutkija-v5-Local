# run_full_diagnostics.py (Versio 21.2 - Korjattu import-virhe)
import logging
import math
import re
import time
from collections import defaultdict

from logic import (
    ARVIOINTI_MALLI_ENSISIJAINEN,
    ARVIOINTI_MALLI_VARAMALLI,
    TIMANTTIJAE_MINIMI_MAARA,
    arvioi_tulokset,
    ehdota_uutta_strategiaa,
    etsi_merkityksen_mukaan,
    lataa_resurssit,
    suorita_tarkennushaku,
)

# --- MÄÄRITYKSET ---
SYOTE_TIEDOSTO = 'syote.txt'
TULOS_LOKI = 'diagnostiikka_raportti_final.txt'
LOPULLISTEN_HAKUTULOSTEN_MAARA = 15
LAAJAN_HAUN_MAARA = 75
ARVIOINTI_ERAN_KOKO = 10

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
    osiot = re.split(r'\n(?=[\d]+\.[\d\.]*\s)', sisalto)

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
            haku = f"{otsikko}: {kuvaus.replace('\n', ' ')}"
            hakulauseet[osio_nro] = haku
            otsikot[osio_nro] = otsikko

    return hakulauseet, otsikot


def suorita_diagnostiikka():
    """Ajaa koko diagnostiikkaprosessin, sisältäen dynaamisen parannusalgoritmin."""
    total_start_time = time.time()
    log_header("RAAMATTU-TUTKIJA - DIAGNOSTIIKKA (Dynaaminen Parannusalgoritmi)")

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
        otsikko = otsikot.get(osio_nro, "")
        log_header(f"Käsitellään osio {i+1}/{len(sorted_osiot)}: {otsikko}")

        # VAIHE 1: ALKUPERÄINEN LAAJA ETSINTÄ
        logging.info(f"Vaihe 1: Suoritetaan laaja haku (haetaan {LAAJAN_HAUN_MAARA} jaetta)...")
        alkuperaiset_ehdokkaat = etsi_merkityksen_mukaan(haku, otsikko, top_k=LAAJAN_HAUN_MAARA)
        logging.info(f"Löytyi {len(alkuperaiset_ehdokkaat)} ehdokasjaetta.")

        if not alkuperaiset_ehdokkaat:
            logging.warning("Laaja haku ei tuottanut tuloksia. Siirrytään seuraavaan osioon.")
            continue

        # VAIHE 2: ALKUPERÄINEN ARVIOINTI PÄÄMALLILLA (ERISSÄ)
        logging.info(f"Vaihe 2: Arvioidaan {len(alkuperaiset_ehdokkaat)} ehdokasta päämallilla...")
        kaikki_arviot = []
        erien_maara = math.ceil(len(alkuperaiset_ehdokkaat) / ARVIOINTI_ERAN_KOKO)

        for j in range(erien_maara):
            alku, loppu = j * ARVIOINTI_ERAN_KOKO, (j + 1) * ARVIOINTI_ERAN_KOKO
            era_ehdokkaat = alkuperaiset_ehdokkaat[alku:loppu]
            logging.info(f"  - Arvioidaan erä {j+1}/{erien_maara}...")
            arvio = arvioi_tulokset(haku, era_ehdokkaat)

            if "virhe" in arvio or len(arvio.get("jae_arviot", [])) != len(era_ehdokkaat):
                logging.warning("Päämalli epäonnistui, yritetään varamallia erälle...")
                arvio = arvioi_tulokset(haku, era_ehdokkaat, malli_nimi=ARVIOINTI_MALLI_VARAMALLI)
                if "virhe" in arvio or len(arvio.get("jae_arviot", [])) != len(era_ehdokkaat):
                    logging.error(f"KRIITTINEN: Myös varamalli epäonnistui erälle {j+1}. Erä ohitetaan.")
                    continue

            kaikki_arviot.extend(arvio.get("jae_arviot", []))

        logging.info(f"Alkuperäinen arviointi valmis. Saatiin {len(kaikki_arviot)} jaearviota.")
        if not kaikki_arviot:
            logging.error("Arviointi epäonnistui kokonaan. Siirrytään seuraavaan osioon.")
            continue

        # VAIHE 3: TULOSTEN KOKOAMINEN JA KESKIARVON LASKENTA
        jarjestetyt_arviot = sorted(kaikki_arviot, key=lambda x: x.get('arvosana', 0), reverse=True)
        parhaat_arviot = jarjestetyt_arviot[:LOPULLISTEN_HAKUTULOSTEN_MAARA]

        valid_scores = [a.get('arvosana') for a in parhaat_arviot if a.get('arvosana') is not None]
        alkuperainen_keskiarvo = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
        logging.info(f"Vaihe 3: Valittu {len(parhaat_arviot)} parasta jaetta. Alkuperäinen laatuarvio: {alkuperainen_keskiarvo:.2f}/10")

        final_tulokset = []
        for arvio_item in parhaat_arviot:
            vastaava_jae = next((item for item in alkuperaiset_ehdokkaat if item['viite'] == arvio_item.get('viite')), None)
            if vastaava_jae:
                vastaava_jae.update(arvio_item)
                final_tulokset.append(vastaava_jae)

        # VAIHE 4: DYNAAMINEN PARANNUSALGORITMI
        log_header(f"KÄYNNISTETÄÄN DYNAAMINEN PARANNUSALGORITMI (OSIO {osio_nro})")
        dynaaminen_raja_arvo = alkuperainen_keskiarvo
        ydinjakeet = [t for t in final_tulokset if t.get('arvosana', 0) >= dynaaminen_raja_arvo]

        if len(ydinjakeet) >= TIMANTTIJAE_MINIMI_MAARA:
            logging.info(f"TILA A: Ydinjakeita löytyi {len(ydinjakeet)} kpl (väh. {TIMANTTIJAE_MINIMI_MAARA}). Suoritetaan tarkennushaku.")
            logging.info(f"Dynaaminen raja-arvo tälle osiolle: {dynaaminen_raja_arvo:.2f}/10")

            heikot_jakeet = sorted([t for t in final_tulokset if t.get('arvosana', 0) < dynaaminen_raja_arvo], key=lambda x: x.get('arvosana', 0))
            haettava_maara = max(10, min(50, len(heikot_jakeet) * 3))

            logging.info(f"Korvattavia heikkoja jakeita: {len(heikot_jakeet)}. Haetaan {haettava_maara} uutta ehdokasta.")
            vanhat_viitteet = {t['viite'] for t in final_tulokset}
            uudet_ehdokkaat = suorita_tarkennushaku(ydinjakeet, vanhat_viitteet, haettava_maara)

            if uudet_ehdokkaat:
                logging.info(f"Tarkennushaku löysi {len(uudet_ehdokkaat)} uutta, uniikkia jaetta. Arvioidaan ne...")
                uudet_arvioidut = arvioi_tulokset(haku, uudet_ehdokkaat).get("jae_arviot", [])

                for jae in uudet_ehdokkaat:
                    vastaava_arvio = next((a for a in uudet_arvioidut if a.get('viite') == jae['viite']), None)
                    if vastaava_arvio:
                        jae.update(vastaava_arvio)

                uudet_parhaat = sorted([j for j in uudet_ehdokkaat if 'arvosana' in j], key=lambda x: x.get('arvosana', 0), reverse=True)

                logging.info("--- LAADUNVALVONTA JA ÄLYKÄS KORVAAMINEN ---")
                korvaus_laskuri = 0
                for i_korv in range(len(heikot_jakeet)):
                    if i_korv < len(uudet_parhaat):
                        vanha_jae = heikot_jakeet[i_korv]
                        uusi_jae = uudet_parhaat[i_korv]
                        if uusi_jae.get('arvosana', 0) > vanha_jae.get('arvosana', 0):
                            log_msg = (
                                f"  -> KORVATAAN: '{vanha_jae['viite']}' ({vanha_jae.get('arvosana'):.2f}/10) ==> '{uusi_jae['viite']}' ({uusi_jae.get('arvosana'):.2f}/10)\n"
                                f"     - Vanha perustelu: {vanha_jae.get('perustelu', 'N/A')}\n"
                                f"     + Uusi perustelu:  {uusi_jae.get('perustelu', 'N/A')}"
                            )
                            logging.info(log_msg)
                            for idx, item in enumerate(final_tulokset):
                                if item['viite'] == vanha_jae['viite']:
                                    final_tulokset[idx] = uusi_jae
                                    break
                            korvaus_laskuri += 1
                        else:
                            logging.info(f"  -> SÄILYTETÄÄN: '{vanha_jae['viite']}' ({vanha_jae.get('arvosana'):.2f}/10), koska uusi ehdokas '{uusi_jae['viite']}' ({uusi_jae.get('arvosana'):.2f}/10) ei ollut parempi.")
                logging.info(f"Laadunvalvonta valmis. {korvaus_laskuri} jaetta korvattu.")
            else:
                logging.warning("Tarkennushaku ei löytänyt uusia jakeita.")
        else:
            logging.warning(f"TILA B: Ydinjakeita löytyi vain {len(ydinjakeet)} kpl (väh. {TIMANTTIJAE_MINIMI_MAARA}). Siirrytään strategian parannukseen.")
            arvio_obj = {"kokonaisarvosana": alkuperainen_keskiarvo, "jae_arviot": parhaat_arviot}
            ehdotus = ehdota_uutta_strategiaa(haku, arvio_obj)

            if "virhe" not in ehdotus and ehdotus.get("selite"):
                logging.info(f"Luotu uusi strategia: {ehdotus.get('selite')}")
                uudet_strategiat = {s.lower(): ehdotus.get("selite") for s in ehdotus.get("avainsanat", [])}
                heikot_lkm = len([t for t in final_tulokset if t.get('arvosana', 0) < dynaaminen_raja_arvo])
                if heikot_lkm > 0:
                    logging.info(f"Haetaan {heikot_lkm} korvaajaa uudella strategialla...")
                    paikkaushaku = etsi_merkityksen_mukaan(haku, otsikko, top_k=heikot_lkm, custom_strategiat=uudet_strategiat)
                    if paikkaushaku:
                        final_tulokset_hyvat = [t for t in final_tulokset if t.get('arvosana', 0) >= dynaaminen_raja_arvo]
                        final_tulokset = final_tulokset_hyvat + paikkaushaku
                        logging.info(f"{len(paikkaushaku)} jaetta korvattu.")
                else:
                    logging.info("Ei heikkoja jakeita korvattavaksi.")
            else:
                logging.error("Strategian luonti epäonnistui.")

        valid_scores_final = [a.get('arvosana') for a in final_tulokset if a.get('arvosana') is not None]
        lopputulos_keskiarvo = sum(valid_scores_final) / len(valid_scores_final) if valid_scores_final else 0.0
        lopulliset_arvosanat[osio_nro] = f"{lopputulos_keskiarvo:.2f}"

        logging.info(f"Parannusprosessin jälkeen lopullinen laatuarvio: {lopputulos_keskiarvo:.2f}/10")
        if lopputulos_keskiarvo > alkuperainen_keskiarvo:
            logging.info(f"LAADUNPARANNUS ONNISTUI! ({alkuperainen_keskiarvo:.2f} -> {lopputulos_keskiarvo:.2f}) ✅")
        else:
            logging.info("Laatu ei parantunut tai pysyi samana.")

        logging.info(f"--- Lopulliset valitut jakeet ja niiden arviot (Päämalli: {ARVIOINTI_MALLI_ENSISIJAINEN}) ---")
        for jae_arvio in sorted(final_tulokset, key=lambda x: x.get('arvosana', 0), reverse=True):
            logging.info(f"  - {jae_arvio.get('viite')}: {jae_arvio.get('arvosana', 0):.2f}/10 ({jae_arvio.get('perustelu')})")

        jae_kartta_tuloksille[osio_nro] = [f"- {t['viite']}: \"{t['teksti']}\"" for t in sorted(final_tulokset, key=lambda x: x.get('arvosana', 0), reverse=True)]

    # YHTEENVETO-OSA
    total_end_time = time.time()
    log_header("DIAGNOSTIIKAN YHTEENVETO")
    aika = total_end_time - total_start_time
    logging.info(f"Koko diagnostiikan ajo kesti: {aika:.2f} sekuntia.")
    valid_scores_float = [float(s) for s in lopulliset_arvosanat.values() if s != 'N/A']
    if valid_scores_float:
        keskiarvo_total = sum(valid_scores_float) / len(valid_scores_float)
        logging.info(f"PÄÄMALLIN antama lopullinen keskiarvo tulosten laadulle: {keskiarvo_total:.2f}/10")

    log_header("YKSITYISKOHTAINEN JAEJAOTTELU (LOPULLISET TULOKSET)")
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