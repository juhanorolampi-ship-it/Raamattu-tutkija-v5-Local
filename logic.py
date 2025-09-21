# logic.py (Versio 14.0 - Manuaalinen kartoitus)
import json
import logging
import re
import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder

# --- VAKIOASETUKSET ---
PAAINDESKI_TIEDOSTO = "D:/Python_AI/Raamattu-tutkija-data/raamattu_vektori_indeksi.faiss"
PAAKARTTA_TIEDOSTO = "D:/Python_AI/Raamattu-tutkija-data/raamattu_viite_kartta.json"
RAAMATTU_TIEDOSTO = "D:/Python_AI/Raamattu-tutkija-data/bible.json"
EMBEDDING_MALLI = "TurkuNLP/sbert-cased-finnish-paraphrase"
CROSS_ENCODER_MALLI = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- STRATEGIAKERROS ---
STRATEGIA_SANAKIRJA = {
    "jännite": (
        "Hae Raamatusta kohtia, jotka kuvaavat rakentavaa erimielisyyttä, "
        "toisiaan täydentäviä rooleja tai sitä, miten erilaisuus johtaa "
        "hengelliseen kasvuun ja terveen jännitteen kautta parempaan lopputulokseen."
    ),
    "tasapaino": (
        "Etsi kohtia, jotka käsittelevät tasapainoa, harmoniaa tai oikeaa "
        "suhdetta kahden eri asian, kuten työn ja levon, tai totuuden "
        "ja rakkauden, välillä."
    ),
    "profeetta": (
        "Etsi kohtia, jotka kuvaavat profeetallisen ja pastoraalisen tai "
        "opettavan roolin välistä dynamiikkaa, yhteistyötä tai jännitettä "
        "seurakunnassa."
    ),
    "paimen": (
        "Etsi kohtia, jotka kuvaavat profeetallisen ja pastoraalisen tai "
        "opettavan roolin välistä dynamiikkaa, yhteistyötä tai jännitettä "
        "seurakunnassa."
    ),
    "pappi": (
        "Etsi kohtia, jotka kuvaavat profeetallisen ja pastoraalisen tai "
        "opettavan roolin välistä dynamiikkaa, yhteistyötä tai jännitettä "
        "seurakunnassa."
    ),
    "koetinkivi": (
        "Etsi jakeita, jotka käsittelevät luonteen testaamista ja koettelemista "
        "erityisissä olosuhteissa, kuten vastoinkäymisissä, menestyksessä, "
        "kritiikin alla tai näkymättömyydessä."
    ),
    "testi": (
        "Etsi jakeita, jotka käsittelevät luonteen testaamista ja koettelemista "
        "erityisissä olosuhteissa, kuten vastoinkäymisissä, menestyksessä, "
        "kritiikin alla tai näkymättömyydessä."
    ),
    "näkymättömyys": (
        "Hae jakeita, jotka käsittelevät palvelemista ilman ihmisten "
        "näkemystä, kiitosta tai tunnustusta, keskittyen Jumalan palkkioon "
        "ja oikeaan sydämen asenteeseen."
    ),
    "kritiikki": (
        "Etsi jakeita, jotka opastavat, miten suhtautua oikeutetusti "
        "tai epäoikeutetusti saatuun kritiikkiin, arvosteluun tai "
        "nuhteeseen säilyttäen nöyrän ja opetuslapseen sopivan sydämen."
    ),
    "intohimo": (
        "Hae jakeita, jotka kuvaavat sydämen paloa, innostusta, "
        "syvää mielenkiintoa tai Jumalan antamaa tahtoa ja paloa "
        "tiettyä asiaa tai tehtävää kohtaan."
    ),
    "kyvyt": (
        "Etsi kohtia, jotka käsittelevät luontaisia, synnynnäisiä "
        "taitoja, lahjakkuutta ja osaamista, jotka Jumala on ihmiselle antanut "
        "ja joita voidaan käyttää hänen kunniakseen."
    )
}

# UUSI MANUAALINEN KARTTA STRATEGIOIDEN JA SIEMENJAKEIDEN VÄLILLÄ
STRATEGIA_SIEMENJAE_KARTTA = {
    "jännite": "Room. 12:4-5",
    "tasapaino": "Saarn. 3:1",
    "profeetta": "Ef. 4:11-12",
    "paimen": "Ef. 4:11-12",
    "pappi": "Ef. 4:11-12",
    "koetinkivi": "Jaak. 1:2-4",
    "testi": "Jaak. 1:2-4",
    "näkymättömyys": "Fil. 2:3-4",
    "kritiikki": "Miika 6:8",
    "intohimo": "Room. 12:1-2",
    "kyvyt": "1. Piet. 4:10",
}


# --- LOKITUKSEN ALUSTUS ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    datefmt="%H:%M:%S",
)


@st.cache_resource
def lataa_resurssit():
    """Lataa kaikki tarvittavat resurssit ja pitää ne muistissa."""
    logging.info("Ladataan hakumallit, indeksi ja datatiedostot muistiin...")
    try:
        model = SentenceTransformer(EMBEDDING_MALLI)
        cross_encoder = CrossEncoder(CROSS_ENCODER_MALLI)
        paaindeksi = faiss.read_index(PAAINDESKI_TIEDOSTO)
        with open(PAAKARTTA_TIEDOSTO, "r", encoding="utf-8") as f:
            paakartta = json.load(f)
        with open(RAAMATTU_TIEDOSTO, "r", encoding="utf-8") as f:
            raamattu_data = json.load(f)

        jae_haku_kartta = {}
        for book_obj in raamattu_data["book"].values():
            kirjan_nimi = book_obj.get("info", {}).get("name")
            luvut_obj = book_obj.get("chapter")
            if not kirjan_nimi or not isinstance(luvut_obj, dict):
                continue
            for luku_nro, luku_obj in luvut_obj.items():
                jakeet_obj = luku_obj.get("verse")
                if not isinstance(jakeet_obj, dict):
                    continue
                for jae_nro, jae_obj in jakeet_obj.items():
                    teksti = jae_obj.get("text", "").strip()
                    if teksti:
                        viite = f"{kirjan_nimi} {luku_nro}:{jae_nro}"
                        jae_haku_kartta[viite] = teksti

        logging.info("Kaikki resurssit ladattu onnistuneesti.")
        return model, cross_encoder, paaindeksi, paakartta, jae_haku_kartta
    except Exception as e:
        logging.error(f"Kriittinen virhe resurssien alustuksessa: {e}")
        st.error(f"Resurssien lataus epäonnistui: {e}")
        return None, None, None, None, None


def poimi_raamatunviitteet(teksti: str) -> list[str]:
    """Etsii ja poimii tekstistä raamatunviitteitä."""
    pattern = r'((?:[1-3]\.\s)?[A-ZÅÄÖa-zåäö]+\.?\s\d+:\d+(?:-\d+)?)'
    return re.findall(pattern, teksti)


def hae_jakeet_viitteella(viite_str: str, jae_haku_kartta: dict) -> list[dict]:
    """Hakee jaejoukon tekstistä poimitun viitteen perusteella."""
    viite_pattern = re.compile(
        r'((?:[1-3]\.\s)?'
        r'[A-ZÅÄÖa-zåäö]+\.?)'
        r'\s(\d+):(\d+)(?:-(\d+))?'
    )
    match = viite_pattern.match(viite_str)
    if not match:
        return []

    kirja, luku, alku, loppu = match.groups()
    kirja = kirja.strip().lower().replace('.', '').replace(' ', '')
    luku_nro_str = str(int(luku))
    alku_jae = int(alku)
    loppu_jae = int(loppu) if loppu else alku_jae

    loytyneet = []
    for avain, teksti in jae_haku_kartta.items():
        avain_norm = avain.lower().replace('.', '').replace(' ', '')
        if kirja in avain_norm:
            try:
                luku_ja_jae_osa = avain.split(' ')[-1]
                haun_luku_str, haun_jae_str = luku_ja_jae_osa.split(':')
                haun_jae = int(haun_jae_str)
                if (haun_luku_str == luku_nro_str and
                        alku_jae <= haun_jae <= loppu_jae):
                    loytyneet.append({"viite": avain, "teksti": teksti})
            except (ValueError, IndexError):
                continue
    return sorted(loytyneet, key=lambda x: int(x['viite'].split(':')[-1]))


def etsi_merkityksen_mukaan(kysely: str, top_k: int = 15) -> list[dict]:
    """
    Etsii Raamatusta käyttäen manuaalisesti kartoitettua hybridihakua.
    """
    resurssit = lataa_resurssit()
    if not all(resurssit):
        logging.error("Haku epäonnistui, koska resursseja ei voitu ladata.")
        return []

    model, cross_encoder, paaindeksi, paakartta, jae_haku_kartta = resurssit

    viite_str_lista = poimi_raamatunviitteet(kysely)
    pakolliset_jakeet = []
    loytyneet_viitteet = set()
    for viite_str in viite_str_lista:
        jakeet = hae_jakeet_viitteella(viite_str, jae_haku_kartta)
        for jae in jakeet:
            if jae["viite"] not in loytyneet_viitteet:
                pakolliset_jakeet.append(jae)
                loytyneet_viitteet.add(jae["viite"])
    logging.info(
        f"Löydettiin {len(viite_str_lista)} viitettä, jotka vastasivat "
        f"{len(pakolliset_jakeet)} uniikkia jaetta."
    )

    alyhaun_tulokset = []
    laajennettu_kysely = kysely
    pien_kysely = kysely.lower()
    strategia_loytyi = False

    # VAIHE 1: Tarkista, aktivoituuko jokin strategia
    for avainsana, selite in STRATEGIA_SANAKIRJA.items():
        if avainsana in pien_kysely:
            strategia_loytyi = True
            logging.info(f"Strategia aktivoitu avainsanalla '{avainsana}'.")

            # VAIHE 2: Hae manuaalisesti kartoitettu siemenjae
            siemenjae_viite = STRATEGIA_SIEMENJAE_KARTTA.get(avainsana)
            if siemenjae_viite:
                siemenjae_teksti = jae_haku_kartta.get(siemenjae_viite, "")
                logging.info(f"Manuaalisesti valittu siemenjae: {siemenjae_viite}")
                # VAIHE 3: Rakenna "superkysely"
                laajennettu_kysely = (
                    f"Aihe on: '{kysely}'. Teeman selitys on: '{selite}'. "
                    f"Tärkeä esimerkki aiheesta on jae '{siemenjae_viite}', "
                    f"joka kuuluu: '{siemenjae_teksti}'."
                )
                logging.info("Rakennettu 'superkysely' strategian ja siemenjakeen pohjalta.")
            else:
                # Varasuunnitelma, jos kartasta ei löydy avainsanaa
                laajennettu_kysely = f"{selite}. Alkuperäinen aihe on: {kysely}"
                logging.info("Ei siemenjaetta määritelty, käytetään vain strategiaa.")
            break

    if not strategia_loytyi:
        logging.info("Strategiaa ei löytynyt. Käytetään perinteistä semanttista hakua.")
        laajennettu_kysely = kysely

    # VAIHE 4: Suorita haku ja uudelleenjärjestys
    if top_k > 0:
        if top_k <= 10:
            kerroin = 10
        elif 11 <= top_k <= 20:
            kerroin = 9
        elif 21 <= top_k <= 40:
            kerroin = 8
        elif 41 <= top_k <= 60:
            kerroin = 7
        elif 61 <= top_k <= 80:
            kerroin = 6
        else:
            kerroin = 5

        haettava_maara = min(top_k * kerroin, paaindeksi.ntotal)

        if haettava_maara > 0:
            kysely_vektori = model.encode([laajennettu_kysely])
            _, indeksit = paaindeksi.search(
                np.array(kysely_vektori, dtype=np.float32), haettava_maara
            )
            ehdokkaat = []
            for idx in indeksit[0]:
                viite = paakartta.get(str(idx))
                if viite and viite not in loytyneet_viitteet:
                    ehdokkaat.append({
                        "viite": viite,
                        "teksti": jae_haku_kartta.get(viite, "")
                    })

            if ehdokkaat:
                parit = [[laajennettu_kysely, j["teksti"]] for j in ehdokkaat]
                pisteet = cross_encoder.predict(parit, show_progress_bar=False)
                for i, j in enumerate(ehdokkaat):
                    j['pisteet'] = pisteet[i]
                jarjestetyt = sorted(
                    ehdokkaat, key=lambda x: x['pisteet'], reverse=True
                )
                alyhaun_tulokset = jarjestetyt[:top_k]

    lopulliset_tulokset = pakolliset_jakeet + alyhaun_tulokset
    logging.info(
        f"Yhdistetty {len(pakolliset_jakeet)} pakollista jaetta ja "
        f"{len(alyhaun_tulokset)} älyhaun tulosta. "
        f"Yhteensä {len(lopulliset_tulokset)} jaetta."
    )
    return lopulliset_tulokset