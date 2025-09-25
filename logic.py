# logic.py (Versio 37.1 - Optimoitu arviointi ja GPU-korjaukset)
import json
import logging
import pprint
import re
import time

import faiss
import numpy as np
import ollama
import streamlit as st
from sentence_transformers import CrossEncoder, SentenceTransformer

# --- VAKIOASETUKSET ---
LOGIC_TIEDOSTOPOLKU = "logic.py"
PAAINDESKI_TIEDOSTO = "D:/Python_AI/Raamattu-tutkija-data/raamattu_indeksi_e5_large.faiss"
PAAKARTTA_TIEDOSTO = "D:/Python_AI/Raamattu-tutkija-data/raamattu_kartta_e5_large.json"
RAAMATTU_TIEDOSTO = "D:/Python_AI/Raamattu-tutkija-data/bible.json"
RAAMATTU_SANAKIRJA_TIEDOSTO = "D:/Python_AI/Raamattu-tutkija-data/bible_dictionary.json"
EMBEDDING_MALLI = "intfloat/multilingual-e5-large"
CROSS_ENCODER_MALLI = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TIMANTTIJAE_MINIMI_MAARA = 3

# --- MALLIMÄÄRITYKSET ---
# Nämä ovat oletusarvoja, jotka voidaan yliajaa käyttöliittymässä
ARVIOINTI_MALLI_ENSISIJAINEN = "llama3.1:8b"
ARVIOINTI_MALLI_VARAMALLI = "gemma:7b"
ASIANTUNTIJA_MALLI = "llama3.1:8b"

# --- STRATEGIAKERROS JA KARTTA ---
STRATEGIA_SANAKIRJA = {
    'intohimo': 'Hae jakeita, jotka kuvaavat sydämen paloa, innostusta, syvää '
                'mielenkiintoa tai Jumalan antamaa tahtoa ja paloa tiettyä asiaa tai '
                'tehtävää kohtaan.',
    'jännite': 'Hae Raamatusta kohtia, jotka kuvaavat rakentavaa erimielisyyttä, '
               'toisiaan täydentäviä rooleja tai sitä, miten erilaisuus johtaa '
               'hengelliseen kasvuun ja terveen jännitteen kautta parempaan '
               'lopputulokseen.',
    'koetinkivi': 'Etsi jakeita, jotka käsittelevät luonteen testaamista ja '
                  'koettelemista erityisissä olosuhteissa, kuten vastoinkymisissä, '
                  'menestyksessä, kritiikin alla tai näkymättömyydessä.',
    'kritiikki': 'Etsi jakeita, jotka opastavat, miten suhtautua oikeutetusti tai '
                 'epäoikeutetusti saatuun kritiikkiin, arvosteluun tai nuhteeseen '
                 'säilyttäen nöyrän ja opetuslapseen sopivan sydämen.',
    'kyvyt': 'Etsi kohtia, jotka käsittelevät luontaisia, synnynnäisiä taitoja, '
             'lahjakkuutta ja osaamista, jotka Jumala on ihmiselle antanut ja joita '
             'voidaan käyttää hänen kunniakseen.',
    'näkymättömyys': 'Hae jakeita, jotka käsittelevät palvelemista ilman ihmisten '
                     'näkemystä, kiitosta tai tunnustusta, keskittyen Jumalan '
                     'palkkioon ja oikeaan sydämen asenteeseen.',
    'tasapaino': 'Etsi kohtia, jotka käsittelevät tasapainoa, harmoniaa tai oikeaa '
                 'suhdetta kahden eri asian, kuten työn ja levon, tai totuuden ja '
                 'rakkauden, välillä.',
}
STRATEGIA_SIEMENJAE_KARTTA = {
    'intohimo': 'Room. 12:1-2',
    'jännite': 'Room. 12:4-5',
    'koetinkivi': 'Jaak. 1:2-4',
    'kritiikki': 'Miika 6:8',
    'kyvyt': '1. Piet. 4:10',
    'näkymättömyys': 'Fil. 2:3-4',
    'tasapaino': 'Saarn. 3:1',
}

# --- LOKITUKSEN ALUSTUS ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    datefmt="%H:%M:%S",
)


# --- RESURSSIEN LATAUS ---
@st.cache_resource
def lataa_resurssit():
    logging.info("Ladataan hakumallit, indeksi ja datatiedostot muistiin...")
    try:
        # TÄRKEÄ KORJAUS: Poistettu device='cpu', jotta GPU:ta käytetään
        model = SentenceTransformer(EMBEDDING_MALLI)
        cross_encoder = CrossEncoder(CROSS_ENCODER_MALLI)
        paaindeksi = faiss.read_index(PAAINDESKI_TIEDOSTO)
        with open(PAAKARTTA_TIEDOSTO, "r", encoding="utf-8") as f:
            paakartta = json.load(f)
        with open(RAAMATTU_TIEDOSTO, "r", encoding="utf-8") as f:
            raamattu_data = json.load(f)
        with open(RAAMATTU_SANAKIRJA_TIEDOSTO, "r", encoding="utf-8") as f:
            raamattu_sanasto_lista = json.load(f)
        raamattu_sanasto = set(raamattu_sanasto_lista)
        jae_haku_kartta = {}
        for book_obj in raamattu_data["book"].values():
            kirjan_nimi = book_obj.get("info", {}).get("name")
            for luku_nro, luku_obj in book_obj.get("chapter", {}).items():
                for jae_nro, jae_obj in luku_obj.get("verse", {}).items():
                    teksti = jae_obj.get("text", "").strip()
                    if teksti and kirjan_nimi:
                        viite = f"{kirjan_nimi} {luku_nro}:{jae_nro}"
                        jae_haku_kartta[viite] = teksti
        logging.info("Kaikki resurssit ladattu onnistuneesti.")
        return model, cross_encoder, paaindeksi, paakartta, jae_haku_kartta, raamattu_sanasto
    except Exception as e:
        logging.error(f"Kriittinen virhe resurssien alustuksessa: {e}")
        st.error(f"Resurssien lataus epäonnistui: {e}")
        return None, None, None, None, None, None


# --- APUFUNKTIOT ---
def poimi_raamatunviitteet(teksti: str) -> list[str]:
    pattern = r'((?:[1-3]\.\s)?[A-ZÅÄÖa-zåäö]+\.?\s\d+:\d+(?:-\d+)?)'
    return re.findall(pattern, teksti)

def hae_jakeet_viitteella(viite_str: str, jae_haku_kartta: dict) -> list[dict]:
    viite_pattern = re.compile(
        r'((?:[1-3]\.\s)?[A-ZÅÄÖa-zåäö]+\.?)'
        r'\s(\d+):(\d+)(?:-(\d+))?'
    )
    match = viite_pattern.match(viite_str)
    if not match:
        return []
    kirja, luku, alku, loppu = match.groups()
    kirja = kirja.strip().lower().replace('.', '').replace(' ', '')
    luku_nro_str = str(int(luku))
    alku_jae, loppu_jae = int(alku), int(loppu) if loppu else int(alku)
    loytyneet = []
    for avain, teksti in jae_haku_kartta.items():
        avain_norm = avain.lower().replace('.', '').replace(' ', '')
        if kirja in avain_norm:
            try:
                luku_jae_osa = avain.split(' ')[-1]
                haun_luku_str, haun_jae_str = luku_jae_osa.split(':')
                if haun_luku_str == luku_nro_str and alku_jae <= int(haun_jae_str) <= loppu_jae:
                    loytyneet.append({"viite": avain, "teksti": teksti})
            except (ValueError, IndexError):
                continue
    return sorted(loytyneet, key=lambda x: int(x['viite'].split(':')[-1]))


# --- VANKKA TEKOÄLYKUTSU ITSEKORJAUKSELLA ---
def suorita_varmistettu_json_kutsu(mallit: list, kehote: str, max_yritykset: int = 2) -> tuple[dict, str]:
    """Suorittaa tekoälykutsun, yrittää jäsentää JSON-vastauksen ja käyttää itsekorjausta virhetilanteessa."""
    vastaus_teksti = ""
    for malli in mallit:
        logging.info(f"Käytetään mallia: {malli}")
        for yritys in range(max_yritykset):
            try:
                messages = [{'role': 'user', 'content': kehote.strip()}]
                logging.info(f"Lähetetään JSON-pyyntö mallille {malli} (Yritys {yritys + 1}/{max_yritykset})...")
                response = ollama.chat(model=malli, messages=messages, format='json')
                vastaus_teksti = response['message']['content']
                data = json.loads(vastaus_teksti)
                logging.info("Vastaus jäsennelty onnistuneesti.")
                return data, malli
            except (json.JSONDecodeError, KeyError) as e:
                logging.warning(f"Virhe mallin {malli} kanssa (yritys {yritys + 1}): {e}. Käynnistetään itsekorjaus...")
                korjaus_kehote = (
                    f"Edellinen vastauksesi ei ollut kelvollinen JSON-objekti. Virhe oli: '{e}'.\n"
                    f"Tässä on virheellinen teksti:\n---\n{vastaus_teksti}\n---\n"
                    f"Korjaa se. Palauta AINOASTAAN korjattu, validi JSON-objekti ilman mitään muuta tekstiä tai selityksiä."
                )
                try:
                    korjaus_messages = [{'role': 'user', 'content': korjaus_kehote}]
                    logging.info(f"Lähetetään itsekorjauspyyntö mallille {malli}...")
                    korjaus_response = ollama.chat(model=malli, messages=korjaus_messages, format='json')
                    korjattu_teksti = korjaus_response['message']['content']
                    data = json.loads(korjattu_teksti)
                    logging.info("Itsekorjaus onnistui ja vastaus jäsenneltiin.")
                    return data, malli
                except (json.JSONDecodeError, KeyError) as e_korjaus:
                    logging.error(f"Itsekorjaus epäonnistui mallilla {malli}. Virhe: {e_korjaus}.")
                    if yritys < max_yritykset - 1:
                         logging.info("Yritetään alkuperäistä pyyntöä uudelleen...")
                    continue # Yritetään uudelleen alusta
    logging.error(f"Kaikki mallit ({mallit}) epäonnistuivat. Palautetaan virhe.")
    return {"virhe": "JSON-vastausta ei saatu malleilta."}, "Tuntematon"


# --- PÄÄFUNKTIOT ---
def onko_strategia_relevantti(kysely: str, selite: str) -> bool:
    kehote = (f"ROOLI: Olet looginen päättelijä.\n"
              f"TEHTÄVÄ: Arvioi, onko strategia hyödyllinen hakukyselyn tarkentamiseen.\n"
              f"- Kysely: \"{kysely}\"\n"
              f"- Strategia: \"{selite}\"\n"
              f"VASTAA AINOASTAAN JSON-MUODOSSA: {{\"sovellu\": true}} TAI {{\"sovellu\": false}}")
    logging.info("Suoritetaan strategian relevanssin esianalyysi...")
    data, _ = suorita_varmistettu_json_kutsu([ARVIOINTI_MALLI_ENSISIJAINEN, ARVIOINTI_MALLI_VARAMALLI], kehote)
    relevanssi = data.get("sovellu", False)
    logging.info(f"Esianalyysin tulos: Soveltuuko strategia? {'Kyllä' if relevanssi else 'Ei'}.")
    return relevanssi


def etsi_merkityksen_mukaan(kysely: str, otsikko: str, top_k: int = 15,
                          custom_strategiat: dict = None,
                          custom_siemenjakeet: dict = None,
                          valitut_tehostesanat: set = None) -> tuple[list[dict], set]:
    resurssit = lataa_resurssit()
    if not all(resurssit):
        return [], set()
    model_encoder, cross_encoder, paaindeksi, paakartta, jae_haku_kartta, raamattu_sanasto = resurssit

    viite_str_lista = poimi_raamatunviitteet(kysely)
    pakolliset_jakeet = []
    loytyneet_viitteet = set()
    for viite_str in viite_str_lista:
        jakeet = hae_jakeet_viitteella(viite_str, jae_haku_kartta)
        for jae in jakeet:
            if jae["viite"] not in loytyneet_viitteet:
                pakolliset_jakeet.append(jae)
                loytyneet_viitteet.add(jae["viite"])

    strategia_lahde = custom_strategiat if custom_strategiat is not None else STRATEGIA_SANAKIRJA
    siemenjae_lahde = custom_siemenjakeet if custom_siemenjakeet is not None else STRATEGIA_SIEMENJAE_KARTTA
    laajennettu_kysely = kysely
    pien_kysely = kysely.lower()
    tehostettavat_sanat = set()

    if valitut_tehostesanat is None:
        viite_pattern = r'((?:[1-3]\.\s)?[A-ZÅÄÖa-zåäö]+\.?\s\d+:\d+(?:-\d+)?)'
        puhdistettu_otsikko = re.sub(viite_pattern, '', otsikko)
        puhdistettu_otsikko = re.sub(r'[^\w\s]', '', puhdistettu_otsikko)
        stop_words = {
            'aiemmin', 'aika', 'aikaa', 'aikaan', 'aikaisemmin', 'aikaisin', 'aikana', 'aikoa', 'aina', 'ainakaan',
            'ainakin', 'ainoa', 'ainut', 'aivan', 'alas', 'alkuisin', 'alla', 'alle', 'alta', 'aluksi', 'antaa', 'asia', 'asti',
            'edes', 'edessä', 'edestä', 'ehkä', 'ei', 'eikä', 'eilen', 'eivät', 'eli', 'ellei', 'emme', 'en', 'enemmän', 'eniten',
            'ensin', 'entinen', 'entä', 'eri', 'erittäin', 'esimerkiksi', 'et', 'eteen', 'etenkin', 'ette', 'he', 'heidän',
            'hän', 'hänen', 'ihan', 'ilman', 'itse', 'itsensä', 'ja', 'jo', 'johon', 'joiden', 'joihin', 'joiksi', 'joilla',
            'joille', 'joilta', 'joina', 'joissa', 'joista', 'joita', 'joka', 'jokainen', 'jokin', 'joko', 'joku', 'jolla',
            'jolle', 'jolloin', 'jolta', 'jonka', 'jonkin', 'jonne', 'jos', 'joskus', 'jossa', 'josta', 'jota', 'jotain',
            'joten', 'jotka', 'jotta', 'juuri', 'jälkeen', 'kanssa', 'keiden', 'keihin', 'keillä', 'keille', 'keiltä',
            'keissä', 'keistä', 'keitä', 'kuka', 'kukaan', 'ken', 'kerran', 'kerta', 'kertaa', 'kesken', 'koska', 'koskaan',
            'kuin', 'kuinka', 'kuitenkaan', 'kuitenkin', 'kun', 'kuten', 'kyllä', 'kymmenen', 'lähellä', 'läheltä', 'lähes',
            'läpi', 'liian', 'lisäksi', 'me', 'meidän', 'melkein', 'melko', 'mihin', 'mikin', 'miksi', 'mikä', 'mikään',
            'mille', 'milloin', 'millä', 'miltä', 'minkä', 'minne', 'minun', 'minut', 'minä', 'missä', 'mistä', 'miten',
            'mitkä', 'mitä', 'mitään', 'mukaan', 'mutta', 'muu', 'muut', 'muuta', 'muutama', 'muuten', 'myös', 'myöskään',
            'ne', 'neljä', 'niiden', 'niihin', 'niiksi', 'niillä', 'niille', 'niiltä', 'niin', 'niinä', 'niissä', 'niistä',
            'niitä', 'noin', 'nopeasti', 'nyt', 'nämä', 'näiden', 'näihin', 'näiksi', 'näillä', 'näille', 'näiltä', 'näinä',
            'näissä', 'näistä', 'näitä', 'ole', 'olemme', 'olen', 'olet', 'olette', 'oleva', 'olevan', 'olevat', 'oli',
            'olimme', 'olin', 'olisi', 'olisimme', 'olisin', 'olisit', 'olisitte', 'olivat', 'olla', 'olleet', 'ollut',
            'oma', 'omat', 'on', 'ovat', 'paljon', 'paremmin', 'perusteella', 'pian', 'pitkin', 'pitäisi', 'pitää', 'pois',
            'puolesta', 'puolestaan', 'päälle', 'päin', 'saakka', 'sata', 'se', 'sekä', 'sen', 'siellä', 'sieltä', 'siihen',
            'siinä', 'siitä', 'sijaan', 'siksi', 'sillä', 'silloin', 'silti', 'sinne', 'sinun', 'sinut', 'sinä', 'sisällä',
            'siten', 'sitten', 'sitä', 'suoraan', 'suuri', 'suurin', 'tai', 'taas', 'takana', 'takia', 'tavalla', 'tavoin',
            'te', 'teidän', 'tietenkin', 'todella', 'toinen', 'toisaalla', 'toisaalta', 'toistaiseksi', 'toki', 'tosin',
            'tuhannen', 'tuhat', 'tulee', 'tulla', 'tämä', 'tämän', 'tänään', 'tässä', 'tästä', 'täysin', 'täytyy', 'täällä',
            'täältä', 'usea', 'useasti', 'usein', 'useita', 'uusi', 'uusia', 'uutta', 'vaan', 'vaikka', 'vain', 'varmasti',
            'varsinkin', 'varten', 'vasta', 'vastaan', 'verran', 'vielä', 'viime', 'viimeksi', 'voida', 'voimme', 'voin',
            'voit', 'voitte', 'voivat', 'vuoksi', 'vuosi', 'vuotta', 'vähemmän', 'vähän', 'yhtä', 'yhtään', 'yksi', 'yleensä',
            'yli', 'myöskin'
        }
        sanat = puhdistettu_otsikko.split()
        potentiaaliset_sanat = {s.lower() for s in sanat if s.lower() not in stop_words and len(s) > 2 and s and s[0].isupper()}
        tehostettavat_sanat = {sana for sana in potentiaaliset_sanat if sana in raamattu_sanasto}
    else:
        tehostettavat_sanat = valitut_tehostesanat
    if tehostettavat_sanat:
        logging.info(f"Avainsana-Tehostin aktivoitu sanoille: {tehostettavat_sanat}")
    else:
        logging.info("Avainsana-Tehostin: Ei relevantteja avainsanoja otsikossa.")
    for avainsana, selite in strategia_lahde.items():
        if avainsana in pien_kysely:
            if onko_strategia_relevantti(kysely, selite):
                logging.info(f"Strategia '{avainsana}' todettiin relevantiksi.")
                siemenjae_viite = siemenjae_lahde.get(avainsana)
                laajennettu_kysely = f"{kysely}. Hakua tarkentava strategia: {selite}"
                if siemenjae_viite and (siemenjae_teksti := jae_haku_kartta.get(siemenjae_viite, "")):
                    laajennettu_kysely = (f"{kysely}. Hakua ohjaava lisäkonteksti: {selite}. Teemaa havainnollistava jae: '{siemenjae_teksti}'.")
                break
            else:
                logging.info(f"Strategia '{avainsana}' hylättiin epärelevanttina.")

    alyhaun_tulokset = []
    if top_k > 0:
        alyhaun_koko = max(0, top_k - len(pakolliset_jakeet))
        if alyhaun_koko > 0:
            haettava_maara = min(alyhaun_koko * max(5, 11 - (alyhaun_koko // 10)), paaindeksi.ntotal)
            if haettava_maara > 0:
                kysely_vektori = model_encoder.encode([f"query: {laajennettu_kysely}"])
                _, indeksit = paaindeksi.search(np.array(kysely_vektori, dtype=np.float32), haettava_maara)
                ehdokkaat = [{'viite': v, 'teksti': jae_haku_kartta.get(v, "")} for i in indeksit[0] if (v := paakartta.get(str(i))) and v not in loytyneet_viitteet]
                if ehdokkaat:
                    parit = [[laajennettu_kysely, j["teksti"]] for j in ehdokkaat]
                    pisteet = cross_encoder.predict(parit, show_progress_bar=False)
                    for i, j in enumerate(ehdokkaat):
                        j['pisteet'] = pisteet[i]
                    if tehostettavat_sanat:
                        for jae in ehdokkaat:
                            found_words = {sana for sana in tehostettavat_sanat if re.search(r'\b' + re.escape(sana) + r'\b', jae['teksti'].lower())}
                            if found_words:
                                jae['pisteet'] += 2.0 * len(found_words)
                                logging.info(f"  -> Tehostettiin jaetta {jae['viite']} sanoilla: {found_words}")
                    alyhaun_tulokset = sorted(ehdokkaat, key=lambda x: x['pisteet'], reverse=True)[:alyhaun_koko]

    yhdistetyt_tulokset = pakolliset_jakeet + alyhaun_tulokset
    return yhdistetyt_tulokset, tehostettavat_sanat


def etsi_puhtaalla_haulla(kysely: str, top_k: int = 15) -> list[dict]:
    resurssit = lataa_resurssit()
    if not all(resurssit):
        return []
    model, _, paaindeksi, paakartta, jae_haku_kartta, _ = resurssit
    kysely_vektori = model.encode([f"query: {kysely}"])
    _, indeksit = paaindeksi.search(np.array(kysely_vektori, dtype=np.float32), top_k * 5)
    ehdokkaat = [{'viite': v, 'teksti': jae_haku_kartta.get(v, "")} for i in indeksit[0] if (v := paakartta.get(str(i)))]
    return ehdokkaat[:top_k]


def arvioi_tulokset(aihe: str, tulokset: list, malli_nimi: str = ARVIOINTI_MALLI_ENSISIJAINEN) -> dict:
    """
    Arvioi jakeet yksitellen, jotta vältetään suuri CPU-kuorma pitkän
    syötteen käsittelyssä. Sisältää aikalaskurin jokaiselle kutsulle.
    """
    if not tulokset:
        return {"kokonaisarvosana": 0.0, "jae_arviot": []}

    kaikki_jae_arviot = []
    yhteiskesto = 0
    logging.info(f"Aloitetaan {len(tulokset)} jakeen yksittäisarviointi...")

    for i, jae in enumerate(tulokset):
        start_time = time.time()  # AIKA ALKAA

        logging.info(f"  - Arvioidaan jae {i+1}/{len(tulokset)}: {jae['viite']}...")
        
        kehote = (
            f"ROOLI: Olet teologian asiantuntija. Tehtäväsi on arvioida, kuinka hyvin YKSI Raamatun jae vastaa annettua aihetta.\n"
            f"AIHE: \"{aihe}\"\n"
            f"ARVIOITAVA JAE:\n"
            f"- Viite: \"{jae['viite']}\"\n"
            f"- Teksti: \"{jae['teksti']}\"\n"
            f"VASTAA AINOASTAAN JSON-MUODOSSA. Anna arvosana (1.0-10.0) ja lyhyt, ytimekäs suomenkielinen perustelu.\n"
            f"ESIMERKKIVASTAUS: {{\"arvosana\": 8.5, \"perustelu\": \"Sopii hyvin, koska...\"}}\n"
            f"Sinun vastauksesi:"
        )
        
        data, malli = suorita_varmistettu_json_kutsu([malli_nimi, ARVIOINTI_MALLI_VARAMALLI], kehote)
        
        end_time = time.time()  # AIKA PÄÄTTYY
        kesto = end_time - start_time
        yhteiskesto += kesto
        logging.info(f"    -> Kesto: {kesto:.2f} sekuntia.")

        if "virhe" not in data and 'arvosana' in data and 'perustelu' in data:
            data['viite'] = jae['viite']
            data['mallin_nimi'] = malli
            kaikki_jae_arviot.append(data)
        else:
            logging.warning(f"Jae {jae['viite']} arviointi epäonnistui tai palautti virheellistä dataa.")

    if not kaikki_jae_arviot:
        return {"kokonaisarvosana": 0.0, "jae_arviot": []}

    valid_scores = [a.get('arvosana') for a in kaikki_jae_arviot if isinstance(a.get('arvosana'), (int, float))]
    kokonaisarvosana = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
    
    keskiarvo_kesto = yhteiskesto / len(tulokset) if tulokset else 0
    logging.info("Kaikki jakeet arvioitu onnistuneesti.")
    logging.info(f"Arviointien yhteiskesto: {yhteiskesto:.2f}s, keskimäärin {keskiarvo_kesto:.2f}s per jae.")
    
    return {
        "kokonaisarvosana": kokonaisarvosana,
        "kokonaisperustelu": f"Yhteenveto {len(kaikki_jae_arviot)} jakeen yksittäisarvioinnista.",
        "jae_arviot": kaikki_jae_arviot
    }


def suorita_tarkennushaku(ydinjakeet: list, vanhat_tulokset_viitteet: set, haettava_maara: int) -> list:
    resurssit = lataa_resurssit()
    if not all(resurssit):
        return []
    model, _, paaindeksi, paakartta, jae_haku_kartta, _ = resurssit
    if not ydinjakeet:
        return []
    ydinjakeiden_tekstit = [j['teksti'] for j in ydinjakeet]
    ydin_vektorit = model.encode(ydinjakeiden_tekstit)
    keskipiste_vektori = np.mean(ydin_vektorit, axis=0)
    _, indeksit = paaindeksi.search(np.array([keskipiste_vektori], dtype=np.float32), haettava_maara * 2)
    uudet_ehdokkaat = []
    for i in indeksit[0]:
        viite = paakartta.get(str(i))
        if viite and viite not in vanhat_tulokset_viitteet:
            uudet_ehdokkaat.append({'viite': viite, 'teksti': jae_haku_kartta.get(viite, "")})
    return uudet_ehdokkaat[:haettava_maara]


def ehdota_uutta_strategiaa(aihe: str, arvio: dict, edellinen_ehdotus: dict = None) -> dict:
    kokonaisperustelu = arvio.get('kokonaisperustelu', 'Ei perustelua.')
    analyysi_kehote = (
        f"ONGELMA-ANALYYSIN KEHOTE:\n"
        f"Ensimmäinen haku aiheelle \"{aihe}\" tuotti heikkoja tuloksia, joiden perustelu oli: \"{kokonaisperustelu}\".\n"
        f"TEHTÄVÄ:\n"
        f"1. Ehdota 2-4 TÄSMÄLLISTÄ JA KONKREETTISTA HAKUSANAA tai LYHYTTÄ HAKULAUSETTA.\n"
        f"2. Kirjoita lyhyt selite, joka perustuu näihin konkreettisiin hakusanoihin."
    )
    if edellinen_ehdotus:
        analyysi_kehote = (
            f"SYVEMPI ANALYYSI (ITSEKRITIIKKI):\n"
            f"Edellinen parannusyritys epäonnistui. Laatu ei parantunut.\n"
            f"- EDELLINEN STRATEGIA: {edellinen_ehdotus.get('selite', '')}\n"
            f"- TULOKSEN ARVIOINTI: Perustelu: \"{kokonaisperustelu}\"\n"
            f"UUSI TEHTÄVÄ:\n"
            f"1. Analysoi lyhyesti (1-2 lausetta), miksi edellinen strategia oli liian abstrakti tai epäonnistunut.\n"
            f"2. Ehdota 2-4 TÄSMÄLLISTÄ JA KONKREETTISTA HAKUSANAA tai LYHYTTÄ HAKULAUSETTA, jotka voisivat löytää parempia jakeita.\n"
            f"3. Kirjoita lyhyt selite, joka perustuu näihin konkreettisiin hakusanoihin."
        )

    kehote_strategi = (
        f"ROOLI: Olet huipputason Raamattu-hakustrateegikko.\n"
        f"{analyysi_kehote}\n"
        f"VASTAUKSEN MUOTO: JSON: {{\"selite\": \"Uusi, paranneltu selite...\"}}"
    )
    strategi_data, _ = suorita_varmistettu_json_kutsu([ASIANTUNTIJA_MALLI], kehote_strategi)

    if "virhe" in strategi_data or not (selite := strategi_data.get("selite", "")):
        return {"virhe": "Strategi ei tuottanut selitettä."}

    avainsanat = luo_avainsana_selitteen_pohjalta(selite)
    if not avainsanat:
        logging.warning("Avainsanoittaja epäonnistui. Luodaan geneerinen.")
        sanitized_aihe = re.sub(r'\s+', '_', aihe.split(':')[0].lower())[:20]
        avainsanat = [f"konteksti_{sanitized_aihe}"]

    return {"avainsanat": avainsanat, "selite": selite}


def luo_avainsana_selitteen_pohjalta(selite: str) -> list:
    kehote = (
        f"ROOLI: Olet lingvistiikan asiantuntija.\n"
        f"TEHTÄVÄ: Tiivistä seuraava strategia-selite mahdollisimman ytimekkääksi ja "
        f"yleiskieliseksi termiksi (yleensä 1-2 sanaa).\n"
        f"SELITE: \"{selite}\"\n"
        f"VASTAUKSEN MUOTO: JSON: {{\"avainsanat\": [\"sana1\", \"sana2\"]}}"
    )
    data, _ = suorita_varmistettu_json_kutsu([ASIANTUNTIJA_MALLI], kehote)
    return data.get("avainsanat", [])


def tallenna_uusi_strategia(avainsanat: list, selite: str):
    try:
        with open(LOGIC_TIEDOSTOPOLKU, 'r+', encoding='utf-8') as f:
            content = f.read()
            temp_sanakirja = STRATEGIA_SANAKIRJA.copy()
            for sana in avainsanat:
                temp_sanakirja[sana.lower()] = selite
            sanakirja_str = "STRATEGIA_SANAKIRJA = " + pprint.pformat(temp_sanakirja, indent=4, width=88)
            content_new, count = re.subn(
                r"STRATEGIA_SANAKIRJA = \{.*?\}",
                sanakirja_str, content, count=1,
                flags=re.DOTALL | re.MULTILINE
            )
            if count == 0:
                logging.error("STRATEGIA_SANAKIRJA-lohkoa ei löytynyt.")
                return

            temp_siemenkartta = STRATEGIA_SIEMENJAE_KARTTA.copy()
            for sana in avainsanat:
                if sana.lower() not in temp_siemenkartta:
                    temp_siemenkartta[sana.lower()] = "Lisää siemenjae manuaalisesti"
            siemenkartta_str = "STRATEGIA_SIEMENJAE_KARTTA = " + pprint.pformat(temp_siemenkartta, indent=4, width=88)
            content_new, count = re.subn(
                r"STRATEGIA_SIEMENJAE_KARTTA = \{.*?\}",
                siemenkartta_str, content_new, count=1,
                flags=re.DOTALL | re.MULTILINE
            )

            f.seek(0)
            f.write(content_new)
            f.truncate()
        logging.info(f"Strategia {avainsanat} tallennettu/päivitetty.")
    except Exception as e:
        logging.error(f"Kriittinen virhe strategian tallennuksessa: {e}")


def luo_kontekstisidonnainen_avainsana(sana: str, selite: str) -> str:
    kehote = (
        f"ROOLI: Olet hakustrateegikko.\n"
        f"TEHTÄVÄ: Avainsana '{sana}' on liian yleinen. Luo sille tarkempi, "
        f"1-2 sanan uniikki vastine, joka kuvaa tätä erityistä kontekstiä: \"{selite}\".\n"
        f"Käytä muotoa 'pääsana-tarkenne'.\n"
        f"VASTAUKSEN MUOTO: JSON: {{\"uusi_avainsana\": \"ehdotuksesi tähän\"}}"
    )
    data, _ = suorita_varmistettu_json_kutsu([ASIANTUNTIJA_MALLI], kehote)
    return data.get("uusi_avainsana", f"{sana}_konteksti_{int(time.time())}")