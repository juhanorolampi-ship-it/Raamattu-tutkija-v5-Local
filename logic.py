# logic.py (Versio 21.0 - Jakeiden yksilöllinen arviointi, PEP-8)
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
EMBEDDING_MALLI = "intfloat/multilingual-e5-large"
CROSS_ENCODER_MALLI = "cross-encoder/ms-marco-MiniLM-L-6-v2"
ARVIOINTI_MALLI = "gemma3:latest"
ASIANTUNTIJA_MALLI = "gemma3:latest"


# --- STRATEGIAKERROS JA KARTTA ---
STRATEGIA_SANAKIRJA = {   'intohimo': 'Hae jakeita, jotka kuvaavat sydämen paloa, innostusta, syvää '
                'mielenkiintoa tai Jumalan antamaa tahtoa ja paloa tiettyä asiaa tai '
                'tehtävää kohtaan.',
    'jännite': 'Hae Raamatusta kohtia, jotka kuvaavat rakentavaa erimielisyyttä, '
               'toisiaan täydentäviä rooleja tai sitä, miten erilaisuus johtaa '
               'hengelliseen kasvuun ja terveen jännitteen kautta parempaan '
               'lopputulokseen.',
    'koetinkivi': 'Etsi jakeita, jotka käsittelevät luonteen testaamista ja '
                  'koettelemista erityisissä olosuhteissa, kuten vastoinkäymisissä, '
                  'menestyksessä, kritiikin alla tai näkymättömyydessä.',
    'kohdenna': 'Edellinen strategia oli epäonnistunut, koska se keskittyi liian '
                'yleisesti Korinttilaiskirjeen teemoihin ja ei riittävästi '
                'konkreettisiin ongelmiin, joita Paavali käsittelee kirjeessä. '
                'Hakusanat eivät suoraan kohdistuneet korintlaisten ongelmiin ja '
                'Paavalin vastauksuihin. Parempi lähestymistapa on eristää kirjeen '
                'keskeiset nuhketukset ja Paavalin opetukset.',
    'kritiikki': 'Etsi jakeita, jotka opastavat, miten suhtautua oikeutetusti tai '
                 'epäoikeutetusti saatuun kritiikkiin, arvosteluun tai nuhteeseen '
                 'säilyttäen nöyrän ja opetuslapseen sopivan sydämen.',
    'kyvyt': 'Etsi kohtia, jotka käsittelevät luontaisia, synnynnäisiä taitoja, '
             'lahjakkuutta ja osaamista, jotka Jumala on ihmiselle antanut ja joita '
             'voidaan käyttää hänen kunniakseen.',
    'näkymättömyys': 'Hae jakeita, jotka käsittelevät palvelemista ilman ihmisten '
                     'näkemystä, kiitosta tai tunnustusta, keskittyen Jumalan '
                     'palkkioon ja oikeaan sydämen asenteeseen.',
    'paimen': 'Etsi kohtia, jotka kuvaavat profeetallisen ja pastoraalisen tai '
              'opettavan roolin välistä dynamiikkaa, yhteistyötä tai jännitettä '
              'seurakunnassa.',
    'pappi': 'Etsi kohtia, jotka kuvaavat profeetallisen ja pastoraalisen tai '
             'opettavan roolin välistä dynamiikkaa, yhteistyötä tai jännitettä '
             'seurakunnassa.',
    'profeetta': 'Etsi kohtia, jotka kuvaavat profeetallisen ja pastoraalisen tai '
                 'opettavan roolin välistä dynamiikkaa, yhteistyötä tai jännitettä '
                 'seurakunnassa.',
    'raamatullinen': 'Edellinen strategia oli liian abstrakti ja pyrki käsittelemään '
                     'laajoja teemoja ilman riittävää kontekstuaalista tukea. Se ei '
                     'tarjonnut riittävästi konkreettisia termejä, joiden avulla '
                     'Raamatun jakeita voisi hakea systemaattisesti. Parempi '
                     'lähestymistapa keskittyy selkeisiin johtamisprosessin '
                     'elementteihin ja niiden raamatullisiin esimerkkeihin.',
    'tasapaino': 'Etsi kohtia, jotka käsittelevät tasapainoa, harmoniaa tai oikeaa '
                 'suhdetta kahden eri asian, kuten työn ja levon, tai totuuden ja '
                 'rakkauden, välillä.',
    'testi': 'Etsi jakeita, jotka käsittelevät luonteen testaamista ja koettelemista '
             'erityisissä olosuhteissa, kuten vastoinkäymisissä, menestyksessä, '
             'kritiikin alla tai näkymättömyydessä.'}
STRATEGIA_SIEMENJAE_KARTTA = {   'intohimo': 'Room. 12:1-2',
    'jännite': 'Room. 12:4-5',
    'koetinkivi': 'Jaak. 1:2-4',
    'kohdenna': 'Lisää siemenjae manuaalisesti',
    'kritiikki': 'Miika 6:8',
    'kyvyt': '1. Piet. 4:10',
    'näkymättömyys': 'Fil. 2:3-4',
    'paimen': 'Ef. 4:11-12',
    'pappi': 'Ef. 4:11-12',
    'profeetta': 'Ef. 4:11-12',
    'raamatullinen': 'Lisää siemenjae manuaalisesti',
    'tasapaino': 'Saarn. 3:1',
    'testi': 'Jaak. 1:2-4'}


# --- LOKITUKSEN ALUSTUS ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    datefmt="%H:%M:%S",
)


# --- RESURSSIEN LATAUS JA APUFUNKTIOT ---
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


# --- VANKAT TEKOÄLYKUTSUT ---
def suorita_varmistettu_json_kutsu(malli: str, kehote: str,
                                 max_yritykset: int = 3) -> dict:
    """
    Suorittaa LLM-kutsun ja varmistaa, että vastaus on validi JSON.
    Yrittää uudelleen ja antaa palautetta virheestä.
    """
    alkuperainen_kehote = kehote
    edellinen_virhe_viesti = ""

    for yritys in range(max_yritykset):
        try:
            # Lisätään yksityiskohtainen virheilmoitus kehotteeseen toisella yrittämällä
            if edellinen_virhe_viesti:
                kehote_laajennus = (
                    f"\n\n--- EDELTÄVÄ YRITYS EPÄONNISTUI ---\n"
                    f"Edellinen vastauksesi aiheutti JSON-jäsentelyvirheen: '{edellinen_virhe_viesti}'.\n"
                    f"Ole hyvä ja korjaa virhe. Varmista, että KOKO vastauksesi on YKSI validi JSON-objekti, "
                    f"joka alkaa merkillä '{{' ja päättyy merkkiin '}}', ja että kaikki rakenteet (listat, objektit, pilkut) ovat oikein.\n"
                    f"--- VIRHEEN KUVAUS LOPPUU ---\n\n"
                )
                kehote = alkuperainen_kehote + kehote_laajennus

            msg = (f"Lähetetään JSON-pyyntö mallille {malli} "
                   f"(Yritys {yritys + 1}/{max_yritykset})...")
            logging.info(msg)

            response = ollama.chat(
                model=malli,
                messages=[{'role': 'user', 'content': kehote.strip()}]
            )
            vastaus_teksti = response['message']['content']

            json_match = re.search(r'\{.*\}', vastaus_teksti, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                logging.info("Vastaus jäsennelty onnistuneesti.")
                return data
            else:
                edellinen_virhe_viesti = "Vastauksesta ei löytynyt JSON-objektia."
                raise json.JSONDecodeError(edellinen_virhe_viesti, vastaus_teksti, 0)

        except json.JSONDecodeError as e:
            edellinen_virhe_viesti = str(e)
            msg = (f"Virhe JSON-jäsennnyksessä (yritys {yritys + 1}): {e}. "
                   "Yritetään uudelleen...")
            logging.warning(msg)

    logging.error(f"JSON-vastausta ei saatu {max_yritykset} yrityksestä.")
    return {"virhe": f"JSON-vastausta ei saatu {max_yritykset} yrityksestä."}


# --- PÄÄFUNKTIOT ---
def onko_strategia_relevantti(kysely: str, selite: str) -> bool:
    """Kysyy tekoälyltä, onko löydetty strategia relevantti."""
    kehote = f"""
ROOLI: Olet looginen päättelijä.
TEHTÄVÄ: Arvioi, onko strategia hyödyllinen hakukyselyn tarkentamiseen.
- Kysely: "{kysely}"
- Strategia: "{selite}"
VASTAA JSON-MUODOSSA: {{"sovellu": true/false}}
"""
    logging.info("Suoritetaan strategian relevanssin esianalyysi...")
    data = suorita_varmistettu_json_kutsu(ARVIOINTI_MALLI, kehote)
    relevanssi = data.get("sovellu", False)
    logging.info(f"Esianalyysin tulos: Soveltuuko strategia? "
                 f"{'Kyllä' if relevanssi else 'Ei'}.")
    return relevanssi


def etsi_merkityksen_mukaan(kysely: str, top_k: int = 15,
                          custom_strategiat: dict = None,
                          custom_siemenjakeet: dict = None) -> list[dict]:
    """Etsii Raamatusta käyttäen kontekstitietoista hybridihakua."""
    resurssit = lataa_resurssit()
    if not all(resurssit):
        return []

    model, cross_encoder, paaindeksi, paakartta, jae_haku_kartta = resurssit
    strategia_lahde = (custom_strategiat if custom_strategiat is not None
                     else STRATEGIA_SANAKIRJA)
    siemenjae_lahde = (custom_siemenjakeet if custom_siemenjakeet is not None
                     else STRATEGIA_SIEMENJAE_KARTTA)

    viite_str_lista = poimi_raamatunviitteet(kysely)
    pakolliset_jakeet = []
    loytyneet_viitteet = set()
    for viite_str in viite_str_lista:
        jakeet = hae_jakeet_viitteella(viite_str, jae_haku_kartta)
        for jae in jakeet:
            if jae["viite"] not in loytyneet_viitteet:
                pakolliset_jakeet.append(jae)
                loytyneet_viitteet.add(jae["viite"])

    laajennettu_kysely = kysely
    pien_kysely = kysely.lower()

    for avainsana, selite in strategia_lahde.items():
        if avainsana in pien_kysely:
            if onko_strategia_relevantti(kysely, selite):
                logging.info(f"Strategia '{avainsana}' todettiin relevantiksi.")
                siemenjae_viite = siemenjae_lahde.get(avainsana)
                if siemenjae_viite:
                    siemenjae_teksti = jae_haku_kartta.get(siemenjae_viite, "")
                    laajennettu_kysely = (
                        f"{kysely}. Hakua ohjaava lisäkonteksti: {selite}. "
                        f"Teemaa havainnollistava jae: '{siemenjae_teksti}'."
                    )
                else:
                    # YHDISTETÄÄN ALKUPERÄINEN JA UUSI STRATEGIA
                    laajennettu_kysely = f"{kysely}. Hakua tarkentava strategia: {selite}"
                break
            else:
                msg = (f"Strategia '{avainsana}' hylättiin epärelevanttina.")
                logging.info(msg)

    alyhaun_tulokset = []
    if top_k > 0:
        kerroin = max(5, 11 - (top_k // 10))
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

    return pakolliset_jakeet + alyhaun_tulokset


def arvioi_tulokset(aihe: str, tulokset: list) -> dict:
    """Arvioi hakutulokset ja antaa jokaiselle jakeelle oman arvosanan."""
    if not tulokset:
        return {"kokonaisarvosana": None, "jae_arviot": [],
                "kokonaisperustelu": "Ei tuloksia arvioitavaksi."}

    tulokset_str = "\n".join(
        [f"- {jae['viite']}: \"{jae['teksti']}\"" for jae in tulokset]
    )
    kehote = f"""
ROOLI: Olet ankara teologinen asiantuntija.
TEHTÄVÄ: Arvioi annettujen Raamatun jakeiden relevanssi suhteessa aiheeseen.
Anna KOKONAISARVIO ja KOKONAISPERUSTELU. Lisäksi anna JOKAISELLE jakeelle
oma ARVOSANA (1-10) ja lyhyt PERUSTELU.

- Aihe: {aihe}
- Tulokset:
{tulokset_str}

VASTAUKSEN MUOTO: Vastaa AINA JSON-muodossa. Varmista, että jokainen jae
tuloslistasta löytyy `jae_arviot`-listasta. Esimerkki:
{{
  "kokonaisarvosana": 7,
  "kokonaisperustelu": "Hyvä alku, mutta osa jakeista on epärelevantteja.",
  "jae_arviot": [
    {{ "viite": "1. Kor. 7:15", "arvosana": 9, "perustelu": "Osuu suoraan." }},
    {{ "viite": "Jer. 49:20", "arvosana": 4, "perustelu": "Kontekstisidonnainen." }}
  ]
}}
"""
    data = suorita_varmistettu_json_kutsu(ARVIOINTI_MALLI, kehote)
    if "virhe" in data:
        return {"kokonaisarvosana": None, "jae_arviot": [],
                "kokonaisperustelu": f"Virheellinen vastaus: {data['virhe']}"}

    return data


def luo_avainsana_selitteen_pohjalta(selite: str) -> list:
    """Luo 1-2 avainsanaa annetun selitteen perusteella."""
    logging.info("Luodaan avainsanoja selitteen pohjalta...")
    kehote = f"""
ROOLI: Olet lingvistiikan asiantuntija.
TEHTÄVÄ: Tiivistä seuraava strategia-selite mahdollisimman ytimekkääksi ja
yleiskieliseksi termiksi (yleensä 1-2 sanaa).
SELITE: "{selite}"
VASTAUKSEN MUOTO: JSON: {{"avainsanat": ["sana1", "sana2"]}}
"""
    data = suorita_varmistettu_json_kutsu(ASIANTUNTIJA_MALLI, kehote)
    return data.get("avainsanat", [])

def ehdota_uutta_strategiaa(aihe: str, arvio: dict,
                           edellinen_ehdotus: dict = None) -> dict:
    """Ehdottaa uutta, konkreettista strategiaa hakutulosten parantamiseksi."""
    kokonaisperustelu = arvio.get('kokonaisperustelu', 'Ei perustelua.')
    analyysi_kehote = ""

    if edellinen_ehdotus:
        analyysi_kehote = f"""
SYVEMPI ANALYYSI (ITSEKRITIIKKI):
Edellinen parannusyritys epäonnistui. Laatu ei parantunut.
- EDELLINEN STRATEGIA: {edellinen_ehdotus.get('selite', '')}
- TULOKSEN ARVIOINTI: Perustelu: "{kokonaisperustelu}"

UUSI TEHTÄVÄ:
1.  **Analysoi lyhyesti (1-2 lausetta), miksi edellinen strategia oli liian abstrakti tai epäonnistunut.**
2.  **Ehdota 2-4 TÄSMÄLLISTÄ JA KONKREETTISTA HAKUSANAA tai LYHYTTÄ HAKULAUSETTA**, jotka voisivat löytää parempia jakeita. Älä käytä abstrakteja metaforia. Keskity termeihin, jotka löytyvät todennäköisesti suoraan Raamatusta.
3.  **Kirjoita lyhyt selite**, joka perustuu näihin konkreettisiin hakusanoihin ja selittää, miksi ne ovat parempia.
"""
    else:
        analyysi_kehote = f"""
ONGELMA-ANALYYSIN KEHOTE:
Ensimmäinen haku aiheelle "{aihe}" tuotti heikkoja tuloksia, joiden perustelu oli: "{kokonaisperustelu}".

TEHTÄVÄ:
1.  **Ehdota 2-4 TÄSMÄLLISTÄ JA KONKREETTISTA HAKUSANAA tai LYHYTTÄ HAKULAUSETTA**, jotka ratkaisevat arvioijan mainitsemat puutteet.
2.  **Kirjoita lyhyt selite**, joka perustuu näihin konkreettisiin hakusanoihin.
"""

    kehote_strategi = f"""
ROOLI: Olet huipputason Raamattu-hakustrategikko. Tavoitteesi on tuottaa konkreettisia hakulauseita, ei abstrakteja ideoita.
{analyysi_kehote}
VASTAUKSEN MUOTO: JSON. Varmista, että selite viittaa ehdottamiisi konkreettisiin hakusanoihin. Esimerkki:
{{
  "selite": "Haku epäonnistui, koska se oli liian laaja. Parempi lähestymistapa on keskittyä termeihin kuten 'hengellinen viisaus' ja 'valheelliset opettajat', jotka kohdistavat haun tarkemmin ytimeen."
}}
"""
    strategi_data = suorita_varmistettu_json_kutsu(ASIANTUNTIJA_MALLI,
                                                    kehote_strategi)
    if "virhe" in strategi_data:
        return {"virhe": "Strategian luonti epäonnistui."}

    selite = strategi_data.get("selite", "")
    if not selite:
        return {"virhe": "Strategi ei tuottanut selitettä."}

    avainsanat = luo_avainsana_selitteen_pohjalta(selite)
    if not avainsanat:
        logging.warning("Avainsanoittaja epäonnistui. Luodaan geneerinen.")
        placeholder = re.sub(r'\s+', '_', aihe.split(':')[0].lower())[:20]
        avainsanat = [f"konteksti_{placeholder}"]

    return {"avainsanat": avainsanat, "selite": selite}


def tallenna_uusi_strategia(avainsanat: list, selite: str):
    """Lisää uuden strategian tai päivittää olemassa olevan."""
    try:
        with open(LOGIC_TIEDOSTOPOLKU, 'r', encoding='utf-8') as f:
            content = f.read()

        temp_sanakirja = STRATEGIA_SANAKIRJA.copy()
        for sana in avainsanat:
            temp_sanakirja[sana.lower()] = selite

        sanakirja_str = ("STRATEGIA_SANAKIRJA = " +
                         pprint.pformat(temp_sanakirja, indent=4, width=88))

        content, count = re.subn(
            r"STRATEGIA_SANAKIRJA = \{.*?\}",
            sanakirja_str,
            content,
            count=1,
            flags=re.DOTALL | re.MULTILINE
        )
        if count == 0:
            logging.error("STRATEGIA_SANAKIRJA-lohkoa ei löytynyt.")
            return

        temp_siemenkartta = STRATEGIA_SIEMENJAE_KARTTA.copy()
        for sana in avainsanat:
            if sana.lower() not in temp_siemenkartta:
                temp_siemenkartta[sana.lower()] = "Lisää siemenjae manuaalisesti"

        siemenkartta_str = ("STRATEGIA_SIEMENJAE_KARTTA = " +
                            pprint.pformat(temp_siemenkartta, indent=4, width=88))

        content, count = re.subn(
            r"STRATEGIA_SIEMENJAE_KARTTA = \{.*?\}",
            siemenkartta_str,
            content,
            count=1,
            flags=re.DOTALL | re.MULTILINE
        )
        if count == 0:
            logging.error("STRATEGIA_SIEMENJAE_KARTTA-lohkoa ei löytynyt.")
            return

        with open(LOGIC_TIEDOSTOPOLKU, 'w', encoding='utf-8') as f:
            f.write(content)

        logging.info(f"Strategia {avainsanat} tallennettu/päivitetty.")

    except Exception as e:
        logging.error(f"Kriittinen virhe strategian tallennuksessa: {e}")


def luo_kontekstisidonnainen_avainsana(sana: str, selite: str) -> str:
    """Luo uniikin, kontekstiin sidotun avainsanan duplikaattien välttämiseksi."""
    kehote = f"""
    ROOLI: Olet hakustrategikko.
    TEHTÄVÄ: Avainsana '{sana}' on liian yleinen. Luo sille tarkempi,
    1-2 sanan uniikki vastine, joka kuvaa tätä erityistä kontekstiä: "{selite}".
    Käytä muotoa 'pääsana-tarkenne'.
    Esimerkki: Jos sana on 'tasapaino' ja konteksti 'armo ja totuus',
    hyvä vastine voisi olla 'tasapaino-armo ja totuus'.
    VASTAUKSEN MUOTO: JSON: {{"uusi_avainsana": "ehdotuksesi tähän"}}
    """
    data = suorita_varmistettu_json_kutsu("qwen2.5:14b-instruct", kehote)
    return data.get("uusi_avainsana", f"{sana}_konteksti_{int(time.time())}")