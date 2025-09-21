# logic.py (Versio 20.0 - Finaali: Analyyttinen ote)
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
PAAINDESKI_TIEDOSTO = "D:/Python_AI/Raamattu-tutkija-data/raamattu_vektori_indeksi.faiss"
PAAKARTTA_TIEDOSTO = "D:/Python_AI/Raamattu-tutkija-data/raamattu_viite_kartta.json"
RAAMATTU_TIEDOSTO = "D:/Python_AI/Raamattu-tutkija-data/bible.json"
EMBEDDING_MALLI = "TurkuNLP/sbert-cased-finnish-paraphrase"
CROSS_ENCODER_MALLI = "cross-encoder/ms-marco-MiniLM-L-6-v2"
ARVIOINTI_MALLI = "qwen2.5:14b-instruct"
KIELENHUOLTO_MALLI = "poro-local"
AVAINSAINOITTAJA_MALLI = "poro-local"


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
    'kritiikki': 'Etsi jakeita, jotka opastavat, miten suhtautua oikeutetusti tai '
                 'epäoikeutetusti saatuun kritiikkiin, arvosteluun tai nuhteeseen '
                 'säilyttäen nöyrän ja opetuslapseen sopivan sydämen.',
    'kyvyt': 'Etsi kohtia, jotka käsittelevät luontaisia, synnynnäisiä taitoja, '
             'lahjakkuutta ja osaamista, jotka Jumala on ihmiselle antanut ja joita '
             'voidaan käyttää hänen kunniakseen.',
    'nuorten kehitys': 'Strategiaa etsiessä nuorten uskonnon kehittämisen haasteisiin '
                       'liittyviä laadukkaita ja tarkkoja kirjallisia lähteitä, '
                       'keskitetään pyrkimykset nykyajan ongelmanratkaisemiseen. '
                       'Huomioidaan seuraavat näkökulmat: 1) Nuorten uskonnon '
                       'kehittämisen haasteiden syvällinen ymmärrys ja lähestyminen '
                       'niitä koskevan kontekstin kautta, 2) Esimerkkiesitteet '
                       'vanhempien roolista nuorisokuussa sekä heidän aikeissaan oman '
                       'uskonnutensa kehittämiseksi, 3) Tärkeiden liikkeitjen ja '
                       'verkostojen työn arviointi Jumalan viestinnän näkökulmasta, '
                       'välttäen yksityiskohtisten asiantuntijakriitikoiden '
                       'vaarallisen puuttuvuuden. Huomioitava on myös kyky tunnistaa '
                       'ja arvostaa todellista hedelmää senkin tapahtumisessa '
                       'epätäydellisin olosuhteissa, mikä edellyttää luontevaan '
                       'uskonnon syventämistä ja monipuolisuutta. Tavoitteena on '
                       'laatia strategian, joka antaa nuorille varmasti hyödyllisen ja '
                       'merkityksellisen opas heidän kriittisten huolenpiteen '
                       'tukemiseksi.',
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
    'rakkaus': '5. Yhteenveto: Totuus rakkaudessa - Teema tässä yhteenvetossa on '
               'korostaa sitä tasapainoa, joka meidän on saavutettava totuuden ja '
               'rauhan välillä sekä karittomien ihmisten suojelun ja heidän '
               'kasvattamisen. Tämä sisältö painottaa kärsivällisyyden ja armon '
               'tekemistä niitä kohtaan, jotka ovat edelleen matkansa alussa, samalla '
               'kun valppauttaa eksytystä vastaan ja suojellaa totuuden puolesta. '
               'Tämän aiheen tavoitteena on antaa kuvaus siitä, miten kristinuskunnan '
               'jäsenille pitää olla sekä vahvoja ja lujia että hyvin armollisia ja '
               'käsittävällisiä.',
    'rauhanvari': '5. Yhteenveto: Totuus rakkaudessa - Teema tässä yhteenvetossa on '
                  'korostaa sitä tasapainoa, joka meidän on saavutettava totuuden ja '
                  'rauhan välillä sekä karittomien ihmisten suojelun ja heidän '
                  'kasvattamisen. Tämä sisältö painottaa kärsivällisyyden ja armon '
                  'tekemistä niitä kohtaan, jotka ovat edelleen matkansa alussa, '
                  'samalla kun valppauttaa eksytystä vastaan ja suojellaa totuuden '
                  'puolesta. Tämän aiheen tavoitteena on antaa kuvaus siitä, miten '
                  'kristinuskunnan jäsenille pitää olla sekä vahvoja ja lujia että '
                  'hyvin armollisia ja käsittävällisiä.',
    'strategia': 'Strategiaa etsiessä nuorten uskonnon kehittämisen haasteisiin '
                 'liittyviä laadukkaita ja tarkkoja kirjallisia lähteitä, keskitetään '
                 'pyrkimykset nykyajan ongelmanratkaisemiseen. Huomioidaan seuraavat '
                 'näkökulmat: 1) Nuorten uskonnon kehittämisen haasteiden syvällinen '
                 'ymmärrys ja lähestyminen niitä koskevan kontekstin kautta, 2) '
                 'Esimerkkiesitteet vanhempien roolista nuorisokuussa sekä heidän '
                 'aikeissaan oman uskonnutensa kehittämiseksi, 3) Tärkeiden '
                 'liikkeitjen ja verkostojen työn arviointi Jumalan viestinnän '
                 'näkökulmasta, välttäen yksityiskohtisten asiantuntijakriitikoiden '
                 'vaarallisen puuttuvuuden. Huomioitava on myös kyky tunnistaa ja '
                 'arvostaa todellista hedelmää senkin tapahtumisessa epätäydellisin '
                 'olosuhteissa, mikä edellyttää luontevaan uskonnon syventämistä ja '
                 'monipuolisuutta. Tavoitteena on laatia strategian, joka antaa '
                 'nuorille varmasti hyödyllisen ja merkityksellisen opas heidän '
                 'kriittisten huolenpiteen tukemiseksi.',
    'tasapaino': 'Etsi kohtia, jotka käsittelevät tasapainoa, harmoniaa tai oikeaa '
                 'suhdetta kahden eri asian, kuten työn ja levon, tai totuuden ja '
                 'rakkauden, välillä.',
    'testi': 'Etsi jakeita, jotka käsittelevät luonteen testaamista ja koettelemista '
             'erityisissä olosuhteissa, kuten vastoinkäymisissä, menestyksessä, '
             'kritiikin alla tai näkymättömyydessä.',
    'usko': 'Strategiaa etsiessä nuorten uskonnon kehittämisen haasteisiin liittyviä '
            'laadukkaita ja tarkkoja kirjallisia lähteitä, keskitetään pyrkimykset '
            'nykyajan ongelmanratkaisemiseen. Huomioidaan seuraavat näkökulmat: 1) '
            'Nuorten uskonnon kehittämisen haasteiden syvällinen ymmärrys ja '
            'lähestyminen niitä koskevan kontekstin kautta, 2) Esimerkkiesitteet '
            'vanhempien roolista nuorisokuussa sekä heidän aikeissaan oman '
            'uskonnutensa kehittämiseksi, 3) Tärkeiden liikkeitjen ja verkostojen työn '
            'arviointi Jumalan viestinnän näkökulmasta, välttäen yksityiskohtisten '
            'asiantuntijakriitikoiden vaarallisen puuttuvuuden. Huomioitava on myös '
            'kyky tunnistaa ja arvostaa todellista hedelmää senkin tapahtumisessa '
            'epätäydellisin olosuhteissa, mikä edellyttää luontevaan uskonnon '
            'syventämistä ja monipuolisuutta. Tavoitteena on laatia strategian, joka '
            'antaa nuorille varmasti hyödyllisen ja merkityksellisen opas heidän '
            'kriittisten huolenpiteen tukemiseksi.'}
STRATEGIA_SIEMENJAE_KARTTA = {   'intohimo': 'Room. 12:1-2',
    'jännite': 'Room. 12:4-5',
    'koetinkivi': 'Jaak. 1:2-4',
    'kritiikki': 'Miika 6:8',
    'kyvyt': '1. Piet. 4:10',
    'nuorten kehitys': 'Lisää siemenjae manuaalisesti',
    'näkymättömyys': 'Fil. 2:3-4',
    'paimen': 'Ef. 4:11-12',
    'pappi': 'Ef. 4:11-12',
    'profeetta': 'Ef. 4:11-12',
    'rakkaus': 'Lisää siemenjae manuaalisesti',
    'rauhanvari': 'Lisää siemenjae manuaalisesti',
    'strategia': 'Lisää siemenjae manuaalisesti',
    'tasapaino': 'Saarn. 3:1',
    'testi': 'Jaak. 1:2-4',
    'usko': 'Lisää siemenjae manuaalisesti'}


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
    Yrittää uudelleen tarvittaessa.
    """
    alkuperainen_kehote = kehote
    for yritys in range(max_yritykset):
        try:
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
                raise json.JSONDecodeError("JSON-objektia ei löytynyt",
                                           vastaus_teksti, 0)
        except (json.JSONDecodeError, TypeError) as e:
            msg = (f"Virhe JSON-jäsennnyksessä (yritys {yritys + 1}): {e}. "
                   "Yritetään uudelleen...")
            logging.warning(msg)
            kehote = (
                f"{alkuperainen_kehote}\n\nEDELLINEN VASTAUKSESI OLI "
                f"VIRHEELLINEN:\n---\n{vastaus_teksti}\n---\n"
                "VASTAUKSESI EI OLLUT KELVOLLISTA JSON-MUOTOA. KORJAA VIRHE JA "
                "PALAUTA VAIN JA AINOASTAAN VALIDI JSON-OBJEKTI."
            )
    logging.error(f"JSON-vastausta ei saatu {max_yritykset} yrityksestä.")
    return {"virhe": f"JSON-vastausta ei saatu {max_yritykset} yrityksestä."}


# --- PÄÄFUNKTIOT ---
def onko_strategia_relevantti(kysely: str, selite: str) -> bool:
    """Kysyy tekoälyltä, onko löydetty strategia relevantti."""
    kehote = f"""
ROOLI JA TAVOITE:
Olet looginen päättelijä. Tehtäväsi on arvioida, onko annettu strategia
hyödyllinen tietyn hakukyselyn tarkentamiseen.

KONTEKSTI:
Saat käyttäjän hakukyselyn ja siihen liittyvän strategian selityksen. Päätä,
auttaako strategian soveltaminen löytämään parempia ja tarkempia vastauksia
juuri tähän nimenomaiseen kyselyyn.

- Käyttäjän kysely: "{kysely}"
- Tarjottu strategia: "{selite}"

VASTAUKSEN MUOTO:
Vastaa AINA ja AINOASTAAN JSON-muodossa: {{"sovellu": true/false}}
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
                        f"Aihe on: '{kysely}'. Teeman selitys on: '{selite}'. "
                        f"Tärkeä esimerkki aiheesta on jae '{siemenjae_viite}', "
                        f"joka kuuluu: '{siemenjae_teksti}'."
                    )
                else:
                    laajennettu_kysely = (f"{selite}. "
                                        f"Alkuperäinen aihe on: {kysely}")
                break
            else:
                msg = (f"Strategia '{avainsana}' hylättiin "
                       "epärelevanttina tähän hakuun.")
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
    """Arvioi hakutulokset ja viimeistelee perustelun kielen."""
    if not tulokset:
        return {"arvosana": None, "perustelu": "Ei tuloksia arvioitavaksi."}

    tulokset_str = "\n".join(
        [f"{i+1}. {jae['viite']}: \"{jae['teksti']}\"" for i, jae in enumerate(tulokset)]
    )
    kehote = f"""
ROOLI JA TAVOITE:
Olet teologinen asiantuntija. Arvioi annettujen Raamatun jakeiden relevanssia
ja laatua suhteessa annettuun hakuaiheeseen.
VASTAUKSEN MUOTO:
Vastaa AINA JSON-muodossa: {{"arvosana": <1-10>, "perustelu": "<selitys>"}}
NYKYINEN TEHTÄVÄ:
- Aihe: {aihe}
- Tulokset:
{tulokset_str}
"""
    data = suorita_varmistettu_json_kutsu(ARVIOINTI_MALLI, kehote)
    if "virhe" in data:
        return {"arvosana": None,
                "perustelu": f"Virheellinen vastaus: {data['virhe']}"}

    arvosana = data.get("arvosana")
    raaka_perustelu = data.get("perustelu", "Perustelua ei annettu.")

    if arvosana is not None:
        try:
            arvosana = int(arvosana)
        except (ValueError, TypeError):
            arvosana = None

    # Kielentarkistus perustelulle
    logging.info("Viimeistellään arvioinnin perustelua...")
    kehote_poro = f"""
ROOLI: Olet suomen kielen toimittaja.
TEHTÄVÄ: Viimeistele oheinen lause kieliopillisesti virheettömäksi ja
ytimekkääksi.
LAUSE: "{raaka_perustelu}"
VASTAUKSEN MUOTO: JSON: {{"viimeistelty_perustelu": "Viimeistelty lause."}}
"""
    poro_data = suorita_varmistettu_json_kutsu(KIELENHUOLTO_MALLI,
                                               kehote_poro)
    viimeistelty_perustelu = poro_data.get("viimeistelty_perustelu",
                                          raaka_perustelu)

    return {"arvosana": arvosana, "perustelu": viimeistelty_perustelu}


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
    data = suorita_varmistettu_json_kutsu(AVAINSAINOITTAJA_MALLI, kehote)
    return data.get("avainsanat", [])


def ehdota_uutta_strategiaa(aihe: str, tulokset: list, arvio: dict,
                           edellinen_ehdotus: dict = None) -> dict:
    """Ehdottaa uutta strategiaa selite-ensin-periaatteella."""
    analyysi_kehote = ""
    if edellinen_ehdotus:
        analyysi_kehote = f"""
SYVEMPI ANALYYSI:
Edellinen yritys parantaa tuloksia epäonnistui.
- EDELLINEN STRATEGIA: {edellinen_ehdotus.get('selite', '')}
- TULOKSEN ARVIOINTI: Arvosana oli {arvio.get('arvosana')}/10, koska:
  "{arvio.get('perustelu')}"

UUSI TEHTÄVÄ:
Luo täysin uusi ja laadukkaampi strategian selite, joka ottaa huomioon
yllä olevan perustelun ja korjaa siinä mainitut puutteet.
"""
    else:
        analyysi_kehote = f"""
ONGELMA-ANALYYSIN KEHOTE:
Ensimmäinen haku aiheelle ei tuottanut riittävän laadukkaita tuloksia.
- TULOSTEN ARVIOINTI: Arvosana oli {arvio.get('arvosana')}/10, koska:
  "{arvio.get('perustelu')}"

TEHTÄVÄ:
Tee ensimmäinen, analyyttinen korjausehdotus. Luo laadukas ja tarkka
strategian selite, joka ratkaisee arvioijan mainitsemat puutteet. Keskity
suoraan ongelman korjaamiseen.
"""

    kehote_qwen = f"""
ROOLI: Olet Raamattu-hakukoneen vanhempi kehittäjä.
KONTEKSTI: Haku aiheelle "{aihe}" tuotti heikkoja tuloksia.
{analyysi_kehote}
TEHTÄVÄ: Kirjoita uusi, laadukas ja tarkka selite hakustrategialle.
ÄLÄ luo avainsanoja.
VASTAUKSEN MUOTO: JSON: {{"selite": "Uusi, paranneltu selite..."}}
"""
    qwen_data = suorita_varmistettu_json_kutsu(ARVIOINTI_MALLI, kehote_qwen)
    if "virhe" in qwen_data:
        return {"virhe": "Analyytikko-vaihe epäonnistui."}

    raaka_selite = qwen_data.get("selite", "")
    if not raaka_selite:
        return {"virhe": "Analyytikko ei tuottanut selitettä."}

    luodut_avainsanat = luo_avainsana_selitteen_pohjalta(raaka_selite)
    if not luodut_avainsanat:
        # Varmistus: jos avainsanoittaja epäonnistuu
        logging.warning("Avainsanoittaja epäonnistui. Luodaan geneerinen avainsana.")
        placeholder = re.sub(r'\s+', '_', aihe.split(':')[0].lower())[:20]
        luodut_avainsanat = [f"konteksti_{placeholder}"]

    kehote_poro = f"""
ROOLI: Olet suomen kielen toimittaja.
TEHTÄVÄ: Viimeistele oheinen selite kieliopillisesti virheettömäksi ja
ytimekkääksi.
SELITE: "{raaka_selite}"
VASTAUKSEN MUOTO: JSON: {{"selite": "Viimeistelty selite."}}
"""
    poro_data = suorita_varmistettu_json_kutsu(KIELENHUOLTO_MALLI, kehote_poro)
    if "virhe" in poro_data:
        logging.warning("Toimittaja-vaihe epäonnistui, käytetään raakaa selitettä.")
        return {"avainsanat": luodut_avainsanat, "selite": raaka_selite}

    return {"avainsanat": luodut_avainsanat,
            "selite": poro_data.get("selite", raaka_selite)}


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
    1-2 sanan uniikki vastine, joka kuvaa tätä erityistä kontekstia: "{selite}".
    Käytä muotoa 'pääsana-tarkenne'.
    Esimerkki: Jos sana on 'tasapaino' ja konteksti 'armo ja totuus',
    hyvä vastine voisi olla 'tasapaino-armo ja totuus'.
    VASTAUKSEN MUOTO: JSON: {{"uusi_avainsana": "ehdotuksesi tähän"}}
    """
    data = suorita_varmistettu_json_kutsu("qwen2.5:14b-instruct", kehote)
    return data.get("uusi_avainsana", f"{sana}_konteksti_{int(time.time())}")