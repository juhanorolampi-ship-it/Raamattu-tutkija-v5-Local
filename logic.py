# logic.py (Versio 16.0 - Kontekstitietoinen esianalyysi)
import json
import logging
import re
import faiss
import numpy as np
import streamlit as st
import ollama
from sentence_transformers import SentenceTransformer, CrossEncoder

# --- VAKIOASETUKSET ---
PAAINDESKI_TIEDOSTO = "D:/Python_AI/Raamattu-tutkija-data/raamattu_vektori_indeksi.faiss"
PAAKARTTA_TIEDOSTO = "D:/Python_AI/Raamattu-tutkija-data/raamattu_viite_kartta.json"
RAAMATTU_TIEDOSTO = "D:/Python_AI/Raamattu-tutkija-data/bible.json"
EMBEDDING_MALLI = "TurkuNLP/sbert-cased-finnish-paraphrase"
CROSS_ENCODER_MALLI = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# Mallit eri tehtäviin
ARVIOINTI_MALLI = "qwen2.5:14b-instruct"
KIELENHUOLTO_MALLI = "poro-local"

# --- STRATEGIAKERROS JA KARTTA ---
# (Nämä pysyvät samoina kuin aiemmin, jätetään pois selkeyden vuoksi)
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
    # ... (Sama kuin aiemmin)
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
    # ... (Sama kuin aiemmin)
    pattern = r'((?:[1-3]\.\s)?[A-ZÅÄÖa-zåäö]+\.?\s\d+:\d+(?:-\d+)?)'
    return re.findall(pattern, teksti)


def hae_jakeet_viitteella(viite_str: str, jae_haku_kartta: dict) -> list[dict]:
    """Hakee jaejoukon tekstistä poimitun viitteen perusteella."""
    # ... (Sama kuin aiemmin)
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


# --- UUSI ESIANALYYSIFUNKTIO ---
def onko_strategia_relevantti(kysely: str, selite: str) -> bool:
    """
    Kysyy tekoälyltä, onko löydetty strategia relevantti
    annettuun hakukyselyyn.
    """
    kehote = f"""
ROOLI JA TAVOITE:
Olet looginen päättelijä. Tehtäväsi on arvioida, onko annettu strategia hyödyllinen tietyn hakukyselyn tarkentamiseen.

KONTEKSTI:
Saat käyttäjän hakukyselyn ja siihen liittyvän strategian selityksen. Päätä, auttaako strategian soveltaminen löytämään parempia ja tarkempia vastauksia juuri tähän nimenomaiseen kyselyyn. Älä arvioi strategian yleistä hyvyyttä, vaan ainoastaan sen soveltuvuutta tähän tilanteeseen.

- Käyttäjän kysely: "{kysely}"
- Tarjottu strategia: "{selite}"

VASTAUKSEN MUOTO:
Vastaa AINA ja AINOASTAAN JSON-muodossa. Objektin tulee sisältää yksi avain: "sovellu", jonka arvo on true (jos strategia sopii) tai false (jos strategia ei sovi).
"""
    try:
        logging.info("Suoritetaan strategian relevanssin esianalyysi...")
        response = ollama.chat(
            model=ARVIOINTI_MALLI,
            messages=[{'role': 'user', 'content': kehote.strip()}]
        )
        vastaus_teksti = response['message']['content']
        
        json_match = re.search(r'\{.*\}', vastaus_teksti, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            relevanssi = data.get("sovellu", False)
            logging.info(f"Esianalyysin tulos: Soveltuuko strategia? {'Kyllä' if relevanssi else 'Ei'}.")
            return relevanssi
        return False
    except Exception as e:
        logging.error(f"Virhe esianalyysissä: {e}")
        return False # Oletuksena ei sovelleta, jos tulee virhe


def etsi_merkityksen_mukaan(kysely: str, top_k: int = 15) -> list[dict]:
    """
    Etsii Raamatusta käyttäen kontekstitietoista hybridihakua.
    """
    resurssit = lataa_resurssit()
    if not all(resurssit):
        logging.error("Haku epäonnistui, koska resursseja ei voitu ladata.")
        return []

    model, cross_encoder, paaindeksi, paakartta, jae_haku_kartta = resurssit

    viite_str_lista = poimi_raamatunviitteet(kysely)
    # ... (Pakollisten jakeiden poiminta pysyy samana)
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

    laajennettu_kysely = kysely
    pien_kysely = kysely.lower()

    # PÄIVITETTY LOGIIKKA ESIANALYYSILLÄ
    for avainsana, selite in STRATEGIA_SANAKIRJA.items():
        if avainsana in pien_kysely:
            # Uusi vaihe: Varmista strategian relevanssi
            if onko_strategia_relevantti(kysely, selite):
                logging.info(f"Strategia '{avainsana}' todettiin relevantiksi.")
                siemenjae_viite = STRATEGIA_SIEMENJAE_KARTTA.get(avainsana)
                if siemenjae_viite:
                    siemenjae_teksti = jae_haku_kartta.get(siemenjae_viite, "")
                    logging.info(f"Manuaalisesti valittu siemenjae: {siemenjae_viite}")
                    laajennettu_kysely = (
                        f"Aihe on: '{kysely}'. Teeman selitys on: '{selite}'. "
                        f"Tärkeä esimerkki aiheesta on jae '{siemenjae_viite}', "
                        f"joka kuuluu: '{siemenjae_teksti}'."
                    )
                else:
                    laajennettu_kysely = f"{selite}. Alkuperäinen aihe on: {kysely}"
                break # Käytetään ensimmäistä relevanttia strategiaa
            else:
                logging.info(f"Strategia '{avainsana}' hylättiin epärelevanttina tähän hakuun.")
    
    # Hakulogiikan loppuosa pysyy samana...
    alyhaun_tulokset = []
    if top_k > 0:
        # ... (Kerroinlogiikka)
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
            # ... (Vektorihaku ja uudelleenjärjestys)
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
    return lopulliset_tulokset


# Laadunarviointi- ja strategiaehdotusfunktiot pysyvät samoina
def arvioi_tulokset(aihe: str, tulokset: list) -> dict:
    # ... (Sama kuin aiemmin)
    if not tulokset:
        return {"arvosana": None, "perustelu": "Ei tuloksia arvioitavaksi."}

    tulokset_str = "\n".join(
        [f"{i+1}. {jae['viite']}: \"{jae['teksti']}\"" for i, jae in enumerate(tulokset)]
    )

    kehote = f"""
ROOLI JA TAVOITE:
Olet teologinen asiantuntija. Tehtäväsi on arvioida annettujen Raamatun jakeiden relevanssia ja laatua suhteessa annettuun hakuaiheeseen.

ARVIOINTIKRITEERIT:
- 10/10 (Täydellinen): Tulokset sisältävät juuri ne avainjakeet, joita aiheeseen tarvitaan.
- 7-9/10 (Hyvä/Erinomainen): Tulokset ovat selkeästi relevantteja ja tukevat teemaa hyvin.
- 4-6/10 (Kohtalainen): Tulokset ovat aihepiiriltään oikeansuuntaisia, mutta jäävät yleisiksi.
- 1-3/10 (Heikko): Tulokset ovat pääosin epärelevantteja.

VASTAUKSEN MUOTO:
Vastaa AINA ja AINOASTAAN JSON-muodossa. Älä kirjoita mitään muuta tekstiä. JSON-objektin tulee sisältää kaksi avainta: "arvosana" (kokonaisluku 1-10) ja "perustelu" (lyhyt, merkkijonomuotoinen selitys arvosanalle).
Esimerkki: {{"arvosana": 8, "perustelu": "Tulokset ovat pääosin relevantteja, mutta eivät sisällä kaikkia avainjakeita."}}

NYKYINEN TEHTÄVÄ:
Arvioi seuraavat tulokset ja palauta vastauksesi JSON-muodossa.

- Aihe: {aihe}
- Tulokset:
{tulokset_str}
"""

    try:
        logging.info(f"Lähetetään JSON-arviointipyyntö mallille {ARVIOINTI_MALLI}...")
        response = ollama.chat(
            model=ARVIOINTI_MALLI,
            messages=[{'role': 'user', 'content': kehote.strip()}]
        )
        vastaus_teksti = response['message']['content']
        logging.info("Vastaus vastaanotettu.")

        try:
            json_match = re.search(r'\{.*\}', vastaus_teksti, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                arvosana = data.get("arvosana")
                perustelu = data.get("perustelu", "Perustelua ei löytynyt JSON-vastauksesta.")
                
                if arvosana is not None:
                    arvosana = int(arvosana)
                
                return {"arvosana": arvosana, "perustelu": perustelu}
            else:
                raise json.JSONDecodeError("JSON-objektia ei löytynyt", vastaus_teksti, 0)
        except (json.JSONDecodeError, TypeError) as e:
            logging.error(f"Vastauksen JSON-jäsennys epäonnistui: {e}. Vastaus: {vastaus_teksti}")
            return {"arvosana": None, "perustelu": f"Virheellinen JSON-vastaus: {vastaus_teksti}"}

    except Exception as e:
        logging.error(f"Virhe arviointimallin kutsumisessa: {e}")
        return {"arvosana": None, "perustelu": f"Virhe arvioinnissa: {e}"}


def ehdota_uutta_strategiaa(aihe: str, tulokset: list, arvio: dict) -> dict:
    # ... (Sama kuin aiemmin)
    tulokset_str = "\n".join(
        [f"{i+1}. {jae['viite']}: \"{jae['teksti']}\"" for i, jae in enumerate(tulokset)]
    )
    edellinen_perustelu = arvio.get("perustelu", "Ei annettua perustelua.")

    kehote_qwen = f"""
ROOLI JA TAVOITE:
Olet Raamattu-hakukoneen vanhempi kehittäjä. Tehtäväsi on parantaa hakutulosten laatua luomalla uusia sääntöjä hakukoneen STRATEGIA_SANAKIRJA-komponenttiin.
KONTEKSTI JA ANALYYSI:
Haku tietylle aiheelle tuotti heikkoja tuloksia. Tässä on alkuperäinen aihe, sen tuottamat tulokset ja aiempi analyysisi siitä, MIKSI tulokset olivat heikkoja.
- Alkuperäinen aihe: {aihe}
- Heikot tulokset:
{tulokset_str}
- Aiempi analyysisi: {edellinen_perustelu}
TEHTÄVÄ:
1. Analysoi annettua aihetta ja aiempaa analyysiäsi.
2. Tunnista alkuperäisestä aiheesta 1-2 keskeistä ydinavainsanaa.
3. Kirjoita näille avainsanoille uusi, yleiskäyttöinen ja selittävä sääntö (`selite`).
VASTAUKSEN MUOTO:
Vastaa AINA ja AINOASTAAN JSON-muodossa, joka sisältää avaimet "avainsanat" ja "selite".
Esimerkki:
{{"avainsanat": ["koetinkivi", "testi"], "selite": "Etsi jakeita, jotka käsittelevät luonteen testaamista ja koettelemista erityisissä olosuhteissa."}}
"""

    try:
        logging.info(f"Lähetetään strategiaehdotuspyyntö mallille {ARVIOINTI_MALLI} (Analyytikko)...")
        response_qwen = ollama.chat(
            model=ARVIOINTI_MALLI,
            messages=[{'role': 'user', 'content': kehote_qwen.strip()}]
        )
        raaka_vastaus_teksti = response_qwen['message']['content']
        logging.info("Analyytikon luonnos vastaanotettu.")
        
        json_match = re.search(r'\{.*\}', raaka_vastaus_teksti, re.DOTALL)
        if not json_match:
            raise ValueError("Analyytikko ei tuottanut validia JSON-luonnosta.")
        
        raaka_json = json.loads(json_match.group(0))
    except Exception as e:
        logging.error(f"Virhe strategiaehdotuksen Vaiheessa 1 (Analyytikko): {e}")
        return {"virhe": f"Virhe Analyytikko-vaiheessa: {e}"}

    raaka_avainsanat = raaka_json.get("avainsanat", [])
    raaka_selite = raaka_json.get("selite", "")
    kehote_poro = f"""
ROOLI JA TAVOITE:
Olet suomen kielen asiantuntija. Tehtäväsi on viimeistellä tekoälyn tuottama tekstiluonnos.
TEHTÄVÄ:
1. Tarkista oheiset avainsanat. Varmista, että ne ovat kieliopillisesti oikein ja perusmuodossa (nominatiivi).
2. Uudelleenkirjoita oheinen selite selkeäksi, ytimekkääksi ja kieliopillisesti virheettömäksi suomen kieleksi säilyttäen sen alkuperäinen merkitys.
LUONNOS:
- Avainsanat: {raaka_avainsanat}
- Selite: {raaka_selite}
VASTAUKSEN MUOTO:
Vastaa AINA ja AINOASTAAN JSON-muodossa, joka sisältää avaimet "avainsanat" ja "selite".
"""
    try:
        logging.info(f"Lähetetään kielenhuoltopyyntö mallille {KIELENHUOLTO_MALLI} (Toimittaja)...")
        response_poro = ollama.chat(
            model=KIELENHUOLTO_MALLI,
            messages=[{'role': 'user', 'content': kehote_poro.strip()}]
        )
        valmis_vastaus_teksti = response_poro['message']['content']
        logging.info("Toimittajan viimeistelemä versio vastaanotettu.")
        
        json_match_final = re.search(r'\{.*\}', valmis_vastaus_teksti, re.DOTALL)
        if json_match_final:
            json_str = json_match_final.group(0)
            valmis_json = json.loads(json_str)
            return valmis_json
        else:
            logging.warning(
                "Toimittaja ei tuottanut validia JSON-vastausta. "
                f"Palautetaan Analyytikon raakaversio. Toimittajan vastaus: {valmis_vastaus_teksti}"
            )
            return raaka_json

    except Exception as e:
        logging.warning(
            f"Virhe strategiaehdotuksen Vaiheessa 2 (Toimittaja): {e}. "
            "Palautetaan Analyytikon raakaversio."
        )
        return raaka_json