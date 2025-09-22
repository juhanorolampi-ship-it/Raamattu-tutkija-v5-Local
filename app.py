# app.py (Versio 12.0 - Integroitu oppiminen, PEP-8)
import logging
import re
from collections import defaultdict
from io import BytesIO

import docx
import streamlit as st

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

# --- Sivun asetukset ---
st.set_page_config(
    page_title="Raamattu-tutkija v6",
    page_icon="📚",
    layout="wide"
)


# Aseta lokituksen käsittelijä Streamlitin st.info/st.warning-funktioille
class StreamlitLogHandler(logging.Handler):
    def __init__(self, container):
        super().__init__()
        self.container = container

    def emit(self, record):
        msg = self.format(record)
        if record.levelno == logging.WARNING:
            self.container.warning(msg)
        elif record.levelno >= logging.ERROR:
            self.container.error(msg)
        else:
            self.container.info(msg)


# Poista aiemmat käsittelijät ja lisää omat
logger = logging.getLogger()
if logger.hasHandlers():
    logger.handlers.clear()


# --- Apufunktiot ---
def lue_syote_data(syote_data):
    """Jäsentää syötetiedon vankasti ja valmistelee sen hakua varten."""
    if not syote_data:
        return None, None, None, None

    if hasattr(syote_data, 'getvalue'):
        sisalto = syote_data.getvalue().decode("utf-8")
    else:
        sisalto = str(syote_data)
    sisalto = sisalto.replace('\r\n', '\n')

    paaotsikko_m = re.search(r"^(.*?)\n", sisalto)
    paaotsikko = paaotsikko_m.group(1).strip() if paaotsikko_m else ""

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
            haku = f"{otsikko}: {kuvaus.replace('\n', ' ')}"
            hakulauseet[osio_nro] = haku
            otsikot[osio_nro] = otsikko

    sl_match = re.search(r"Sisällysluettelo:(.*?)(?=\n\d\.|\Z)", sisalto,
                         re.DOTALL)
    sl_teksti = sl_match.group(1).strip() if sl_match else ""

    return paaotsikko, hakulauseet, otsikot, sl_teksti


def luo_raportti_md(sl, jae_kartta, arvosanat):
    """Luo siistin tekstimuotoisen raportin."""
    md = f"# {sl['otsikko']}\n\n"
    if sl['teksti']:
        md += "## Sisällysluettelo\n\n"
        md += sl['teksti'] + "\n\n"

    sorted_osiot = sorted(
        jae_kartta.items(),
        key=lambda item: [int(p) for p in item[0].split('.')]
    )
    for osio_nro, data in sorted_osiot:
        arvosana = arvosanat.get(osio_nro, "N/A")
        md += f"## {data['otsikko']} (Lopullinen arvosana: {arvosana}/10)\n\n"
        if data["jakeet"]:
            for jae in data["jakeet"]:
                md += f"- **{jae['viite']}**: \"{jae['teksti']}\"\n"
        else:
            md += "*Ei jakeita tähän osioon.*\n"
        md += "\n"
    return md


def luo_raportti_doc(sl, jae_kartta, arvosanat):
    """Luo ladattavan Word-dokumentin."""
    doc = docx.Document()
    doc.add_heading(sl['otsikko'], 0)
    if sl['teksti']:
        doc.add_heading("Sisällysluettelo", 1)
        doc.add_paragraph(sl['teksti'])

    sorted_osiot = sorted(
        jae_kartta.items(),
        key=lambda item: [int(p) for p in item[0].split('.')]
    )
    for osio_nro, data in sorted_osiot:
        arvosana = arvosanat.get(osio_nro, "N/A")
        doc.add_heading(f"{data['otsikko']} (Lopullinen arvosana: {arvosana}/10)", 1)
        if data["jakeet"]:
            for jae in data["jakeet"]:
                p = doc.add_paragraph()
                p.add_run(f"{jae['viite']}: ").bold = True
                p.add_run(f"\"{jae['teksti']}\"")
        else:
            doc.add_paragraph("Ei jakeita tähän osioon.")

    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer


# --- Streamlit-käyttöliittymä ---
st.title("📚 Raamattu-tutkija v6 (Itseoppiva agentti)")
st.markdown("---")

with st.spinner("Valmistellaan hakukonetta..."):
    lataa_resurssit()

if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Syötä tutkielman aihe ja rakenne")
    uploaded_file = st.file_uploader("1. Lataa .txt-tiedosto", type=['txt'])
    syote_alue = st.text_area("2. Kirjoita tai liitä sisältö tähän", height=350)

with col2:
    st.subheader("Asetukset")
    top_k_valinta = st.slider("Jakeita per osio:", 1, 100, 15)
    tavoitearvosana_valinta = st.number_input(
        "Tavoiteltava laatuarvosana (1-10)",
        min_value=1, max_value=10, value=8
    )
    max_yritykset_valinta = st.number_input(
        "Maksimi parannusyritystä per osio",
        min_value=1, max_value=5, value=3
    )
    oppiminen_paalla = st.checkbox(
        "Oppiminen päällä (tallentaa strategiat pysyvästi)",
        value=False,
        help=("Jos tämä on valittuna, sovellus muokkaa `logic.py`-tiedostoa, "
              "kun se löytää parannuksia.")
    )
    suorita_nappi = st.button("Suorita älykäs haku", type="primary")

st.markdown("---")
st.subheader("Prosessi ja tulokset")

if suorita_nappi:
    koko_syote = ""
    if uploaded_file is not None:
        koko_syote += uploaded_file.getvalue().decode("utf-8")
    koko_syote += "\n\n" + syote_alue.strip()
    koko_syote = koko_syote.strip()

    if not koko_syote:
        st.warning("Syötä aihe ja rakenne.")
    else:
        st.session_state.processing_complete = False
        paaotsikko, hakulauseet, otsikot, sl_teksti = lue_syote_data(koko_syote)

        if not hakulauseet:
            st.error("Syötettä ei voitu jäsentää.")
            st.stop()

        sl = {"otsikko": paaotsikko, "teksti": sl_teksti}
        jae_kartta = defaultdict(lambda: {"jakeet": [], "otsikko": ""})
        lopulliset_arvosanat = {}

        log_container = st.expander("Näytä prosessin loki", expanded=True)
        sh = StreamlitLogHandler(log_container)
        sh.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(sh)

        temp_strategiat = STRATEGIA_SANAKIRJA.copy()
        temp_siemenjakeet = STRATEGIA_SIEMENJAE_KARTTA.copy()

        sorted_hakulauseet = sorted(
            hakulauseet.items(),
            key=lambda item: [int(p) for p in item[0].split('.')]
        )

        for i, (osio_nro, haku) in enumerate(sorted_hakulauseet):
            log_container.markdown(f"--- \n ### Käsitellään osiota "
                                   f"{i+1}/{len(sorted_hakulauseet)}: "
                                   f"{otsikot.get(osio_nro, '')}")

            tulokset = etsi_merkityksen_mukaan(
                haku,
                top_k=top_k_valinta,
                custom_strategiat=temp_strategiat,
                custom_siemenjakeet=temp_siemenjakeet
            )
            arvio = arvioi_tulokset(haku, tulokset)
            kokonaisarvosana = arvio.get("kokonaisarvosana")

            logging.info(f"Alkuperäinen arvosana: {kokonaisarvosana}/10")

            if (kokonaisarvosana is not None and
                    kokonaisarvosana < tavoitearvosana_valinta):
                yritykset = 0
                edellinen_ehdotus = None

                while (arvio.get("kokonaisarvosana") is not None and
                       arvio["kokonaisarvosana"] < tavoitearvosana_valinta and
                       yritykset < max_yritykset_valinta):
                    yritykset += 1
                    log_container.markdown(f"**Parannusyritys "
                                           f"{yritykset}/{max_yritykset_valinta}**")

                    jae_arviot = arvio.get("jae_arviot", [])
                    hyvat_jakeet = [
                        next((item for item in tulokset if item['viite'] == jae_arvio['viite']), None)
                        for jae_arvio in jae_arviot
                        if jae_arvio.get('arvosana', 0) >= tavoitearvosana_valinta
                    ]
                    hyvat_jakeet = [j for j in hyvat_jakeet if j]

                    poistettavien_maara = top_k_valinta - len(hyvat_jakeet)
                    logging.info(f"Säilytetään {len(hyvat_jakeet)} jaetta. "
                                 f"Haetaan {poistettavien_maara} korvaajaa.")

                    if poistettavien_maara > 0:
                        ehdotus = ehdota_uutta_strategiaa(haku, arvio,
                                                           edellinen_ehdotus)
                        edellinen_ehdotus = ehdotus.copy()

                        if "virhe" in ehdotus:
                            logging.error("Strategiaehdotus epäonnistui.")
                            break

                        avainsanat = ehdotus.get('avainsanat', [])
                        selite = ehdotus.get('selite', '')
                        if not avainsanat or not selite:
                            logging.error("Strategiaehdotus oli tyhjä.")
                            break

                        logging.info(f"  -> Luotu strategia, avainsanat: {avainsanat}")

                        for sana in avainsanat:
                            temp_strategiat[sana.lower()] = selite

                        paikkaushaku = etsi_merkityksen_mukaan(
                            haku, poistettavien_maara, temp_strategiat
                        )
                        tulokset = hyvat_jakeet + paikkaushaku

                        vanha_kokonaisarvosana = kokonaisarvosana
                        arvio = arvioi_tulokset(haku, tulokset)
                        kokonaisarvosana = arvio.get("kokonaisarvosana")

                        logging.info(f"Ed. arvosana: {vanha_kokonaisarvosana}/10 -> "
                                     f"UUSI: {kokonaisarvosana}/10")

                        if (kokonaisarvosana is not None and
                                vanha_kokonaisarvosana is not None and
                                kokonaisarvosana > vanha_kokonaisarvosana):
                            logging.info("TULOS: Laatu parani! ✅")
                            if oppiminen_paalla:
                                uniikit_sanat = []
                                for sana in avainsanat:
                                    if sana.lower() in STRATEGIA_SANAKIRJA:
                                        u_sana = luo_kontekstisidonnainen_avainsana(sana, selite)
                                        uniikit_sanat.append(u_sana)
                                    else:
                                        uniikit_sanat.append(sana)
                                tallenna_uusi_strategia(uniikit_sanat, selite)
                        else:
                            logging.warning("TULOS: Laatu ei parantunut. ⚠️")
                    else:
                        logging.info("Kaikki jakeet OK.")
                        break

            jae_kartta[osio_nro]["jakeet"] = tulokset
            jae_kartta[osio_nro]["otsikko"] = otsikot.get(osio_nro, haku.split(':')[0])
            lopulliset_arvosanat[osio_nro] = kokonaisarvosana

        logger.removeHandler(sh)
        st.session_state.final_report_md = luo_raportti_md(sl, jae_kartta, lopulliset_arvosanat)
        st.session_state.final_report_doc = luo_raportti_doc(sl, jae_kartta, lopulliset_arvosanat)
        st.session_state.processing_complete = True
        st.success("Haku suoritettu onnistuneesti!")
        st.rerun()

if st.session_state.processing_complete:
    st.markdown("---")
    st.subheader("Lopullinen tutkielma")
    st.markdown(st.session_state.final_report_md)
    st.divider()
    st.subheader("Lataa raportti")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Lataa .txt-tiedostona",
            st.session_state.final_report_md,
            "raamattu_tutkielma.txt"
        )
    with col2:
        st.download_button(
            "Lataa .docx-tiedostona",
            st.session_state.final_report_doc,
            "raamattu_tutkielma.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )