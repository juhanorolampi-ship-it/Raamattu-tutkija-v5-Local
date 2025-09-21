# app.py (Versio 8.2 - Yhdistetty syötteenkäsittely)
import re
from collections import defaultdict
from io import BytesIO
import docx
import streamlit as st

# Varmistetaan, että tuodaan uusin logiikka
from logic import etsi_merkityksen_mukaan, lataa_resurssit


# --- Sivun asetukset ---
st.set_page_config(
    page_title="Raamattu-tutkija v4",
    page_icon="📚",
    layout="wide"
)


# --- Apufunktiot ---
def lue_syote_data(syote_data):
    """
    Jäsentää syötetiedon vankasti ja valmistelee sen hakua varten.
    """
    if not syote_data:
        return None, None, None, None

    # Käsitellään sekä tiedostoa että tekstikenttää
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

    sl_match = re.search(r"Sisällysluettelo:(.*?)(?=\n\d\.|\Z)", sisalto, re.DOTALL)
    sl_teksti = sl_match.group(1).strip() if sl_match else ""

    return paaotsikko, hakulauseet, otsikot, sl_teksti


def luo_raportti_md(sl, jae_kartta):
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
        md += f"## {data['otsikko']}\n\n"
        if data["jakeet"]:
            for jae in data["jakeet"]:
                md += f"- **{jae['viite']}**: \"{jae['teksti']}\"\n"
        else:
            md += "*Ei jakeita tähän osioon.*\n"
        md += "\n"
    return md


def luo_raportti_doc(sl, jae_kartta):
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
        doc.add_heading(data['otsikko'], 1)
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
st.title("📚 Raamattu-tutkija v4")
st.markdown("---")

with st.spinner("Valmistellaan hakukonetta..."):
    lataa_resurssit()

if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Syötä tutkielman aihe ja rakenne")
    
    uploaded_file = st.file_uploader(
        "1. Lataa .txt-tiedosto (valinnainen)",
        type=['txt'],
        help="Voit joko ladata tiedoston tai käyttää alla olevaa tekstikenttää."
    )

    syote_alue = st.text_area(
        "2. Kirjoita tai liitä sisältö tähän (voit myös täydentää ladattua tiedostoa)",
        height=350,
        placeholder="Esimerkiksi:\n\nOsa 1: Johdanto\nTeema: Kutsu kuuluu kaikille..."
    )

with col2:
    st.subheader("Asetukset")
    top_k_valinta = st.slider(
        "Kuinka monta jaetta haetaan per osio?", 1, 100, 15
    )
    suorita_nappi = st.button("Suorita haku", type="primary")

st.markdown("---")
st.subheader("Hakutulokset")

if suorita_nappi:
    syote_tiedostosta = ""
    if uploaded_file is not None:
        syote_tiedostosta = uploaded_file.getvalue().decode("utf-8")
        st.info(f"Käytetään ladattua tiedostoa: {uploaded_file.name}")

    syote_tekstikentasta = syote_alue.strip()
    
    koko_syote = (syote_tiedostosta + "\n\n" + syote_tekstikentasta).strip()
    
    if not koko_syote:
        st.warning("Syötä aihe ja rakenne joko tiedostona tai tekstikenttään.")
    else:
        with st.spinner("Suoritetaan älykästä hakua... Tämä voi kestää hetken."):
            paaotsikko, hakulauseet, otsikot, sl_teksti = lue_syote_data(koko_syote)

            if not hakulauseet:
                st.error("Syötettä ei voitu jäsentää. Varmista, että se on oikeassa muodossa.")
                st.stop()
            
            sl = {"otsikko": paaotsikko, "teksti": sl_teksti}
            jae_kartta = defaultdict(lambda: {"jakeet": [], "otsikko": ""})
            total = len(hakulauseet)
            p_bar = st.progress(0, text="Aloitetaan...")

            sorted_hakulauseet = sorted(
                hakulauseet.items(),
                key=lambda item: [int(p) for p in item[0].split('.')]
            )

            for i, (osio_nro, haku) in enumerate(sorted_hakulauseet):
                tulokset = etsi_merkityksen_mukaan(haku, top_k_valinta)
                jae_kartta[osio_nro]["jakeet"] = tulokset
                jae_kartta[osio_nro]["otsikko"] = otsikot.get(
                    osio_nro, haku.split(':')[0]
                )
                p_text = f"Käsitellään osiota {i+1}/{total}: {osio_nro}"
                p_bar.progress((i + 1) / total, text=p_text)

            st.session_state.final_report_md = luo_raportti_md(sl, jae_kartta)
            st.session_state.final_report_doc = luo_raportti_doc(sl, jae_kartta)
            st.session_state.processing_complete = True
            p_bar.empty()
            st.success("Haku suoritettu onnistuneesti!")

if st.session_state.processing_complete:
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