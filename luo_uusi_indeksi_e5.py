# luo_uusi_indeksi_e5.py (Versio 1.0)
import json
import logging
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# --- MÄÄRITYKSET ---
RAAMATTU_TIEDOSTO = "D:/Python_AI/Raamattu-tutkija-data/bible.json"
UUSI_EMBEDDING_MALLI = "intfloat/multilingual-e5-large"
UUSI_INDEKSI_TIEDOSTO = "D:/Python_AI/Raamattu-tutkija-data/raamattu_indeksi_e5_large.faiss"
UUSI_KARTTA_TIEDOSTO = "D:/Python_AI/Raamattu-tutkija-data/raamattu_kartta_e5_large.json"
ERAKOKO = 32  # Käsitellään jakeita erissä muistin säästämiseksi

# --- LOKITUS ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    datefmt="%H:%M:%S",
)


def luo_ja_tallenna_indeksi():
    """
    Lukee Raamatun, luo kontekstuaaliset upotukset e5-large-mallilla
    ja tallentaa ne uuteen FAISS-indeksiin.
    """
    logging.info(f"Aloitetaan uuden indeksin luonti mallilla: {UUSI_EMBEDDING_MALLI}")

    # Tarkistetaan, onko GPU käytettävissä
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Käytetään laitetta: {device}")

    try:
        with open(RAAMATTU_TIEDOSTO, "r", encoding="utf-8") as f:
            raamattu_data = json.load(f)
    except Exception as e:
        logging.error(f"Tiedostoa '{RAAMATTU_TIEDOSTO}' ei voitu lukea: {e}")
        return

    # 1. Jäsennellään Raamattu ja kerätään kaikki jakeet
    kaikki_jakeet = []
    logging.info("Jäsennellään Raamattua...")
    for book_obj in raamattu_data.get("book", {}).values():
        kirjan_nimi = book_obj.get("info", {}).get("name")
        for luku_nro, luku_obj in book_obj.get("chapter", {}).items():
            for jae_nro, jae_obj in luku_obj.get("verse", {}).items():
                teksti = jae_obj.get("text", "").strip()
                if teksti and kirjan_nimi:
                    viite = f"{kirjan_nimi} {luku_nro}:{jae_nro}"
                    kaikki_jakeet.append({"viite": viite, "teksti": teksti})

    logging.info(f"Jäsennys valmis. Löydettiin {len(kaikki_jakeet)} jaetta.")
    if not kaikki_jakeet:
        return

    # 2. Luodaan 3 jakeen konteksti-ikkunat
    konteksti_tekstit = []
    konteksti_viitteet = []
    logging.info("Luodaan kontekstuaalisia kokonaisuuksia (3 jakeen ikkuna)...")
    for i, jae in enumerate(kaikki_jakeet):
        edellinen = kaikki_jakeet[i-1]["teksti"] if i > 0 else ""
        nykyinen = jae["teksti"]
        seuraava = kaikki_jakeet[i+1]["teksti"] if i < len(kaikki_jakeet) - 1 else ""
        
        # E5-mallit vaativat tämän etuliitteen parhaan suorituskyvyn saavuttamiseksi
        koko_teksti = f"passage: {edellinen} {nykyinen} {seuraava}".strip()
        
        konteksti_tekstit.append(koko_teksti)
        konteksti_viitteet.append(jae["viite"])

    # 3. Ladataan uusi, tehokas embedding-malli
    logging.info(f"Ladataan mallia '{UUSI_EMBEDDING_MALLI}'. Tämä voi kestää hetken...")
    model = SentenceTransformer(UUSI_EMBEDDING_MALLI, device=device)

    # 4. Luodaan vektorit erissä
    logging.info(f"Muunnetaan {len(konteksti_tekstit)} tekstinpätkää vektoreiksi...")
    
    # model.encode tukee suoraan eräkäsittelyä ja näyttää edistymispalkin
    vektorit = model.encode(
        konteksti_tekstit,
        batch_size=ERAKOKO,
        show_progress_bar=True
    )

    logging.info("Vektorien luonti valmis.")

    # 5. Luodaan ja tallennetaan FAISS-indeksi ja viitekartta
    vektorin_ulottuvuus = vektorit.shape[1]
    indeksi = faiss.IndexFlatL2(vektorin_ulottuvuus)
    indeksi.add(np.array(vektorit, dtype=np.float32))

    faiss.write_index(indeksi, UUSI_INDEKSI_TIEDOSTO)
    logging.info(f"Uusi indeksi tallennettu: '{UUSI_INDEKSI_TIEDOSTO}'")

    viite_kartta = {str(i): viite for i, viite in enumerate(konteksti_viitteet)}
    with open(UUSI_KARTTA_TIEDOSTO, "w", encoding="utf-8") as f:
        json.dump(viite_kartta, f, ensure_ascii=False, indent=4)
    logging.info(f"Uusi viitekartta tallennettu: '{UUSI_KARTTA_TIEDOSTO}'")
    
    logging.info("Valmista! Uusi, tehokkaampi vektoritietokanta on luotu.")


if __name__ == "__main__":
    luo_ja_tallenna_indeksi()