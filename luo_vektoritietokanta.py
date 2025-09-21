# luo_vektoritietokanta.py (Versio 3.4 - Oikea JSON-rakenne)
import json
import logging
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# --- MÄÄRITYKSET ---
RAAMATTU_TIEDOSTO = "D:/Python_AI/Raamattu-tutkija-data/bible.json"
VEKTORI_INDEKSI_TIEDOSTO = "D:/Python_AI/Raamattu-tutkija-data/raamattu_vektori_indeksi.faiss"
VIITE_KARTTA_TIEDOSTO = "D:/Python_AI/Raamattu-tutkija-data/raamattu_viite_kartta.json"
EMBEDDING_MALLI = "TurkuNLP/sbert-cased-finnish-paraphrase"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    datefmt="%H:%M:%S",
)

def luo_vektoritietokanta():
    """
    Lukee Raamatun, luo kontekstuaalisia 3 jakeen kokonaisuuksia,
    luo niistä vektoriupotukset ja tallentaa ne FAISS-indeksiin.
    """
    logging.info("Aloitetaan vektoritietokannan (v3.4 - Oikea JSON-rakenne) luonti...")

    try:
        with open(RAAMATTU_TIEDOSTO, "r", encoding="utf-8") as f:
            raamattu_data = json.load(f)
    except Exception as e:
        logging.error(f"Raamatun datatiedostoa '{RAAMATTU_TIEDOSTO}' ei voitu lukea: {e}")
        return

    model = SentenceTransformer(EMBEDDING_MALLI)
    
    # --- TÄYSIN UUSITTU JÄSENNYSLOGIIKKA, JOKA VASTAA bible.json RAKENNETTA ---
    kaikki_jakeet = []
    logging.info("Jäsennellään Raamattua ja kerätään kaikki jakeet...")
    
    if "book" not in raamattu_data or not isinstance(raamattu_data["book"], dict):
        logging.error(f"Tiedostosta '{RAAMATTU_TIEDOSTO}' ei löytynyt 'book'-objektia.")
        return

    # Käydään läpi kirja-objektin arvot (1, 2, 3...)
    for book_obj in raamattu_data["book"].values():
        kirjan_nimi = book_obj.get("info", {}).get("name")
        luvut_obj = book_obj.get("chapter")

        if not kirjan_nimi or not isinstance(luvut_obj, dict):
            continue

        # Käydään läpi luku-objektin arvot (1, 2, 3...)
        for luku_nro, luku_obj in luvut_obj.items():
            jakeet_obj = luku_obj.get("verse")
            
            if not isinstance(jakeet_obj, dict):
                continue
            
            # Käydään läpi jae-objektin arvot (1, 2, 3...)
            for jae_nro, jae_obj in jakeet_obj.items():
                teksti = jae_obj.get("text", "").strip()
                if teksti:
                    viite = f"{kirjan_nimi} {luku_nro}:{jae_nro}"
                    kaikki_jakeet.append({"viite": viite, "teksti": teksti})

    logging.info(f"Jäsennys valmis. Löydettiin yhteensä {len(kaikki_jakeet)} jaetta.")

    if not kaikki_jakeet:
        logging.error("Jakeiden kerääminen epäonnistui. Vektorikantaa ei luoda.")
        return
        
    konteksti_tekstit = []
    konteksti_viitteet = []
    logging.info("Luodaan kontekstuaalisia jakeiden kokonaisuuksia (3 jakeen ikkuna)...")

    for i, jae in enumerate(kaikki_jakeet):
        edellinen_teksti = kaikki_jakeet[i-1]["teksti"] if i > 0 else ""
        nykyinen_teksti = jae["teksti"]
        seuraava_teksti = kaikki_jakeet[i+1]["teksti"] if i < len(kaikki_jakeet) - 1 else ""
        
        koko_teksti = f"{edellinen_teksti} {nykyinen_teksti} {seuraava_teksti}".strip()
        
        konteksti_tekstit.append(koko_teksti)
        konteksti_viitteet.append(jae["viite"])

    logging.info(f"Kerätty {len(konteksti_tekstit)} kontekstuaalista kokonaisuutta. Muunnetaan vektoreiksi...")
    
    vektorit = model.encode(konteksti_tekstit, show_progress_bar=True)
    
    if vektorit.ndim < 2 or vektorit.shape[0] == 0:
        logging.error("Vektorien luonti epäonnistui. Indeksiä ei luoda.")
        return

    vektorin_ulottuvuus = vektorit.shape[1]
    indeksi = faiss.IndexFlatL2(vektorin_ulottuvuus)
    indeksi.add(np.array(vektorit, dtype=np.float32))

    faiss.write_index(indeksi, VEKTORI_INDEKSI_TIEDOSTO)
    logging.info(f"Uusi indeksi tallennettu: '{VEKTORI_INDEKSI_TIEDOSTO}'")

    viite_kartta = {str(i): viite for i, viite in enumerate(konteksti_viitteet)}
    with open(VIITE_KARTTA_TIEDOSTO, "w", encoding="utf-8") as f:
        json.dump(viite_kartta, f, ensure_ascii=False, indent=4)
    logging.info(f"Uusi viitekartta tallennettu: '{VIITE_KARTTA_TIEDOSTO}'")
    
    logging.info("Vektorikannan luonti onnistui!")

if __name__ == "__main__":
    luo_vektoritietokanta()