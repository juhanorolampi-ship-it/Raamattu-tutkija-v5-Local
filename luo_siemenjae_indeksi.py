# luo_siemenjae_indeksi.py
import json
import logging
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# --- MÄÄRITYKSET ---
RAAMATTU_TIEDOSTO = "D:/Python_AI/Raamattu-tutkija-data/bible.json"
SIEMENJAE_INDEKSI_TIEDOSTO = "D:/Python_AI/Raamattu-tutkija-data/siemenjae_indeksi.faiss"
SIEMENJAE_KARTTA_TIEDOSTO = "D:/Python_AI/Raamattu-tutkija-data/siemenjae_kartta.json"
EMBEDDING_MALLI = "TurkuNLP/sbert-cased-finnish-paraphrase"

# --- KURATOITU LISTA SUPERJAKEISTA ---
SUPERJAKEET = [
    # Jumalan olemus ja toiminta
    "Joh. 3:16", "5. Moos. 7:9", "1. Joh. 4:16",
    "Jes. 46:9-10", "Dan. 2:20-21", "Kol. 1:16-17",
    "Ef. 2:8-9", "2. Moos. 34:6-7", "Hepr. 4:16",
    "Jes. 6:3", "1. Piet. 1:15-16", "Ilm. 4:8",
    "5. Moos. 32:4", "Ps. 119:160", "2. Tim. 2:13",
    "Room. 11:33", "1. Kor. 1:24-25", "Jaak. 3:17",
    "Ps. 9:9-10", "Ap. t. 17:31", "Room. 2:6",
    "Ps. 139:7-10", "Matt. 28:20", "2. Moos. 33:14",
    # Pelastus ja usko
    "Ap. t. 4:12", "Room. 10:9", "1. Tim. 2:5",
    "Room. 5:1", "Gal. 2:16", "Hepr. 11:1",
    "Ap. t. 3:19", "Luuk. 13:3", "2. Kor. 7:10",
    "1. Joh. 2:2", "Room. 3:24-25", "Ef. 1:7",
    "Joh. 3:3", "2. Kor. 5:17", "Jer. 31:33",
    "Room. 8:24", "1. Piet. 1:3", "Joh. 11:25",
    # Kristityn elämä ja kasvu
    "Fil. 4:6", "Matt. 6:9-13", "1. Tess. 5:17",
    "Ef. 4:32", "Kol. 3:13", "Matt. 6:14-15",
    "Joh. 14:26", "Gal. 5:22-23", "Room. 8:14",
    "Hepr. 12:14", "Room. 12:1-2", "Joh. 14:15",
    "Matt. 22:39", "Joh. 13:34-35", "1. Joh. 4:7",
    "Fil. 4:4", "1. Tess. 5:16-18", "Ps. 100:4",
    "Room. 8:28", "Jaak. 1:2-4", "2. Kor. 4:17",
    "Fil. 2:3-4", "Miika 6:8", "1. Piet. 5:6",
    "Sananl. 3:5-6", "Ps. 23:1-3", "Jer. 29:11",
    "Hepr. 12:1-2", "Matt. 24:13", "Ilm. 2:10",
    # Seurakunta ja yhteys
    "1. Kor. 12:27", "Ef. 4:15-16", "Room. 12:4-5",
    "Gal. 3:28", "Ef. 4:3", "Joh. 17:21",
    "1. Kor. 12:4-7", "Room. 12:6", "1. Piet. 4:10",
    "Mark. 10:45", "Gal. 5:13", "Fil. 2:5-7",
    "Ef. 4:11-12", "1. Tim. 3:1-2", "Hepr. 13:17",
    "1. Kor. 11:23-26", "Matt. 28:19", "Ap. t. 2:38",
    "Matt. 28:19-20", "Room. 1:16", "2. Tim. 4:2",
    # Eskatologia
    "Ap. t. 1:11", "1. Tess. 4:16-17", "Ilm. 22:20",
    "1. Kor. 15:20-22", "Joh. 11:25-26", "Room. 6:5",
    "Mark. 1:15", "Room. 14:17", "Luuk. 17:21",
    "Matt. 25:46", "Ilm. 20:15", "2. Tess. 1:9",
    "Ilm. 21:1-4", "2. Piet. 3:13", "Jes. 65:17",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    datefmt="%H:%M:%S",
)


def luo_siemenjae_indeksi():
    """
    Lukee Raamatusta vain ennalta määritellyt superjakeet,
    luo niistä vektoriupotukset ja tallentaa ne omaan FAISS-indeksiin.
    """
    logging.info("Aloitetaan siemenjae-vektoritietokannan luonti...")

    try:
        with open(RAAMATTU_TIEDOSTO, "r", encoding="utf-8") as f:
            raamattu_data = json.load(f)
    except Exception as e:
        logging.error(f"Raamatun datatiedostoa '{RAAMATTU_TIEDOSTO}' ei voitu lukea: {e}")
        return

    model = SentenceTransformer(EMBEDDING_MALLI)

    # Muunnetaan lista setiksi nopeaa hakua varten
    superjakeet_set = set(SUPERJAKEET)
    valitut_jakeet = []

    logging.info("Jäsennellään Raamattua ja poimitaan superjakeet...")
    for book_obj in raamattu_data.get("book", {}).values():
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
                viite = f"{kirjan_nimi} {luku_nro}:{jae_nro}"
                
                # Tarkistetaan, onko jae haluttujen superjakeiden listalla
                if teksti and viite in superjakeet_set:
                    valitut_jakeet.append({"viite": viite, "teksti": teksti})

    logging.info(f"Jäsennys valmis. Löydettiin {len(valitut_jakeet)}/{len(SUPERJAKEET)} superjaetta.")

    if not valitut_jakeet:
        logging.error("Jakeiden kerääminen epäonnistui. Vektorikantaa ei luoda.")
        return
        
    # Erotetaan tekstit ja viitteet omiin listoihinsa
    tekstit_vektorointiin = [jae["teksti"] for jae in valitut_jakeet]
    viitteet_karttaan = [jae["viite"] for jae in valitut_jakeet]

    logging.info(f"Muunnetaan {len(tekstit_vektorointiin)} jaetta vektoreiksi...")
    
    vektorit = model.encode(tekstit_vektorointiin, show_progress_bar=True)
    
    if vektorit.ndim < 2 or vektorit.shape[0] == 0:
        logging.error("Vektorien luonti epäonnistui. Indeksiä ei luoda.")
        return

    vektorin_ulottuvuus = vektorit.shape[1]
    indeksi = faiss.IndexFlatL2(vektorin_ulottuvuus)
    indeksi.add(np.array(vektorit, dtype=np.float32))

    faiss.write_index(indeksi, SIEMENJAE_INDEKSI_TIEDOSTO)
    logging.info(f"Uusi siemenjae-indeksi tallennettu: '{SIEMENJAE_INDEKSI_TIEDOSTO}'")

    viite_kartta = {str(i): viite for i, viite in enumerate(viitteet_karttaan)}
    with open(SIEMENJAE_KARTTA_TIEDOSTO, "w", encoding="utf-8") as f:
        json.dump(viite_kartta, f, ensure_ascii=False, indent=4)
    logging.info(f"Uusi siemenjae-kartta tallennettu: '{SIEMENJAE_KARTTA_TIEDOSTO}'")
    
    logging.info("Siemenjae-vektorikannan luonti onnistui!")


if __name__ == "__main__":
    luo_siemenjae_indeksi()