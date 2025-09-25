# monitor_app.py (Versio 2.0 - Korjattu päivityslogiikka)
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from monitoring import get_gpu_stats, get_system_stats

st.set_page_config(
    page_title="Järjestelmän kuormitus",
    page_icon="⚙️",
    layout="centered"
)


st.title("⚙️ Reaaliaikainen kuormitus")

# Streamlitin autorefresh-komponentti. Päivitysväli 2 sekuntia.
st_autorefresh(interval=2000, limit=None, key="autorefresh")

# Haetaan ja näytetään järjestelmän tiedot
sys_stats = get_system_stats()
gpu_stats = get_gpu_stats()

st.metric(
    label="CPU Käyttöaste", value=f"{sys_stats['cpu_percent']:.1f} %"
)
st.metric(
    label="RAM Käyttöaste", value=f"{sys_stats['ram_percent']:.1f} %"
)

if gpu_stats:
    st.metric(
        label="GPU Käyttöaste", value=f"{gpu_stats['gpu_util_percent']:.1f} %"
    )
    st.metric(
        label="VRAM Käyttöaste",
        value=(
            f"{gpu_stats['vram_used_gb']:.2f} / "
            f"{gpu_stats['vram_total_gb']:.2f} GB"
        )
    )
    st.metric(
        label="GPU Lämpötila", value=f"{gpu_stats['temp_c']} °C"
    )
else:
    st.warning("NVIDIA GPU -tietoja ei saatavilla.")