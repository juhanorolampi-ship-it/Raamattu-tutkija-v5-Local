# monitoring.py (Versio 3.1 - Lisätty tiedoston flush-toiminto)
import csv
import time
import psutil

try:
    import pynvml
    # Alustetaan yhteys vain kerran, kun moduuli ladataan
    pynvml.nvmlInit()
    NVIDIA_SMI_AVAILABLE = True
except (ImportError, pynvml.NVMLError):
    NVIDIA_SMI_AVAILABLE = False

def get_gpu_stats():
    """Hakee NVIDIA-näytönohjaimen tilastot, jos saatavilla."""
    if not NVIDIA_SMI_AVAILABLE:
        return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

        stats = {
            "vram_total_gb": mem_info.total / (1024**3),
            "vram_used_gb": mem_info.used / (1024**3),
            "gpu_util_percent": util.gpu,
            "temp_c": temp,
        }
        return stats
    except pynvml.NVMLError:
        return None

def get_system_stats():
    """Hakee prosessorin ja RAM-muistin käytön."""
    stats = {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "ram_percent": psutil.virtual_memory().percent,
    }
    return stats

def setup_performance_logger():
    """Luo aikaleimatun CSV-lokin ja kirjoittaa otsikkorivin."""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_filename = f"performance_log_{timestamp}.csv"
    
    file = open(log_filename, 'w', newline='', encoding='utf-8')
    writer = csv.writer(file)
    
    header = [
        "aikaleima", "cpu_kayttoaste_%", "ram_kayttoaste_%",
        "gpu_kayttoaste_%", "vram_kaytetty_gb", "gpu_lampotila_c"
    ]
    writer.writerow(header)
    # Palautetaan nyt myös tiedosto-olio, jotta voimme käyttää sitä
    return writer, file

def log_performance_stats(writer, file_handle):
    """Hakee ja kirjaa nykyiset suorituskykytiedot ja varmistaa tallennuksen."""
    sys_stats = get_system_stats()
    gpu_stats = get_gpu_stats()
    
    timestamp = time.strftime("%H:%M:%S")
    
    row = [
        timestamp,
        f"{sys_stats['cpu_percent']:.1f}",
        f"{sys_stats['ram_percent']:.1f}",
    ]
    
    if gpu_stats:
        row.extend([
            f"{gpu_stats['gpu_util_percent']:.1f}",
            f"{gpu_stats['vram_used_gb']:.2f}",
            f"{gpu_stats['temp_c']}"
        ])
    else:
        row.extend(["N/A", "N/A", "N/A"])
        
    writer.writerow(row)
    # HUOM: Tämä on tärkeä lisäys! Se pakottaa puskurin kirjoittamaan levylle heti.
    file_handle.flush()