# -*- coding: utf-8 -*-
"""
Autor: Hernane Velozo Rosa
Trabalho de Conclusão de Curso II – Engenharia de Computação – PUC Minas
Data: 30/11/2025

Descrição:
Este programa implementa o pipeline completo de pré-processamento e extração de características
de sinais eletroencefalográficos (EEG) do dataset TUH EEG Seizure Corpus (v2.0.3). Ele realiza:

1. Detecção automática de hardware e configuração otimizada de processamento paralelo.
2. Varredura hierárquica dos diretórios (train, dev, eval) para localizar arquivos EDF válidos.
3. Leitura robusta dos traçados EEG, padronização dos canais e segmentação em janelas fixas
   com sobreposição.
4. Extração de um conjunto abrangente de features, incluindo:

   * Espectrais (PSD normalizada por bandas clássicas)
   * Estatísticas (média, variância, quantis, derivadas)
   * Complexidade (autovalores, variância de correlação, Higuchi FD)
   * Parâmetros de Hjorth
   * Entropias (permutação, amostral e espectral)
   * Métricas inter-canais (correlações)
5. Salvamento incremental em formato Parquet, com sistema de checkpoints para permitir
   retomada segura após interrupções.
6. Gerenciamento de memória e coleta de lixo para suportar extração em larga escala.

O objetivo é gerar um dataset estruturado e otimizado de features numéricas para treinar,
validar e testar modelos de aprendizagem de máquina aplicados à detecção de convulsões
a partir de sinais EEG.
"""

# ==================== IMPORTS ====================
import os
import gc
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import mne
from scipy.signal import welch
from scipy.stats import skew, kurtosis
import antropy
import nolds

import psutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import warnings

# ==================== CONFIGURAÇÃO DE LOGGING ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)
mne.set_log_level('ERROR')
warnings.filterwarnings('ignore')

# ==================== DETECÇÃO DE HARDWARE ====================
TOTAL_CORES = psutil.cpu_count(logical=False)
TOTAL_THREADS = psutil.cpu_count(logical=True)
TOTAL_RAM_GB = psutil.virtual_memory().total / (1024**3)

print(f"🖥️ Hardware Detectado:")
print(f"  Cores: {TOTAL_CORES} | Threads: {TOTAL_THREADS}")
print(f"  RAM: {TOTAL_RAM_GB:.1f} GB")

# *** CONFIGURAÇÃO OTIMIZADA/PARALELISMO ***
MAX_WORKERS = min(8, TOTAL_THREADS - 2)
RAM_PER_WORKER = 1.5
SAFE_WORKERS = int((TOTAL_RAM_GB * 0.7) / RAM_PER_WORKER)
MAX_WORKERS = min(MAX_WORKERS, SAFE_WORKERS)

CHUNK_SIZE = 2
SAVE_EVERY_N_FILES = 50

print(f"\n⚙️ Configuração Otimizada:")
print(f"  Workers: {MAX_WORKERS}")
print(f"  Chunk Size: {CHUNK_SIZE}")
print(f"  Checkpoint: a cada {SAVE_EVERY_N_FILES} arquivos")

# ==================== PATHS ====================
BASE_PATH = Path(r"C:\Users\herna\OneDrive\Área de Trabalho\TCC2\Base\tuh_eeg_seizure_v2.0.3\edf")
OUTPUT_DIR = Path("./output_v9")
OUTPUT_DIR.mkdir(exist_ok=True)

TRAIN_DIR = BASE_PATH / "train"
DEV_DIR = BASE_PATH / "dev"
EVAL_DIR = BASE_PATH / "eval"

TRAIN_FEATURES_FILE = OUTPUT_DIR / "train_features.parquet"
DEV_FEATURES_FILE = OUTPUT_DIR / "dev_features.parquet"
EVAL_FEATURES_FILE = OUTPUT_DIR / "eval_features.parquet"

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

# ==================== PARÂMETROS ====================
SEGMENT_SECONDS = 60
SAMPLE_RATE = 250
OVERLAP_RATIO = 0.5

BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta': (13, 30),
    'gamma': (30, 70)
}

STANDARD_CHANNELS = [
    'EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF',
    'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF',
    'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG F8-REF',
    'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF',
    'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF',
    'EEG A1-REF', 'EEG A2-REF'
]

# ==================== FUNÇÕES AUXILIARES ====================

def list_edf_files(directory: Path, dir_name: str) -> List[Path]:
    """Encontra arquivos EDF 01_tcp_ar"""
    logger.info(f"Procurando arquivos EDF em: {dir_name}")
    files = list(directory.glob("**/01_tcp_ar/*.edf"))
    logger.info(f"Encontrados {len(files)} arquivos.")
    return files

def save_checkpoint(checkpoint_file: Path, data_file: Path, features_df: pd.DataFrame):
    """Salva checkpoint"""
    if features_df is not None and len(features_df) > 0:
        features_df.to_parquet(data_file, index=False)
        checkpoint_meta = {
            'timestamp': time.time(),
            'total_segments': len(features_df),
            'total_files': features_df['file'].nunique()
        }
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_meta, f, indent=2)
        logger.info(f"✓ Checkpoint: {checkpoint_meta['total_segments']} segmentos de {checkpoint_meta['total_files']} arquivos.")

def load_checkpoint(checkpoint_file: Path, data_file: Path) -> Tuple[set, pd.DataFrame]:
    """Carrega checkpoint"""
    processed_files = set()
    features_df = pd.DataFrame()
    
    if checkpoint_file.exists() and data_file.exists():
        try:
            features_df = pd.read_parquet(data_file)
            processed_files = set(features_df['file'])
            logger.info(f"✓ Checkpoint: {len(features_df)} segmentos de {len(processed_files)} arquivos carregados.")
        except Exception as e:
            logger.warning(f"Erro no checkpoint: {e}")
    
    return processed_files, features_df

# ==================== EXTRAÇÃO DE FEATURES ====================

def calculate_psd_features(data: np.ndarray, fs: int) -> Dict:
    """PSD normalizado conforme tese"""
    features = {}
    nperseg = min(fs * 2, data.shape[1])
    noverlap = nperseg // 2
    
    freqs, psd = welch(data, fs=fs, nperseg=nperseg, noverlap=noverlap, axis=1)
    total_power = np.sum(psd, axis=1, keepdims=True) + 1e-10
    
    for band_name, (low_freq, high_freq) in BANDS.items():
        band_mask = (freqs >= low_freq) & (freqs <= high_freq)
        
        if not np.any(band_mask):
            features[f'p_{band_name}'] = 0.0
            features[f'p_{band_name}_std'] = 0.0
            features[f'p_{band_name}_max'] = 0.0
            features[f'peak_freq_{band_name}'] = 0.0
            features[f'acf_p_{band_name}'] = 0.0
            continue
        
        band_psd = psd[:, band_mask]
        band_freqs = freqs[band_mask]
        band_power_norm = np.sum(band_psd, axis=1) / total_power.squeeze()
        
        features[f'p_{band_name}'] = np.mean(band_power_norm)
        features[f'p_{band_name}_std'] = np.std(band_power_norm)
        features[f'p_{band_name}_max'] = np.max(band_power_norm)
        
        peak_idx = np.argmax(np.mean(band_psd, axis=0))
        features[f'peak_freq_{band_name}'] = float(band_freqs[peak_idx]) if len(band_freqs) > 0 else 0.0
        features[f'acf_p_{band_name}'] = np.var(band_power_norm) if len(band_power_norm) > 1 else 0.0
    
    delta = features.get('p_delta', 1e-10)
    theta = features.get('p_theta', 1e-10)
    alpha = features.get('p_alpha', 1e-10)
    beta = features.get('p_beta', 1e-10)
    
    features['ratio_delta_beta'] = delta / (beta + 1e-10)
    features['ratio_delta_alpha'] = delta / (alpha + 1e-10)
    features['ratio_alpha_theta'] = alpha / (theta + 1e-10)
    
    return features

def calculate_statistical_features(data: np.ndarray) -> Dict:
    """Features estatísticas com D1_Q3 e A_Q3"""
    features = {}
    
    features['mean'] = np.mean(data)
    features['std'] = np.std(data)
    features['var'] = np.var(data)
    features['skewness'] = skew(data.flatten())
    features['kurtosis'] = kurtosis(data.flatten())
    features['Q1'] = np.quantile(data, 0.25)
    features['Q2'] = np.quantile(data, 0.50)
    features['Q3'] = np.quantile(data, 0.75)
    
    d1 = np.diff(data, axis=1)
    features['D1_mean'] = np.mean(d1)
    features['D1_std'] = np.std(d1)
    features['D1_Q1'] = np.quantile(d1, 0.25)
    features['D1_Q2'] = np.quantile(d1, 0.50)
    features['D1_Q3'] = np.quantile(d1, 0.75)
    
    amplitude = np.ptp(data, axis=1)
    features['A_mean'] = np.mean(amplitude)
    features['A_std'] = np.std(amplitude)
    features['A_Q1'] = np.quantile(amplitude, 0.25)
    features['A_Q2'] = np.quantile(amplitude, 0.50)
    features['A_Q3'] = np.quantile(amplitude, 0.75)
    
    return features

def calculate_complexity_features(data: np.ndarray) -> Dict:
    """Features de complexidade com eigenvalues"""
    features = {}
    
    if data.shape[0] > 1:
        corr_matrix = np.corrcoef(data)
        eigenvalues = np.linalg.eigvalsh(corr_matrix)
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        for i in range(min(5, len(eigenvalues))):
            features[f'complexity_ev_{i+1}'] = eigenvalues[i]
        for i in range(len(eigenvalues), 5):
            features[f'complexity_ev_{i+1}'] = 0.0
        
        features['complexity_ev_std'] = np.std(eigenvalues)
        features['complexity_value_std'] = np.std(corr_matrix.flatten())
        features['complexity_value_q_25'] = np.quantile(corr_matrix, 0.25)
        features['complexity_value_q_50'] = np.quantile(corr_matrix, 0.50)
        features['complexity_value_q_75'] = np.quantile(corr_matrix, 0.75)
    else:
        for i in range(1, 6):
            features[f'complexity_ev_{i}'] = 0.0
        features['complexity_ev_std'] = 0.0
        features['complexity_value_std'] = 0.0
        features['complexity_value_q_25'] = 0.0
        features['complexity_value_q_50'] = 0.0
        features['complexity_value_q_75'] = 0.0
    
    mean_channel = np.mean(data, axis=0)
    sample_data = mean_channel[:min(1000, len(mean_channel))]
    try:
        features['higuchi_fd'] = nolds.hfd(sample_data)
    except:
        features['higuchi_fd'] = 0.0
    
    return features

def calculate_hjorth_parameters(data: np.ndarray) -> Dict:
    """Parâmetros de Hjorth"""
    activity = np.var(data, axis=1)
    d1 = np.diff(data, axis=1)
    activity_d1 = np.var(d1, axis=1)
    mobility = np.sqrt(activity_d1 / (activity + 1e-10))
    d2 = np.diff(d1, axis=1)
    activity_d2 = np.var(d2, axis=1)
    mobility_d1 = np.sqrt(activity_d2 / (activity_d1 + 1e-10))
    complexity = mobility_d1 / (mobility + 1e-10)
    
    return {
        'hjorth_activity': np.mean(activity),
        'hjorth_mobility': np.mean(mobility),
        'hjorth_complexity': np.mean(complexity),
    }

def calculate_entropy_features(data: np.ndarray) -> Dict:
    """Features de entropia"""
    features = {}
    sample_data = np.mean(data, axis=0)[:min(1000, data.shape[1])]
    
    try:
        features['perm_entropy'] = antropy.perm_entropy(sample_data, normalize=True)
        features['sample_entropy'] = antropy.sample_entropy(sample_data)
        features['spectral_entropy'] = antropy.spectral_entropy(sample_data, sf=SAMPLE_RATE, method='welch')
    except:
        features['perm_entropy'] = 0.0
        features['sample_entropy'] = 0.0
        features['spectral_entropy'] = 0.0
    
    return features

def calculate_interchannel_features(data: np.ndarray, fs: int) -> Dict:
    """Features inter-canal"""
    features = {}
    if data.shape[0] < 2:
        features['inter_corr_mean'] = 0.0
        features['inter_corr_std'] = 0.0
        return features
    
    corr_matrix = np.corrcoef(data)
    upper_triangle = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
    features['inter_corr_mean'] = np.mean(upper_triangle)
    features['inter_corr_std'] = np.std(upper_triangle)
    
    return features

def extract_all_features(segment_data: np.ndarray, fs: int) -> Dict:
    """Extrai todas as features"""
    all_features = {}
    try:
        all_features.update(calculate_psd_features(segment_data, fs))
        all_features.update(calculate_complexity_features(segment_data))
        all_features.update(calculate_statistical_features(segment_data))
        all_features.update(calculate_hjorth_parameters(segment_data))
        all_features.update(calculate_entropy_features(segment_data))
        all_features.update(calculate_interchannel_features(segment_data, fs))
    except Exception as e:
        logger.warning(f"Erro extraindo features: {e}")
        all_features['error'] = 1
    return all_features

# ==================== PARSE SEIZURE LABELS ====================

def parse_seizure_labels(csv_bi_path: Path, seg_start: float, seg_end: float) -> int:
    """Parse .csv_bi para labels"""
    if not csv_bi_path.exists():
        return 0
    
    try:
        data = np.loadtxt(
            csv_bi_path,
            usecols=(1, 2, 3),
            dtype=object,
            comments=None,
            skiprows=6,
            ndmin=2,
            delimiter=',',
            encoding='utf-8'
        )
        
        seizure_intervals = []
        for row in data:
            if str(row[2]) == 'seiz':
                seizure_intervals.append([float(row[0]), float(row[1])])
        
        if not seizure_intervals:
            return 0
        
        seizure_intervals = np.array(seizure_intervals)
        for onset, offset in seizure_intervals:
            if offset > seg_start and onset < seg_end:
                return 1
        
        return 0
    except:
        return 0

# ==================== PROCESSAMENTO DE ARQUIVO ====================

def process_single_file(args: Tuple[str, Path]) -> Optional[List[Dict]]:
    """Processa um arquivo EDF"""
    split, file_path = args
    
    try:
        parts = file_path.parts
        patient_id = parts[-4] if len(parts) > 4 else "unknown"
        
        with mne.io.read_raw_edf(file_path, preload=False, verbose=False) as raw:
            raw.load_data()
        
        available_channels = [ch for ch in STANDARD_CHANNELS if ch in raw.ch_names]
        
        if len(available_channels) < 10:
            return (None, {"status": "SKIPPED_CHANNELS", "file": file_path.name})
        
        raw.pick_channels(available_channels, ordered=True)
        
        if raw.info['sfreq'] != SAMPLE_RATE:
            raw.resample(SAMPLE_RATE, verbose=False)
        
        raw.filter(l_freq=0.5, h_freq=70.0, verbose=False)
        raw.notch_filter(freqs=60.0, verbose=False)
        
        data = raw.get_data()
        data = (data - np.mean(data, axis=1, keepdims=True)) / (np.std(data, axis=1, keepdims=True) + 1e-10)
        
        fs = int(raw.info['sfreq'])
        samples_per_segment = SEGMENT_SECONDS * fs
        hop_size = int(samples_per_segment * (1 - OVERLAP_RATIO))
        
        csv_bi_path = file_path.with_suffix('.csv_bi')
        
        segments = []
        n_samples = data.shape[1]
        
        for start_idx in range(0, n_samples - samples_per_segment + 1, hop_size):
            end_idx = start_idx + samples_per_segment
            start_sec = start_idx / fs
            end_sec = end_idx / fs
            
            segment_data = data[:, start_idx:end_idx]
            features = extract_all_features(segment_data, fs)
            label = parse_seizure_labels(csv_bi_path, start_sec, end_sec)
            
            features['split'] = split
            features['patient_id'] = patient_id
            features['file'] = file_path.name
            features['start_time'] = start_sec
            features['label'] = label
            
            segments.append(features)
        
        # Garbage collection
        gc.collect()
        
        return (segments, {"status": "OK", "file": file_path.name})
    
    except Exception as e:
        return (None, {"status": "ERROR", "file": file_path.name, "error": str(e)})

# ==================== PIPELINE ====================

def run_extraction_pipeline(directory_path: Path, output_file: Path):
    """Pipeline & Monitoramento"""
    dir_name = directory_path.name.upper()
    logger.info(f"\n{'='*70}")
    logger.info(f" EXTRAÇÃO - {dir_name}")
    logger.info(f"{'='*70}")
    
    CHECKPOINT_FILE = CHECKPOINT_DIR / f"{dir_name.lower()}_checkpoint.json"
    CHECKPOINT_DATA_FILE = CHECKPOINT_DIR / f"{dir_name.lower()}_checkpoint_data.parquet"
    
    processed_files_set, df_checkpoint = load_checkpoint(CHECKPOINT_FILE, CHECKPOINT_DATA_FILE)
    all_segments_features = df_checkpoint.to_dict('records') if len(df_checkpoint) > 0 else []
    
    all_files_to_scan = list_edf_files(directory_path, dir_name)
    files_to_process_paths = [f for f in all_files_to_scan if f.name not in processed_files_set]
    files_to_process = [(dir_name.lower(), f) for f in files_to_process_paths]
    
    if not files_to_process:
        if len(df_checkpoint) > 0:
            logger.info(f"✅ Todos os arquivos processados. Salvando final...")
            df_checkpoint.to_parquet(output_file, index=False)
            if CHECKPOINT_FILE.exists(): os.remove(CHECKPOINT_FILE)
            if CHECKPOINT_DATA_FILE.exists(): os.remove(CHECKPOINT_DATA_FILE)
        return True
    
    logger.info(f"📂 Total: {len(all_files_to_scan)} | Processados: {len(processed_files_set)} | Restantes: {len(files_to_process)}")
    
    new_segments_this_session = []
    files_processed_this_session = 0
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        
        results_map = executor.map(process_single_file, files_to_process, chunksize=CHUNK_SIZE)
        
        logger.info(f"🚀 Processamento paralelo iniciado ({MAX_WORKERS} workers)...")
        
        for segments_list, summary in tqdm(results_map, total=len(files_to_process), desc=f"{dir_name}", unit="arquivo"):
            
            if summary['status'] == 'OK' and segments_list:
                new_segments_this_session.extend(segments_list)
                processed_files_set.add(summary['file'])
                files_processed_this_session += 1
            
            if files_processed_this_session > 0 and files_processed_this_session % SAVE_EVERY_N_FILES == 0:
                logger.info(f"💾 Checkpoint ({files_processed_this_session} arquivos)...")
                df_new = pd.DataFrame(new_segments_this_session)
                df_to_save = pd.concat([df_checkpoint, df_new], ignore_index=True)
                save_checkpoint(CHECKPOINT_FILE, CHECKPOINT_DATA_FILE, df_to_save)
                df_checkpoint = df_to_save
                new_segments_this_session = []
                gc.collect()
    
    logger.info(f"✅ {files_processed_this_session} arquivos processados.")
    
    if new_segments_this_session:
        df_new = pd.DataFrame(new_segments_this_session)
        df_final = pd.concat([df_checkpoint, df_new], ignore_index=True)
    elif len(df_checkpoint) > 0:
        df_final = df_checkpoint
    else:
        logger.error(f"❌ Nenhum segmento extraído de {dir_name}.")
        return False
    
    df_final = df_final.fillna(0)
    df_final.to_parquet(output_file, index=False)
    logger.info(f"💾 Salvo: {output_file} ({len(df_final):,} segmentos)")
    
    if CHECKPOINT_FILE.exists(): os.remove(CHECKPOINT_FILE)
    if CHECKPOINT_DATA_FILE.exists(): os.remove(CHECKPOINT_DATA_FILE)
    
    return True

# ==================== MAIN ====================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print(" PIPELINE PARALELO OTIMIZADO - v9")
    print("="*70)
    
    start_total_time = time.time()
    
    # TRAIN
    if not TRAIN_FEATURES_FILE.exists():
        logger.info(f"\n📊 Processando TRAIN...")
        run_extraction_pipeline(TRAIN_DIR, TRAIN_FEATURES_FILE)
    else:
        logger.info(f"\n✅ TRAIN já processado: {TRAIN_FEATURES_FILE}")
    
    # DEV
    if not DEV_FEATURES_FILE.exists():
        logger.info(f"\n📊 Processando DEV...")
        run_extraction_pipeline(DEV_DIR, DEV_FEATURES_FILE)
    else:
        logger.info(f"\n✅ DEV já processado: {DEV_FEATURES_FILE}")
    
    # EVAL
    if not EVAL_FEATURES_FILE.exists():
        logger.info(f"\n📊 Processando EVAL...")
        run_extraction_pipeline(EVAL_DIR, EVAL_FEATURES_FILE)
    else:
        logger.info(f"\n✅ EVAL já processado: {EVAL_FEATURES_FILE}")
    
    end_total_time = time.time()
    total_hours = (end_total_time - start_total_time) / 3600
    
    print("\n" + "="*70)
    print(" PIPELINE CONCLUÍDO")
    print("="*70)
    print(f"  Tempo total: {total_hours:.2f}h ({(end_total_time - start_total_time)/60:.1f} min)")
    print(f"  Arquivos gerados:")
    print(f"    ✓ {TRAIN_FEATURES_FILE}")
    print(f"    ✓ {DEV_FEATURES_FILE}")
    print(f"    ✓ {EVAL_FEATURES_FILE}")