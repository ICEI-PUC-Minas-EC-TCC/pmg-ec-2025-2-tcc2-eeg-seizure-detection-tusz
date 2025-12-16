
# Dicionário de Dados  
### Aplicação de Aprendizado de Máquina na Análise de Sinais de EEG para Detecção de Convulsões  


## Introdução

Este documento fornece uma descrição técnica e formal de todas as variáveis, parâmetros, características extraídas e saídas utilizadas no pipeline computacional para detecção automática de convulsões em sinais de eletroencefalograma (EEG). O dicionário é organizado por categorias funcionais.

---

## 1. Estrutura de Dados da Base TUSZ

### 1.1 Identificadores e Metadados Demográficos

| Variável | Tipo | Descrição | Intervalo/Valores | Notas |
|----------|------|-----------|-------------------|-------|
| `pid` | String | Identificador único do paciente no corpus TUSZ | Ex: "00000045", "00000674" | Composto por números zero-padronizados |
| `patient_id` | Integer | Identificador numérico equivalente ao pid | 0 a 674 | Convertido de pid para uso computacional |
| `age` | Float | Idade do paciente em anos no momento da gravação | 0.0 a 95.0+ | Valores ausentes tratados como NaN |
| `gender` | String | Sexo biológico do paciente | "M" (masculino), "F" (feminino) | Variável categórica |
| `session_id` | String | Identificador único da sessão de EEG para um paciente | Ex: "s001", "s010", "s024" | Múltiplas sessões por paciente são permitidas |
| `record_id` | String | Identificador composto único para cada registro EEG | Concatenação: pid + "_" + session_id | Ex: "00000045_s001" |
| `seizure_label` | Integer | Indicador binário de presença de convulsão no registro | 0 (sem convulsão), 1 (com convulsão) | Rótulo no nível do registro completo |
| `seizure_count` | Integer | Número total de eventos de convulsão detectados no registro | ≥ 0 | Pode haver múltiplas convulsões em um registro |

### 1.2 Características Técnicas do Sinal EEG

| Variável | Tipo | Descrição | Valores Típicos | Unidade |
|----------|------|-----------|-----------------|--------|
| `num_channels` | Integer | Número total de canais de EEG registrados | 19, 20, 21 | Contagem |
| `channel_names` | List[String] | Nomes padronizados dos eletrodos (montagem 01_tcp_ar) | "Fp1", "Fp2", "F3", "F4", ..., "Pz" | Nomenclatura 10-20 com referência média |
| `sampling_rate_original` | Integer | Taxa de amostragem original do equipamento de EEG | 250, 256, 500 Hz | Hz (amostras por segundo) |
| `sampling_rate_normalized` | Integer | Taxa de amostragem após normalização para uso consistente | 250 Hz | Hz |
| `duration_seconds` | Float | Duração total do registro de EEG | 600 a 3600+ segundos | Segundos |
| `total_samples` | Integer | Número total de amostras por canal = duration_seconds × sampling_rate_normalized | 150.000 a 900.000+ | Contagem |


---

## 2. Variáveis de Pré-processamento

### 2.1 Parâmetros de Filtragem

| Parâmetro | Tipo | Descrição | Especificação | Justificativa |
|-----------|------|-----------|---------------|---------------|
| `lowcut_hz` | Float | Frequência de corte inferior do filtro passa-banda Butterworth | 0.5 Hz | Remove componentes DC e flutuações lenta de muito baixa frequência |
| `highcut_hz` | Float | Frequência de corte superior do filtro passa-banda Butterworth | 70.0 Hz | Remove ruído de alta frequência e artefatos de movimento |
| `filter_order` | Integer | Ordem do filtro Butterworth (número de pólos) | 4 | Equilíbrio entre seletividade e estabilidade numérica |
| `notch_frequency` | Float | Frequência de rejeição do filtro notch | 60.0 Hz | Remove interferência da linha de força (60 Hz na América do Norte) |
| `notch_bandwidth` | Float | Largura de banda de rejeição do filtro notch | 2.0 Hz | Q = notch_frequency / notch_bandwidth = 30 |

### 2.2 Parâmetros de Segmentação Temporal

| Parâmetro | Tipo | Descrição | Valor | Justificativa |
|-----------|------|-----------|-------|---------------|
| `window_length_seconds` | Integer | Duração de cada janela de análise | 60 segundos | Captura contexto temporal suficiente para padrões de convulsão |
| `window_overlap_percent` | Float | Percentual de sobreposição entre janelas consecutivas | 50 (0.5) | Reduz efeitos de borda e aumenta densidade de amostras |
| `window_stride_seconds` | Integer | Deslocamento temporal entre centros de janelas consecutivas | 30 segundos | Calculado como window_length_seconds × (1 - window_overlap_percent) |
| `samples_per_window` | Integer | Número de amostras por janela | 15.000 (60 seg × 250 Hz) | Determinado por window_length_seconds × sampling_rate_normalized |

### 2.3 Variáveis Derivadas de Normalização

| Variável | Tipo | Descrição | Método de Cálculo | Aplicação |
|----------|------|-----------|-------------------|-----------|
| `signal_mean_per_channel` | Float | Média aritmética do sinal por canal | μ = (1/N) × Σ x_i | Valor subtraído em z-score |
| `signal_std_per_channel` | Float | Desvio padrão do sinal por canal | σ = √[(1/N) × Σ(x_i - μ)²] | Divisor em z-score |
| `normalized_signal` | np.ndarray | Sinal normalizado usando transformação z-score | x_normalized = (x - μ) / σ | Entrada para extração de características |

---

## 3. Variáveis de Características Extraídas

### 3.1 Características Espectrais (Bandas de Frequência)

As características espectrais são calculadas usando a Densidade Espectral de Potência (PSD) via método de Welch.

#### 3.1.1 Bandas de Frequência Padrão em EEG

| Banda | Intervalo de Frequência | Interpretação Clínica |
|-------|-------------------------|----------------------|
| Delta (δ) | 0.5 - 4 Hz | Atividade lenta, associada a sono profundo e algumas condições patológicas |
| Theta (θ) | 4 - 8 Hz | Atividade frontal, associada a sonolência e processamento emocional |
| Alpha (α) | 8 - 13 Hz | Ritmo de repouso com olhos fechados, maior amplitude em regiões occipitais |
| Beta (β) | 13 - 30 Hz | Atividade rápida, associada a estados de alerta e processamento motor |
| Gamma (γ) | 30 - 70 Hz | Atividade muito rápida, associada a processamento cognitivo e sincronização neural |

#### 3.1.2 Variáveis Espectrais por Canal

| Variável | Tipo | Descrição Técnica | Fórmula | Intervalo |
|----------|------|-------------------|---------|-----------|
| `p_delta` | Float | Potência relativa normalizada na banda delta | P_delta = (∑ PSD[0.5-4Hz]) / (∑ PSD[total]) × 100 | [0.0, 100.0] |
| `p_theta` | Float | Potência relativa normalizada na banda theta | P_theta = (∑ PSD[4-8Hz]) / (∑ PSD[total]) × 100 | [0.0, 100.0] |
| `p_alpha` | Float | Potência relativa normalizada na banda alpha | P_alpha = (∑ PSD[8-13Hz]) / (∑ PSD[total]) × 100 | [0.0, 100.0] |
| `p_beta` | Float | Potência relativa normalizada na banda beta | P_beta = (∑ PSD[13-30Hz]) / (∑ PSD[total]) × 100 | [0.0, 100.0] |
| `p_gamma` | Float | Potência relativa normalizada na banda gamma | P_gamma = (∑ PSD[30-70Hz]) / (∑ PSD[total]) × 100 | [0.0, 100.0] |
| `log_power_delta` | Float | Logaritmo natural da potência absoluta na banda delta | log_P_delta = ln(∑ PSD[0.5-4Hz] + 1e-10) | ℝ |
| `log_power_theta` | Float | Logaritmo natural da potência absoluta na banda theta | log_P_theta = ln(∑ PSD[4-8Hz] + 1e-10) | ℝ |
| `log_power_alpha` | Float | Logaritmo natural da potência absoluta na banda alpha | log_P_alpha = ln(∑ PSD[8-13Hz] + 1e-10) | ℝ |
| `log_power_beta` | Float | Logaritmo natural da potência absoluta na banda beta | log_P_beta = ln(∑ PSD[13-30Hz] + 1e-10) | ℝ |
| `log_power_gamma` | Float | Logaritmo natural da potência absoluta na banda gamma | log_P_gamma = ln(∑ PSD[30-70Hz] + 1e-10) | ℝ |

**Parâmetros do Método de Welch:**
- `nperseg`: min(250 × 2, número de amostras na janela) = 500 amostras
- `noverlap`: 50% de nperseg = 250 amostras
- `window`: "hann" (janela de Hann)

### 3.2 Características Estatísticas Temporais

| Variável | Tipo | Descrição | Fórmula Matemática | Intervalo |
|----------|------|-----------|-------------------|-----------|
| `mean` | Float | Média aritmética do sinal na janela | μ = (1/N) × Σ x_i | ℝ |
| `std` | Float | Desvio padrão do sinal na janela | σ = √[(1/N) × Σ(x_i - μ)²] | [0, ∞) |
| `var` | Float | Variância do sinal na janela | σ² | [0, ∞) |
| `min_val` | Float | Valor mínimo observado no sinal | min_val = min(x_i) | ℝ |
| `max_val` | Float | Valor máximo observado no sinal | max_val = max(x_i) | ℝ |
| `range` | Float | Amplitude total (diferença pico a pico) | range = max_val - min_val | [0, ∞) |
| `skewness` | Float | Assimetria da distribuição (terceiro momento normalizado) | γ = (1/N) × Σ[(x_i - μ)/σ]³ | (-∞, ∞) |
| `kurtosis` | Float | Curtose (quarto momento normalizado) | κ = [(1/N) × Σ[(x_i - μ)/σ]⁴] - 3 | (-3, ∞) |
| `q25` | Float | Quartil 25% (percentil 0.25) | Valor abaixo do qual 25% dos dados caem | ℝ |
| `q50` | Float | Mediana (percentil 0.50) | Valor do meio da distribuição | ℝ |
| `q75` | Float | Quartil 75% (percentil 0.75) | Valor abaixo do qual 75% dos dados caem | ℝ |
| `iqr` | Float | Intervalo interquartil | IQR = q75 - q25 | [0, ∞) |

### 3.3 Características Não-Lineares

Medem complexidade, regularidade e auto-similaridade do sinal, características importantes para diferenciar atividade normal de convulsões.

| Variável | Tipo | Descrição | Parâmetros | Interpretação |
|----------|------|-----------|------------|---------------|
| `permutation_entropy` | Float | Entropia baseada em padrões ordinais (permutações) de subsequências | m=3 (ordem da incorporação), delay=1 (atraso temporal) | Varia de 0 (altamente previsível/periódico) a 1 (aleatório). Sinais de convulsão tendem a mostrar alterações neste valor |
| `sample_entropy` | Float | Medida de regularidade: probabilidade de que padrões semelhantes não se repitam conforme o comprimento aumenta | m=2 (dimensão incorporada), r=0.2×σ (tolerância de semelhança) | Valores baixos indicam alta regularidade (repetição de padrões). Sinais de convulsão podem apresentar maior regularidade |
| `higuchi_fractal_dimension` | Float | Dimensão fractal que quantifica a auto-similaridade do sinal em diferentes escalas | Kmax=10 (número máximo de escalas) | Mede complexidade estrutural. Valores mais altos indicam maior complexidade temporal |
| `hurst_exponent` | Float | Expoente que caracteriza a persistência/antipersistência de sequências de valores | Calculado via análise de variância rescalonada (R/S) | H < 0.5: antipersistente (reversão à média), H = 0.5: aleatório, H > 0.5: persistente (tendências). Útil para diferenciar dinâmicas de convulsão |

### 3.4 Características Espaciais Inter-canal

Capturam correlações e dependências entre diferentes canais de EEG, refletindo conectividade cerebral.

| Variável | Tipo | Descrição | Dimensão | Propriedades |
|----------|------|-----------|----------|-------------|
| `correlation_matrix` | np.ndarray (float) | Matriz de correlação de Pearson entre todos os pares de canais | (n_channels, n_channels) = (21, 21) | Simétrica, diagonal = 1.0, valores em [-1, 1] |
| `covariance_matrix` | np.ndarray (float) | Matriz de covariância entre canais | (n_channels, n_channels) | Simétrica, positiva semi-definida |
| `eigenvalues` | np.ndarray (float) | Autovalores da matriz de correlação | (n_channels,) = (21,) | Ordenados decrescentemente. Representam variância explicada por cada componente principal |
| `eigenvector_1_norm` | Float | Norma (L2) do primeiro autovetor da matriz de correlação | Valor singular | Mede coerência global: valores elevados indicam alta sincronização entre canais |
| `trace_correlation_matrix` | Float | Traço da matriz de correlação (soma dos autovalores) | = Σ λ_i | Sempre igual a n_channels (propriedade das matrizes de correlação). Usado para normalização |
| `spatial_connectivity` | Float | Conectividade média: média de todas as correlações pairwise (excluindo diagonal) | = (1 / [n_channels × (n_channels-1)]) × Σ |r_ij| | [0, 1]. Quantifica sincronização global do EEG |
| `spectral_power_sum` | Float | Soma das potências espectrais de todos os canais | Σ_canais (∑_frequências PSD) | Medida de amplitude geral do sinal |

### 3.5 Parâmetros de Hjorth

Descritores temporais que caracterizam as propriedades dinâmicas do sinal no domínio do tempo.

| Variável | Tipo | Descrição | Fórmula | Intervalo |
|----------|------|-----------|---------|-----------|
| `hjorth_activity` | Float | Atividade: variância do sinal, proporcionalmente semelhante a potência | Activity = var(x) = σ²_x | [0, ∞) |
| `hjorth_mobility` | Float | Mobilidade: raiz quadrada da razão entre variância da primeira derivada e variância do sinal | Mobility = √[var(dx) / var(x)] | [0, ∞). Proporcional à frequência média |
| `hjorth_complexity` | Float | Complexidade: razão entre a mobilidade da primeira derivada e a mobilidade do sinal | Complexity = Mobility(dx) / Mobility(x) = √[var(d²x) / var(dx)] / √[var(dx) / var(x)] | [0, ∞). Medida de mudança de frequência |

**Propriedades:**
- Descritores sem dimensionalidade explícita, normalizados internamente
- Particularmente úteis para discriminar entre diferentes estados cerebrais

---

## 4. Variáveis de Configuração dos Modelos

### 4.1 K-Nearest Neighbors (KNN)

| Parâmetro | Tipo | Descrição | Valor Utilizado | Intervalo/Opcões |
|-----------|------|-----------|-----------------|------------------|
| `n_neighbors` | Integer | Número de vizinhos mais próximos consultados para classificação | 20 | Inteiro positivo |
| `metric` | String | Métrica de distância para cálculo de proximidade | "euclidean" | "euclidean", "manhattan", "minkowski", etc. |
| `weights` | String | Ponderação dos vizinhos na classificação | "uniform" | "uniform" (todos pesam igual) ou "distance" (inverso da distância) |
| `algorithm` | String | Algoritmo de busca para encontrar vizinhos | "auto" | "auto", "ball_tree", "kd_tree", "brute" |
| `n_jobs` | Integer | Número de processadores para paralelização | -1 (todos os cores disponíveis) | -1 ou inteiro positivo |
| `leaf_size` | Integer | Tamanho de folha para estruturas de árvore | 30 (padrão) | Inteiro positivo |

### 4.2 Support Vector Machine (SVM)

| Parâmetro | Tipo | Descrição | Valor Utilizado | Justificativa |
|-----------|------|-----------|-----------------|---------------|
| `kernel` | String | Tipo de função kernel para transformação de espaço | "rbf" (Radial Basis Function) | RBF captura relações não-lineares complexas nos dados de EEG |
| `C` | Float | Parâmetro de regularização (inverso da força de regularização) | 10.0 | Maior C = menor regularização, maior flexibilidade |
| `gamma` | String | Coeficiente do kernel RBF: influencia raio de influência de cada ponto | "scale" (1 / n_features) | Controla alcance de influência de exemplos individuais |
| `probability` | Boolean | Habilitar estimativa de probabilidade de classe | True | Necessário para obter y_pred_proba |
| `random_state` | Integer | Seed para reprodutibilidade | 42 | Garante resultados consistentes entre execuções |
| `max_iter` | Integer | Número máximo de iterações | -1 (ilimitado) | |

### 4.3 Random Forest (RF)

| Parâmetro | Tipo | Descrição | Valor Utilizado | Justificativa |
|-----------|------|-----------|-----------------|---------------|
| `n_estimators` | Integer | Número de árvores de decisão no ensemble | 500 | Número suficiente para capturar padrões complexos sem overfitting excessivo |
| `max_depth` | Integer | Profundidade máxima das árvores | 15 | Limita complexidade de cada árvore, reduz overfitting |
| `min_samples_split` | Integer | Número mínimo de amostras para dividir um nó interno | 2 (padrão) | Valores maiores reduzem overfitting |
| `min_samples_leaf` | Integer | Número mínimo de amostras em cada folha | 1 (padrão) | Requer pelo menos esta quantidade em folhas |
| `max_features` | Float | Fração de features consideradas para cada divisão | 0.1 (10% das features) | Reduz correlação entre árvores, melhora generalização |
| `bootstrap` | Boolean | Usar amostras com reposição (bootstrap) para treino | True | Cria diversidade entre árvores |
| `n_jobs` | Integer | Número de processadores para paralelização | -1 (todos) | Acelera treinamento |
| `random_state` | Integer | Seed para reprodutibilidade | 42 | Garante resultados consistentes |

---

## 5. Estrutura de Dados de Entrada (Features)

### 5.1 DataFrame de Características Extraídas

O arquivo de features gerado (`features_train.csv`, `features_dev.csv`, `features_eval.csv`) contém uma linha por janela de EEG.

| Coluna | Tipo | Descrição | Valores |
|--------|------|-----------|---------|
| `pid` | String | Identificador do paciente | Ex: "00000045" |
| `record_id` | String | Identificador único do registro | Ex: "00000045_s001" |
| `window_id` | Integer | Número sequencial da janela no registro | 0, 1, 2, ... |
| `time_start_sec` | Float | Tempo de início da janela em relação ao início do registro | ≥ 0 |
| `time_end_sec` | Float | Tempo de término da janela | > time_start_sec |
| `label` | Integer | Rótulo de convulsão para esta janela | 0 (sem convulsão), 1 (com convulsão) |
| `channel` | String | Nome do eletrodo (canal) | "Fp1", "Fp2", ..., "Pz" |
| `dataset_split` | String | Conjunto de dados a que pertence | "train", "dev", "eval" |

### 5.2 Colunas de Características (por variável descrita em Seção 3)

**Espectrais (10 colunas por canal):**
`p_delta`, `p_theta`, `p_alpha`, `p_beta`, `p_gamma`, `log_power_delta`, `log_power_theta`, `log_power_alpha`, `log_power_beta`, `log_power_gamma`

**Estatísticas (12 colunas por canal):**
`mean`, `std`, `var`, `min_val`, `max_val`, `range`, `skewness`, `kurtosis`, `q25`, `q50`, `q75`, `iqr`

**Não-lineares (4 colunas por canal):**
`permutation_entropy`, `sample_entropy`, `higuchi_fd`, `hurst_exponent`

**Hjorth (3 colunas por canal):**
`hjorth_activity`, `hjorth_mobility`, `hjorth_complexity`

**Espaciais (calculadas uma vez por janela, não por canal):**
`spatial_connectivity`, `eigenvector_1_norm`, `trace_corr_matrix`

**Total de features por registro:** (10 + 12 + 4 + 3) × 21 canais + 3 features espaciais = 610 features

---

## 6. Variáveis de Saída e Resultados

### 6.1 Predições do Modelo

| Variável | Tipo | Descrição | Valores | Significado |
|----------|------|-----------|---------|------------|
| `y_pred` | np.ndarray (int) | Predição binária obtida do classificador | {0, 1} | 0 = sem convulsão, 1 = com convulsão |
| `y_pred_proba` | np.ndarray (float) | Probabilidade estimada de cada classe | [0.0, 1.0] | Confiança da predição |
| `y_pred_proba_class_0` | Float | Probabilidade da classe negativa (sem convulsão) | [0.0, 1.0] | P(y=0 \| x) |
| `y_pred_proba_class_1` | Float | Probabilidade da classe positiva (convulsão) | [0.0, 1.0] | P(y=1 \| x), usada para agregação |

### 6.2 Métricas de Desempenho em Nível de Janela

| Métrica | Tipo | Fórmula | Intervalo | Interpretação |
|---------|------|---------|-----------|---------------|
| `accuracy_window` | Float | (TP + TN) / (TP + TN + FP + FN) | [0.0, 1.0] | Proporção de predições corretas |
| `precision_window` | Float | TP / (TP + FP) | [0.0, 1.0] | De predições positivas, quantas estão corretas |
| `recall_window` | Float | TP / (TP + FN) | [0.0, 1.0] | De convulsões reais, quantas foram detectadas (Sensibilidade) |
| `f1_score_window` | Float | 2 × (precision × recall) / (precision + recall) | [0.0, 1.0] | Média harmônica de precisão e recall |
| `specificity_window` | Float | TN / (TN + FP) | [0.0, 1.0] | De ausências reais, quantas foram corretamente identificadas |
| `roc_auc_window` | Float | Área sob a curva ROC | [0.0, 1.0] | Capacidade discriminativa em diferentes thresholds |

### 6.3 Metricas de Desempenho em Nível de Paciente

Agregação de predições de todas as janelas de um paciente usando média de probabilidades.

| Métrica | Tipo | Fórmula | Intervalo |
|---------|------|---------|-----------|
| `patient_proba_mean` | Dict | { pid: mean(y_pred_proba[janelas_paciente]) } | [0.0, 1.0] |
| `patient_proba_std` | Dict | { pid: std(y_pred_proba[janelas_paciente]) } | [0.0, ∞) |
| `patient_prediction` | Integer | 1 se patient_proba_mean > 0.5 else 0 | {0, 1} |
| `accuracy_subject` | Float | Proporção de pacientes corretamente classificados | [0.0, 1.0] |
| `precision_subject` | Float | TP_subj / (TP_subj + FP_subj) | [0.0, 1.0] |
| `recall_subject` | Float | TP_subj / (TP_subj + FN_subj) | [0.0, 1.0] |
| `f1_score_subject` | Float | 2 × (prec_subj × rec_subj) / (prec_subj + rec_subj) | [0.0, 1.0] |
| `roc_auc_subject` | Float | AUC em nível de paciente | [0.0, 1.0] |

**Legenda:**
- **TP** (True Positive): Convulsão corretamente identificada
- **TN** (True Negative): Ausência corretamente identificada
- **FP** (False Positive): Falso alarme (predito convulsão, não havia)
- **FN** (False Negative): Falha crítica (havia convulsão, não foi detectada)

### 6.4 Matriz de Confusão

| Métrica | Tipo | Descrição | Dimensão |
|---------|------|-----------|----------|
| `confusion_matrix` | np.ndarray (int) | Matriz 2×2 de contagem | (2, 2) |
| `cm[0,0]` | Integer | TN (True Negatives) | Valor ≥ 0 |
| `cm[0,1]` | Integer | FP (False Positives) | Valor ≥ 0 |
| `cm[1,0]` | Integer | FN (False Negatives) | Valor ≥ 0 |
| `cm[1,1]` | Integer | TP (True Positives) | Valor ≥ 0 |

---

## 7. Constantes e Configurações Globais

### 7.1 Configurações de Sistema

| Constante | Tipo | Descrição | Valor | Notas |
|-----------|------|-----------|-------|-------|
| `RANDOM_SEED` | Integer | Seed global para reprodutibilidade | 42 | Garante mesmos resultados em múltiplas execuções |
| `N_JOBS` | Integer | Número máximo de processos paralelos | -1 | -1 utiliza todos os cores disponíveis |
| `TEST_SIZE` | Float | Proporção do conjunto de teste (não utilizado neste projeto) | 0.2 | Em futuras divisões |
| `VALIDATION_SIZE` | Float | Proporção do conjunto de validação | 0.1 | Em futuras otimizações de hiperparâmetros |

### 7.2 Bandas de Frequência Padrão

```python
BANDS = {
    'delta':  (0.5, 4),
    'theta':  (4, 8),
    'alpha':  (8, 13),
    'beta':   (13, 30),
    'gamma':  (30, 70)
}
```

### 7.3 Canais EEG Padrão (Montagem 01_tcp_ar)

```python
STANDARD_CHANNELS = [
    'Fp1', 'Fp2',                          # Frontais polares
    'F3', 'F4',                            # Frontais (linha média)
    'C3', 'C4',                            # Centrais
    'P3', 'P4',                            # Parietais
    'O1', 'O2',                            # Occipitais
    'F7', 'F8',                            # Frontais temporais
    'T3', 'T4',                            # Temporais (anteriores)
    'T5', 'T6',                            # Temporais (posteriores)
    'Fz', 'Cz', 'Pz'                       # Linha média (frontal, central, parietal)
]
```

Total: 19 eletrodos padrão + referência média (calculada como média de todos os canais)

---

## 8. Exemplos de Utilização em Código

### 8.1 Carregamento e Acesso de Features

```python
import pandas as pd
import numpy as np

# Carregar arquivo de características
df = pd.read_csv('output/features_eval.csv')

# Acessar variável específica por canal
delta_power = df[df['channel'] == 'Fp1']['p_delta'].values

# Agregação por paciente
patient_features = df.groupby('pid').agg({
    'p_delta': 'mean',
    'p_theta': 'mean',
    'sample_entropy': 'mean',
    'spatial_connectivity': 'first'  # não varia por canal
})

# Obter rótulo por paciente
patient_labels = df.groupby('pid')['label'].max()
```

### 8.2 Predições e Avaliação

```python
from sklearn.metrics import classification_report, confusion_matrix

# Predições em nível de janela
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Agregação em nível de paciente
patient_proba = df_test.groupby('pid').apply(
    lambda g: g['y_pred_proba'].mean()
)
patient_pred = (patient_proba > 0.5).astype(int)

# Avaliação
print(classification_report(patient_labels, patient_pred))
cm = confusion_matrix(patient_labels, patient_pred)
```

---

## 9. Notas Importantes e Considerações

### 9.1 Normalização e Padronização

- Todas as características espectrais são normalizadas em relação à potência total para garantir invariância a mudanças de amplitude geral do sinal.
- Características estatísticas e não-lineares são calculadas sobre sinais já normalizados em z-score.
- Features são **não normalizadas** antes de entrada nos modelos (SVM com kernel RBF realiza normalização interna).

### 9.2 Tratamento de Dados Faltantes

- Valores ausentes ou inválidos em características são tratados como NaN
- Registros com > 20% de dados faltantes são excluídos
- NaNs remanescentes são imputados com a média da feature durante treinamento

### 9.3 Desbalanceamento de Classes

- O corpus TUSZ apresenta desbalanceamento natural: nem todos os pacientes apresentam convulsões
- Proporção aproximada: 65% sem convulsão, 35% com convulsão no nível de sujeito
- **Métrica primária:** F1-Score e Recall (sensibilidade a casos positivos)
- **Métrica secundária:** Especificidade (minimizar falsos alarmes)

### 9.4 Cenário Inter-paciente

- Modelos são treinados no conjunto TRAIN, validados em DEV, e avaliados em EVAL
- Todos os splits separam pacientes inteiros: nenhum paciente aparece em mais de um split
- Garante avaliação realística de generalização para pacientes não vistos

### 9.5 Reprodutibilidade

- `RANDOM_SEED = 42` é definido globalmente em todos os scripts
- Garante resultados idênticos em múltiplas execuções
- Versões de bibliotecas devem ser fixadas em `requirements.txt`

---

## 10. Referências Cruzadas

- **Entrada:** Arquivos EDF do TUSZ v2.0.3
- **Saída Principal:** `features_train.csv`, `features_dev.csv`, `features_eval.csv`
- **Processamento:** Scripts `TCC2_Extract_Model.py` e `TCC2_Treinamento_Metodologia.py`
- **Publicação:** Artigo desenvolvido como requisito para obtenção do título de Engenheiro de Computação. 

---

**Data de Última Atualização:** Dezembro de 2025  
**Versão:** 1.0  
**Projeto:** Aplicação de Aprendizado de Máquina na Análise de Sinais de EEG para Detecção de Convulsões  
**Autores:** Céline Fayad, Hernane Velozo Rosa, Marta Dias Moreira Noronha