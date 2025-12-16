# Aplicação de Aprendizado de Máquina na Análise de Sinais de EEG para Detecção de Convulsões

Implementação de um pipeline computacional para detecção automática de convulsões em sinais de EEG utilizando o Temple University Hospital Seizure Corpus (TUSZ) v2.0.3 em configuração inter-paciente.

## Resumo

Este trabalho desenvolve e avalia um sistema para detecção automática de convulsões em sinais de eletroencefalograma (EEG) utilizando modelos clássicos de aprendizado de máquina. O pipeline incorpora pré-processamento padronizado, extração de características multiescalares e comparação entre três classificadores: K-Nearest Neighbors, Support Vector Machine e Random Forest.

### Resultados Principais

- **SVM**: F1-Score de 91,89%, Acurácia de 85,37% e Recall de 100% em nível de sujeito
- **KNN**: F1-Score de 87,67% com desempenho estável entre conjuntos
- **RF**: F1-Score de 77,42% com limitações na generalização

## Metodologia

### Base de Dados
- Temple University Hospital Seizure Corpus (TUSZ) v2.0.3
- 675 pacientes, 1476 horas de gravações
- Montagem 01_tcp_ar com referência média
- Taxa de amostragem padronizada para 250 Hz

### Pré-processamento
- Filtragem passa-banda (0,5-70 Hz)
- Filtro notch (60 Hz)
- Segmentação em janelas de 60 segundos com 50% de sobreposição
- Padronização usando z-score

### Extração de Características

```python
# Exemplo de extração de características espectrais
def calculate_psd_features(data: np.ndarray, fs: int) -> Dict:
    freqs, psd = welch(data, fs=fs, nperseg=min(fs * 2, data.shape[1]))
    total_power = np.sum(psd, axis=1, keepdims=True) + 1e-10
    
    for band_name, (low_freq, high_freq) in BANDS.items():
        band_mask = (freqs >= low_freq) & (freqs <= high_freq)
        band_psd = psd[:, band_mask]
        band_power_norm = np.sum(band_psd, axis=1) / total_power.squeeze()
        features[f'p_{band_name}'] = np.mean(band_power_norm)
```

Características extraídas:
- **Espectrais**: Densidade espectral de potência nas bandas delta, theta, alpha, beta, gamma
- **Estatísticas**: Média, variância, assimetria, curtose, quartis
- **Não-lineares**: Entropia de permutação, entropia amostral, dimensão fractal de Higuchi
- **Espaciais**: Autovalores da matriz de correlação, conectividade inter-canal
- **Parâmetros de Hjorth**: Atividade, mobilidade, complexidade

### Modelos

```python
# Configuração dos modelos
knn = KNeighborsClassifier(n_neighbors=20, n_jobs=-1)
svm = SVC(kernel='rbf', C=10, probability=True)
rf = RandomForestClassifier(n_estimators=500, max_depth=15, 
                           max_features=0.1, n_jobs=-1, random_state=42)
```

## Estrutura do Projeto

```
eeg-seizure-detection-tusz/
├── src/
│   ├── TCC2_Extract_Model.py               # Pipeline de extração de features
│   └── TCC2_Treinamento_Metodologia.py     # Treinamento e avaliação
├── docs/
│   ├── TCC_II_Final.pdf                    # Artigo final (a ser disponibilizado)
│   └── DESCRICAO_VARIAVEIS.md              # Dicionário de dados (variáveis)
├── data/                                   # Dados TUSZ (obter separadamente, sob autorização)
├── output/                                 # Resultados e modelos
└── requirements.txt
```

## Instalação e Uso




### Requisitos
- Python: 3.10
- Sistema operacional testado: Linux (Ubuntu 22.04) e/ou Windows 11
- Principais dependências: ver `requirements.txt`

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# ou
.\.venv\Scripts\activate   # Windows

pip install -r requirements.txt

```

### Execução
1. **Extrair características**:
```bash
python src/TCC2_Extract_Model.py
```

2. **Treinar e avaliar modelos**:
```bash
python src/TCC2_Treinamento_Metodologia.py
```

### Acesso aos Dados
O Temple University Hospital Seizure Corpus (TUSZ) v2.0.3 deve ser obtido mediante solicitação de acesso individual em: 
[https://isip.piconepress.com/projects/tuh_eeg/](https://isip.piconepress.com/projects/tuh_eeg/)  
_É necessário preencher um formulário de registro e aguardar aprovação para acesso aos dados._

Estrutura esperada dos dados:
```
data/tuh_eeg_seizure_v2.0.3/edf/
├── train/
├── dev/
└── eval/
```

## Fluxo de Processamento

Execução síncrona de múltiplos arquivos EDF, com balanceamento automático de recursos, tolerância a falhas individuais, checkpoints de progresso e execução adaptativa ao hardware.

```python
# Exemplo do pipeline de agregação por paciente
def build_subject_proba(model, X, pid_series):
    probs = model.predict_proba(X)[:, 1]
    df = pd.DataFrame({'pid': pid_series.values, 'p': probs})
    return df.groupby('pid')['p'].mean().to_dict()
```

## Resultados Adquiridos

### Desempenho em Nível de Sujeito (EVAL)
| Modelo | Acurácia | Precisão | Recall | F1-Score | AUC |
|--------|----------|----------|--------|----------|-----|
| RF | 0.6585 | 0.8571 | 0.7059 | 0.7742 | 0.5252 |
| KNN | 0.7805 | 0.8205 | 0.9412 | 0.8767 | 0.5252 |
| SVM | 0.8537 | 0.8500 | 1.0000 | 0.9189 | 0.6429 |

## Dicionário de Dados

Este projeto utiliza um dicionário de dados estruturado para documentar todas as variáveis utilizadas no pipeline. Para consultar a definição completa de todas as variáveis, incluindo tipos, intervalos de valores, fórmulas de cálculo e exemplos práticos, consulte:

[docs/DESCRICAO_VARIAVEIS.md](./docs/DESCRICAO_VARIAVEIS.md)

O dicionário contém as definições de:

- **Metadados dos pacientes**: IDs, idade, sexo, rótulos de convulsão
- **Características dos sinais EEG**: Número de canais, taxa de amostragem, duração
- **Variáveis de pré-processamento**: Parâmetros de filtragem, segmentação, normalização
- **Variáveis de características extraídas**: Espectrais, estatísticas, não-lineares, espaciais e Hjorth
- **Configurações dos modelos**: Hiperparâmetros de KNN, SVM e Random Forest
- **Variáveis de saída e resultados**: Predições, probabilidades, métricas de desempenho
- **Constantes globais**: Bandas de frequência, canais EEG padrão, configurações de sistema

## Contribuições

- Pipeline completo para detecção de convulsões em cenário inter-paciente
- Extração de características multiescalares integrando informações temporais, espectrais, não-lineares e espaciais
- Demonstração da superioridade do SVM em termos de capacidade de generalização
- Estratégia de agregação por paciente que se mostrou determinante para os resultados
- Dicionário de dados detalhado facilitando reprodutibilidade e uso por terceiros

## Trabalhos Futuros

- Direções futuras incluem otimização de hiperparâmetros, incorporação de técnicas de adaptação de domínio, exploração e comparação com arquiteturas de deep learning.

## Citação

```bibtex
@article{fayad2025eeg,
  title={Aplicação de Aprendizado de Máquina na Análise de Sinais de EEG para Detecção de Convulsões},
  author={Fayad, Céline and Rosa, Hernane Velozo and Noronha, Marta Dias Moreira},
  journal={Revista Abakos},
  year={2025}
}
```

## Autores
Pontifícia Universidade Católica de Minas Gerais  
Instituto de Ciências Exatas e Informática  

- **Céline Fayad** - Bacharelanda em Engenharia de Computação
- **Hernane Velozo Rosa** - Bacharelando em Engenharia de Computação  
- **Marta Dias Moreira Noronha** (Orientadora) - Doutoranda em Informática

---

[https://github.com/ICEI-PUC-Minas-EC-TCC/pmg-ec-2025-2-tcc2-eeg-seizure-detection-tusz](https://github.com/ICEI-PUC-Minas-EC-TCC/pmg-ec-2025-2-tcc2-eeg-seizure-detection-tusz)
