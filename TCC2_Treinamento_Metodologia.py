# -*- coding: utf-8 -*-

"""
Autor: Hernane Velozo Rosa
Trabalho de Conclusão de Curso II – Engenharia de Computação – PUC Minas
Data: 30/11/2025

Descrição:
Este programa implementa o pipeline completo de treinamento e avaliação da Metodologia
para detecção de convulsões a partir de sinais EEG processados anteriormente. Ele opera sobre
os arquivos de features gerados no estágio de pré-processamento, realizando:

1. Carregamento padronizado dos conjuntos TRAIN, DEV e EVAL, incluindo:

   * Separação entre metadados e vetores de características numéricas.
   * Garantia de consistência e alinhamento das features entre os splits.

2. Pré-processamento dos vetores:

   * Imputação de valores faltantes com mediana.
   * Padronização (normalização Z-score) baseada exclusivamente no conjunto de treino.

3. Seleção de características:

   * Treinamento de um Random Forest inicial.
   * Seleção automática das features mais relevantes usando SelectFromModel, com limiar
     definido pela importância mediana das árvores.

4. Treinamento de modelos clássicos de classificação:

   * k-Nearest Neighbors (KNN)
   * Support Vector Machine (SVM, kernel RBF)
   * Random Forest (modelo final principal da metodologia)

5. Agregação a nível de paciente (subject-level inference):

   * Conversão de predições por janela em probabilidades médias por paciente.
   * Construção das tabelas de verdade para DEV e EVAL.

6. Otimização do limiar de decisão:

   * Varredura exaustiva de limiares entre 0.05 e 0.95.
   * Seleção do limiar que maximiza o F1-score no conjunto DEV.
   * Salvamento completo da curva de varredura (threshold_scan_DEV.csv).

7. Avaliação final em EVAL:

   * Cálculo de ACC, Precisão, Recall e AUC-ROC por paciente.
   * Geração da curva ROC (roc_subject_eval.csv).

8. Persistência dos resultados:

   * Salvamento dos modelos treinados, features selecionadas e limiar ótimo em models_v9.joblib.

Objetivo:
Produzir modelos de aprendizagem de máquina alinhados ao pipeline metodológico do TCC,
com avaliação rigorosa em nível de paciente e salvamento reprodutível de todos os artefatos
necessários para análise, comparação e futura implantação.
"""


import warnings
warnings.filterwarnings("ignore")

print("=== TCC Treinamento Metodologia 9 ===")

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_curve, roc_auc_score
import joblib

BASE_DIR = Path("./output_v9")
TRAIN = BASE_DIR/"train_features.parquet"
DEV   = BASE_DIR/"dev_features.parquet"
EVAL  = BASE_DIR/"eval_features.parquet"

SAVE = BASE_DIR/"models_v9"
SAVE.mkdir(exist_ok=True)

def load(p):
    df=pd.read_parquet(p).fillna(0)
    meta={'label','patient_id','file','start_time','split'}
    feat=[c for c in df.columns if c not in meta]
    return df[feat].astype(float),df['label'].astype(int),df['patient_id'].astype(str),feat

def truth(pid,y): return pd.DataFrame({'pid':pid,'y':y}).groupby('pid')['y'].max().astype(int).to_dict()
def pred(model,X,pid): return pd.DataFrame({'pid':pid,'p':model.predict_proba(X)[:,1]}).groupby('pid')['p'].mean().to_dict()

print("loading parquet...")
Xtr,Ytr,PTR,F=load(TRAIN)
Xdv,Ydv,PDV,_=load(DEV)
Xev,Yev,PEV,_=load(EVAL)

Xdv=Xdv.reindex(columns=F,fill_value=0)
Xev=Xev.reindex(columns=F,fill_value=0)

print("scaling...")
imp=SimpleImputer(strategy='median')
sc=StandardScaler()

Xtr=pd.DataFrame(sc.fit_transform(imp.fit_transform(Xtr)),columns=F)
Xdv=pd.DataFrame(sc.transform(imp.transform(Xdv)),columns=F)
Xev=pd.DataFrame(sc.transform(imp.transform(Xev)),columns=F)

print("feature selection...")
rf0=RandomForestClassifier(n_estimators=200,max_depth=15,max_features=0.1,n_jobs=-1,random_state=42)
rf0.fit(Xtr,Ytr)
sel=SelectFromModel(rf0,prefit=True,threshold="median")
msk=sel.get_support()
Fx=[f for f,m in zip(F,msk) if m]

Xtr=pd.DataFrame(sel.transform(Xtr),columns=Fx)
Xdv=pd.DataFrame(sel.transform(Xdv),columns=Fx)
Xev=pd.DataFrame(sel.transform(Xev),columns=Fx)

print("training models...")
knn=KNeighborsClassifier(n_neighbors=20,n_jobs=-1)
svm=SVC(kernel='rbf',C=10,probability=True)
rf =RandomForestClassifier(n_estimators=500,max_depth=15,max_features=0.1,n_jobs=-1,random_state=42)

for m in [knn,svm,rf]: m.fit(Xtr,Ytr)

print("subject DEV aggregation...")
TD=truth(PDV,Ydv)
PD=pred(rf,Xdv,PDV)

print("threshold scan...")
best_thr=0;best_f1=-1
rows=[]
for thr in np.linspace(0.05,0.95,181):
    ids=sorted(TD.keys())
    yt=np.array([TD[i] for i in ids])
    yp=np.array([PD[i] for i in ids])
    yb=(yp>=thr).astype(int)
    prec=((yb==1)&(yt==1)).sum()/max((yb==1).sum(),1)
    rec=((yb==1)&(yt==1)).sum()/max((yt==1).sum(),1)
    f1=2*prec*rec/(prec+rec+1e-9)
    rows.append([thr,prec,rec,f1])
    if f1>best_f1: best_f1=f1;best_thr=thr

pd.DataFrame(rows,columns=['thr','prec','rec','f1']).to_csv(SAVE/'threshold_scan_DEV.csv',index=False)

print("subject EVAL aggregation...")
TE=truth(PEV,Yev)
PE=pred(rf,Xev,PEV)
ids=sorted(TE.keys())
yt=np.array([TE[i] for i in ids])
yp=np.array([PE[i] for i in ids])
yb=(yp>=best_thr).astype(int)

acc=float((yb==yt).mean())
prec=((yb==1)&(yt==1)).sum()/max((yb==1).sum(),1)
rec=((yb==1)&(yt==1)).sum()/max((yt==1).sum(),1)
fpr,tpr,_=roc_curve(yt,yp)
auc=roc_auc_score(yt,yp)

pd.DataFrame({'fpr':fpr,'tpr':tpr}).to_csv(SAVE/'roc_subject_eval.csv',index=False)
joblib.dump({'knn':knn,'svm':svm,'rf':rf,'thr':best_thr,'feat':Fx},SAVE/'models_v9.joblib')

print("=== RESULT ===")
print("BEST_THR_DEV =",best_thr)
print("ACC =",acc)
print("PREC =",prec)
print("REC =",rec)
print("AUC =",auc)
print("ok v9-log")
