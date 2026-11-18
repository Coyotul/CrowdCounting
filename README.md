# Sistem de Numărare și Estimare a Densității de Mulțimi

**Materia:** Analiza Conținutului Imaginilor (ACI)  
**Echipă:** Constantin Florin-George & Jugaru Cosmin-Alexandru  

Proiectul vizează estimarea numărului de persoane și generarea hărților de densitate (*Density Maps*) în imagini cu aglomerări mari, abordare în care metodele clasice de detecție bazate pe bounding boxes eșuează din cauza ocluziilor severe și a variațiilor mari de perspectivă.

---

## Metodologie și Set de Date

### Set de Date: ShanghaiTech
Proiectul utilizează benchmark-ul standard **ShanghaiTech** (Zhang et al., CVPR 2016):
- **Part A:** 482 imagini preluate de pe internet, reprezentând scene foarte dense (concerte, evenimente publice), cu rezoluții variate și variație mare de perspectivă.
- **Part B:** 716 imagini preluate din camere de supraveghere stradale din Shanghai, cu rezoluție fixă de $1024 \times 768$ și densitate moderată.

### Generarea Hărților de Densitate Ground-Truth
Fiecare adnotare punctuală $\mathbf{x}_i$ este reprezentată printr-o funcție Dirac convoluționată cu un kernel Gaussian normalizat:
- **Geometry-Adaptive Gaussian Kernel (Part A):** $\sigma_i = \beta \cdot \bar{d}_{k\text{NN}}$, cu $k = 3$ și $\beta = 0.3$. Parametrul $\sigma$ se adaptează local în funcție de densitatea din proximitatea fiecărui cap, păstrând rezoluția spațială în zonele dense din fundal.
- **Fixed Gaussian Kernel (Part B):** kernel Gaussian cu $\sigma = 15.0$ pixeli pe toate punctele, adecvat pentru perspective uniforme.
- **Conservarea masei:** fiecare kernel este normalizat strict pe suportul vizibil din cadrul imaginii, garantând egalitatea dintre integrala hărții de densitate și numărul real de persoane ($\sum_{x,y} D(x, y) = N$).

### Arhitectura Vizată: CSRNet
Arhitectura principală planificată pentru antrenare este **CSRNet** (*Congested Scene Recognition Network*, Li et al., CVPR 2018):
- **Front-end:** primele 10 straturi convoluționale VGG-16 pre-antrenate pe ImageNet (factor de scalare $1/8$).
- **Back-end:** 6 straturi de convoluții dilatate (*dilated convolutions*, rata de dilatare 2) urmate de un strat $1 \times 1$, extinzând câmpul receptiv fără pierdere de rezoluție spațială.

---

## Stadiul Proiectului (Milestones)

- [x] **M1: Setup & Data Preparation**
  - Organizare și verificare dataset ShanghaiTech (Part A & Part B).
  - Audit și curățare adnotări (filtrare puncte invalide/duplicate, aducerea în cadru a punctelor de pe margine, conversie imagini grayscale la 3 canale RGB).
  - Generare hărți de densitate Ground-Truth în format HDF5 (`.h5`) cu conservarea masei.
  - Creare split stratificat Train / Validation (85% / 15%) pe baza cuantilelor de densitate; setul oficial de testare este păstrat separat.
  - Implementare loader PyTorch cu augmentări sincronizate (random crop, random horizontal flip) și downsampling factor 8 cu conservarea masei.
  - Teste unitare automate pentru pipeline-ul de date.
  - Instrument interactiv (Streamlit) pentru inspecția vizuală a hărților de densitate generate.
- [ ] **M2: Model & Training**
  - Implementare arhitectură CSRNet și funcție de pierdere (MSE pe hărți de densitate).
  - Antrenare pe GPU (Google Colab / Kaggle), monitorizare loss de antrenare și validare.
  - Salvare checkpoint-uri și evaluare pe setul de test (metrici MAE și RMSE).
- [ ] **M3: Inference & UI Demo**
  - Pipeline de inferență pe imagini și fluxuri video.
  - Interfață demonstrativă (upload imagine/video, afișare număr estimat și suprapunere heatmap).
- [ ] **M4: Raport Final & Prezentare**
  - Centralizare rezultate cantitative și calitative.
  - Redactare raport tehnic final și pregătire demo.

---

## Structura Proiectului

```text
CrowdCounting/
├── configs/
│   └── dataset_config.yaml      # Parametri configurabili pentru date, kernel-uri și split-uri
├── notebooks/
│   └── data_exploration.ipynb   # Notebook explorare date și verificare pipeline
├── src/
│   └── data/
│       ├── io_utils.py          # Citire adnotări .mat și I/O HDF5 (.h5)
│       ├── cleaning.py          # Audit integritate, curățare coordonate, split stratificat
│       ├── density_map.py       # Generare kernel-uri Gaussiene (fix și adaptiv k-NN)
│       ├── prepare_dataset.py   # Pipeline CLI pentru preprocesare și generare hărți .h5
│       ├── inspect_density_app.py # Instrument interactiv pentru inspecție vizuală
│       └── pytorch_dataset.py   # Dataset PyTorch pentru antrenare (stride 8)
├── tests/
│   └── test_data_pipeline.py    # Teste unitare automate
├── requirements.txt             # Dependențe Python
└── README.md
```

---

## Instrucțiuni de Reproducere (M1)

### 1. Instalare Dependențe
```bash
pip install -r requirements.txt
```

### 2. Preprocesare Date și Generare Hărți de Densitate
Presupunând arhiva ShanghaiTech extrasă în directorul `ShanghaiTech/`:
```bash
python -m src.data.prepare_dataset --config configs/dataset_config.yaml
```
Scriptul generează:
- Hărțile de densitate Ground-Truth în format `.h5` în `data/processed/part_{A,B}/`.
- Manifesturile CSV stratificate în `data/processed/manifests/` (`part_{A,B}_{train,val,test}.csv`).
- Raportul de audit în format JSON (`data/processed/manifests/audit_summary.json`).

### 3. Rulare Teste Automate
```bash
python -m unittest discover -s tests -v
```

### 4. Inspecție Vizuală Hărți de Densitate
Pentru verificarea vizuală a hărților generate (`.h5`) comparativ cu imaginea originală și adnotările Ground-Truth:
```bash
python -m streamlit run src/data/inspect_density_app.py
```
*(Poate fi rulat și direct prin butonul Run din PyCharm.)*

---

## Referințe Bibliografice

1. Zhang, Y., Zhou, D., Chen, S., Gao, S., & Ma, Y. (2016). *Single-Image Crowd Counting via Multi-Column Convolutional Neural Network*. CVPR 2016.
2. Li, Y., Zhang, X., & Chen, D. (2018). *CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes*. CVPR 2018.
