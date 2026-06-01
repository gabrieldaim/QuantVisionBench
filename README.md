# Edge Vision Benchmark

Benchmark para avaliação do impacto da quantização INT8 em modelos YOLO aplicados à detecção de objetos sob diferentes condições de qualidade visual.

## Objetivo

Este projeto tem como objetivo avaliar o impacto da quantização INT8 sobre modelos de detecção de objetos, utilizando o dataset VisDrone e a arquitetura YOLO26n.

Os experimentos foram desenvolvidos para responder questões como:

* Qual a perda de acurácia causada pela quantização INT8?
* Qual o ganho de desempenho obtido após a quantização?
* Como modelos quantizados se comportam em cenários de degradação visual?
* Qual o equilíbrio entre precisão e eficiência computacional em dispositivos com recursos limitados?

---

## Tecnologias Utilizadas

* Python 3.12+
* Ultralytics YOLO26
* OpenVINO
* Google Colab
* CUDA
* PyTorch
* VisDrone Dataset

---

# Estrutura do Projeto

```text
project/
│
├── datasets/
│   ├── VisDrone2019-DET-train.zip
│   └── VisDrone2019-DET-val.zip
│
├── visdrone_raw/
│
├── visdrone_yolo/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   │
│   └── labels/
│       ├── train/
│       └── val/
│
├── visdrone.yaml
│
└── runs/
    └── detect/
        └── yolo26n_visdrone_fp32/
            └── weights/
                ├── best.pt
                ├── last.pt
                └── best_int8_openvino_model/
```

---

# Dataset

## VisDrone

Este projeto utiliza o **VisDrone2019-DET**, um benchmark de detecção de objetos em imagens capturadas por drones, contendo classes como pedestres, carros, vans, caminhões, ônibus, bicicletas e motocicletas.

### Task 1: Object Detection in Images

#### VisDrone-DET Dataset

| Conjunto | Tamanho | Download |
|-----------|----------|----------|
| Train Set | 1.44 GB | [BaiduYun](https://pan.baidu.com/s/1K-JtLnlHw98UuBDrYJvw3A) \| [Google Drive](https://drive.google.com/file/d/1a2oHjcEcwXP8oUF95qiwrqzACb2YlUhn/view?usp=sharing) |
| Validation Set | 0.07 GB | [BaiduYun](https://pan.baidu.com/s/1jdK_dAxRJeF2Xi50IoML1g) \| [Google Drive](https://drive.google.com/file/d/1bxK5zgLn0_L8x276eKkuYA_FzwCIjb59/view?usp=sharing) |
| testset-dev (0.28 GB): [BaiduYun](https://pan.baidu.com/s/1RdRfSWV-1IFK7aWljLU_LQ) | [GoogleDrive](https://drive.google.com/open?id=1PFdW_VFSCfZ_sTSZAGjQdifF_Xd5mf0V) (GT avalialbe) |


composto por imagens capturadas por drones contendo objetos como:

| Classe | Descrição       |
| ------ | --------------- |
| 0      | pedestrian      |
| 1      | people          |
| 2      | bicycle         |
| 3      | car             |
| 4      | van             |
| 5      | truck           |
| 6      | tricycle        |
| 7      | awning-tricycle |
| 8      | bus             |
| 9      | motor           |

---

# Instalação

## Dependências

```bash
pip install ultralytics
pip install opencv-python
```

---

# Preparação do Dataset

## Extração dos arquivos

O dataset deve conter:

```text
VisDrone2019-DET-train/
├── images/
└── annotations/

VisDrone2019-DET-val/
├── images/
└── annotations/
```

---

## Conversão VisDrone → YOLO

O formato original do VisDrone utiliza:

```text
x,y,width,height,score,class,truncation,occlusion
```

Exemplo:

```text
871,572,54,92,1,4,0,0
```

Já o YOLO utiliza:

```text
class x_center y_center width height
```

com todos os valores normalizados entre 0 e 1.

Durante a conversão:

* Classes são convertidas de 1–10 para 0–9.
* Regiões ignoradas são removidas.
* Bounding boxes inválidas são descartadas.
* Imagens são copiadas para a estrutura padrão YOLO.

---

# Arquivo YAML

Após a conversão é criado o arquivo:

```yaml
path: /content/visdrone_yolo

train: images/train
val: images/val

names:
  0: pedestrian
  1: people
  2: bicycle
  3: car
  4: van
  5: truck
  6: tricycle
  7: awning-tricycle
  8: bus
  9: motor
```

---

# Treinamento

## YOLO26n

O modelo base utilizado é:

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
```

Treinamento:

```python
results = model.train(
    data="/content/visdrone.yaml",
    epochs=100,
    imgsz=640,
    batch=32,
    device=0,
    patience=20,
    cache=True,
    name="yolo26n_visdrone_fp32"
)
```

### Parâmetros

| Parâmetro  | Descrição                       |
| ---------- | ------------------------------- |
| epochs     | Número máximo de épocas         |
| batch      | Quantidade de imagens por batch |
| imgsz      | Resolução de treinamento        |
| patience   | Early stopping                  |
| device=0   | Primeira GPU disponível         |
| cache=True | Carrega imagens em memória      |

---

# Validação

Após o treinamento:

```python
from ultralytics import YOLO

model_fp32 = YOLO(
    "/content/runs/detect/yolo26n_visdrone_fp32/weights/best.pt"
)

metrics = model_fp32.val(
    data="/content/visdrone.yaml",
    imgsz=640
)

print(metrics)
```

As principais métricas avaliadas são:

* Precision
* Recall
* mAP50
* mAP50-95

---

# Quantização INT8

## Objetivo

Reduzir o tamanho do modelo e aumentar a velocidade de inferência mantendo a maior precisão possível.

## Exportação

```python
model_fp32.export(
    format="openvino",
    int8=True,
    data="/content/visdrone.yaml",
    imgsz=640
)
```

---

# Modelos Gerados

## Modelo FP32

```text
best.pt
```

Modelo treinado em PyTorch com precisão original.

---

## Modelo INT8

```text
best_int8_openvino_model/
├── best.xml
├── best.bin
└── metadata.yaml
```

O modelo quantizado INT8 é armazenado em formato OpenVINO.

Importante:

O OpenVINO não gera um arquivo `.pt` quantizado.

O diretório completo representa o modelo INT8.

---

# Inferência

## Modelo FP32

```python
model_fp32 = YOLO(
    "weights/best.pt"
)
```

## Modelo INT8

```python
model_int8 = YOLO(
    "weights/best_int8_openvino_model/"
)
```

## Predição

```python
result = model_fp32.predict(
    "imagem.jpg",
    imgsz=640,
    conf=0.25
)
```

---

# Resultados Esperados

Ao final do treinamento:

```text
runs/detect/yolo26n_visdrone_fp32/
```

contém:

```text
weights/
├── best.pt
├── last.pt
└── best_int8_openvino_model/
```

Onde:

* `best.pt` → modelo FP32.
* `last.pt` → último checkpoint.
* `best_int8_openvino_model` → modelo quantizado INT8.

---

# Trabalhos Futuros

* Avaliação sob Motion Blur.
* Avaliação sob Gaussian Blur.
* Avaliação sob compressão JPEG.
* Avaliação sob baixa resolução.
* Comparação YOLO26n vs YOLO26s.
* Comparação FP32 vs INT8.
* Avaliação em dispositivos Edge.

---

# Licença

Projeto desenvolvido para fins acadêmicos e de pesquisa em Visão Computacional, Edge Computing e Quantização de Modelos de Deep Learning.
