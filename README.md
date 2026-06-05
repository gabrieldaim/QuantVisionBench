# Edge Vision Benchmark

Benchmark para avaliação de modelos YOLO aplicados à detecção de aeronaves, com foco inicial em treinamento FP32 e posterior análise do impacto da quantização INT8 e de degradações visuais no desempenho do modelo.

## Objetivo

Este projeto tem como objetivo treinar e avaliar modelos YOLO26n para detecção de aeronaves em imagens, utilizando o dataset **Aircraft Detection Model v14**, disponibilizado no Roboflow Universe.

A proposta inicial é estabelecer uma baseline em FP32, que posteriormente será usada como referência para experimentos com:

* quantização INT8;
* degradação visual das imagens;
* comparação entre precisão e desempenho;
* análise de robustez do modelo sob diferentes condições de qualidade visual;
* avaliação de viabilidade para cenários de Edge AI.

---

## Tecnologias Utilizadas

* Python 3.12+
* Ultralytics YOLO26
* PyTorch
* CUDA
* Google Colab
* Roboflow
* Google Drive

---

# Estrutura do Projeto

A estrutura esperada do projeto durante os experimentos no Google Colab é:

```text
project/
│
├── datasets/
│   └── airplane/
│       ├── train/
│       │   ├── images/
│       │   └── labels/
│       │
│       ├── valid/
│       │   ├── images/
│       │   └── labels/
│       │
│       ├── test/
│       │   ├── images/
│       │   └── labels/
│       │
│       └── data.yaml
│
└── runs/
    └── detect/
        └── yolo26n_airplane_fp32/
            └── weights/
                ├── best.pt
                └── last.pt
```

No Google Colab, recomenda-se manter uma cópia persistente do dataset no Google Drive e copiar os arquivos para `/content` antes do treinamento, pois o acesso ao `/content` tende a ser mais rápido do que o acesso direto ao Drive.

Exemplo de organização recomendada:

```text
/content/drive/MyDrive/datasets/airplane/
```

e, durante o treino:

```text
/content/airplane/
```

---

# Dataset

## Aircraft Detection Model v14

Este projeto utiliza o dataset **Aircraft Detection Model v14**, disponibilizado no Roboflow Universe pelo workspace **Aircraft recognition**.

O dataset é voltado para **detecção de aeronaves** em imagens e está no formato de detecção de objetos, com anotações compatíveis com YOLO.

Página do dataset:

```text
https://universe.roboflow.com/aircraft-recognition/aircraft-detection-model-citoe/dataset/14
```

A versão utilizada é a **v14**, descrita no Roboflow como uma versão com mais dados nulos, melhor tratamento de múltiplas aeronaves e maior facilidade para execução local.

## Características do Dataset

| Informação        | Valor                    |
| ----------------- | ------------------------ |
| Nome              | Aircraft Detection Model |
| Versão            | v14                      |
| Tarefa            | Detecção de Objetos      |
| Formato utilizado | YOLO26                   |
| Total de imagens  | 10.756                   |
| Treino            | 9.513 imagens            |
| Validação         | 830 imagens              |
| Teste             | 413 imagens              |
| Licença           | CC BY 4.0                |

## Divisão dos Dados

| Conjunto  | Percentual | Quantidade    |
| --------- | ---------- | ------------- |
| Treino    | 88%        | 9.513 imagens |
| Validação | 8%         | 830 imagens   |
| Teste     | 4%         | 413 imagens   |

## Pré-processamento aplicado pelo Roboflow

A versão 14 do dataset já contém alguns pré-processamentos aplicados pelo Roboflow:

| Etapa             | Configuração             |
| ----------------- | ------------------------ |
| Auto-orientação   | Aplicada                 |
| Redimensionamento | Ajuste dentro de 640x640 |

## Aumentos de dados aplicados pelo Roboflow

A versão baixada do dataset inclui aumentos de dados configurados no Roboflow:

| Aumento         | Configuração                      |
| --------------- | --------------------------------- |
| Flip            | Horizontal                        |
| Crop            | Zoom mínimo de 0% e máximo de 20% |
| Rotação         | Entre -15° e +15°                 |
| Shear           | ±10° horizontal e vertical        |
| Escala de cinza | Aplicada em 15% das imagens       |
| Saturação       | Entre -25% e +25%                 |
| Brilho          | Entre -15% e +15%                 |
| Blur            | Até 3px                           |
| Ruído           | Até 1% dos pixels                 |

---

# Instalação

## Dependências

No Google Colab, instale as dependências principais:

```bash
pip install ultralytics
pip install roboflow
```

Caso seja necessário validar versões ou disponibilidade de GPU:

```python
import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

---

# Download do Dataset

O dataset é baixado diretamente do Roboflow utilizando a API oficial.

```python
!pip install roboflow

from roboflow import Roboflow

rf = Roboflow(api_key="xxx")
project = rf.workspace("aircraft-recognition").project("aircraft-detection-model-citoe")
version = project.version(14)

dataset = version.download("yolo26")

print(dataset.location)
```

Após o download, o dataset normalmente será salvo em uma pasta semelhante a:

```text
/content/Aircraft-Detection-Model-14
```

A estrutura esperada é:

```text
Aircraft-Detection-Model-14/
├── train/
│   ├── images/
│   └── labels/
│
├── valid/
│   ├── images/
│   └── labels/
│
├── test/
│   ├── images/
│   └── labels/
│
└── data.yaml
```

---

# Backup do Dataset no Google Drive

Como o ambiente `/content` do Google Colab é temporário, recomenda-se salvar uma cópia do dataset no Google Drive.

Primeiro, monte o Drive:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Depois, copie o dataset baixado para o Drive:

```python
!mkdir -p /content/drive/MyDrive/datasets/airplane
!cp -r /content/Aircraft-Detection-Model-14/* /content/drive/MyDrive/datasets/airplane/
```

Para verificar se a cópia foi feita corretamente:

```python
!ls /content/drive/MyDrive/datasets/airplane
```

A saída esperada deve conter:

```text
data.yaml
train
valid
test
```

---

# Uso recomendado no Google Colab

Embora o dataset fique salvo no Google Drive, recomenda-se treinar usando uma cópia local em `/content`, pois a leitura de muitas imagens pequenas diretamente do Drive pode ser mais lenta.

No início de cada sessão do Colab:

```python
from google.colab import drive
drive.mount("/content/drive")

!rm -rf /content/airplane
!cp -r /content/drive/MyDrive/datasets/airplane /content/airplane
```

Assim, o dataset usado no treino fica em:

```text
/content/airplane
```

e o arquivo YAML fica em:

```text
/content/airplane/data.yaml
```

---

# Arquivo YAML

O arquivo `data.yaml` é gerado automaticamente pelo Roboflow no download do dataset.

Exemplo de localização:

```text
/content/airplane/data.yaml
```

Para visualizar o conteúdo:

```python
!cat /content/airplane/data.yaml
```

Esse arquivo define:

* caminho das imagens de treino;
* caminho das imagens de validação;
* caminho das imagens de teste;
* número de classes;
* nomes das classes.

Exemplo geral:

```yaml
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 28
names:
  0: classe_0
  1: classe_1
  2: classe_2
```

Observação: os nomes reais das classes devem ser conferidos diretamente no `data.yaml` baixado pelo Roboflow, pois o arquivo é a fonte usada pelo Ultralytics durante o treinamento.

---

# Treinamento

## Modelo Base

O modelo base utilizado é o **YOLO26n**:

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
```

Esse modelo é utilizado como ponto de partida para fine-tuning no dataset de aeronaves.

---

## Treinamento FP32

Configuração principal de treino:

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

results = model.train(
    data="/content/airplane/data.yaml",
    epochs=100,
    imgsz=640,
    batch=-1,
    device=0,
    patience=20,
    cache=True,
    workers=2,
    project="/content/drive/MyDrive/yolo_treinos",
    name="yolo26n_airplane_fp32"
)
```

Com essa configuração, os resultados do treino serão salvos diretamente no Google Drive:

```text
/content/drive/MyDrive/yolo_treinos/yolo26n_airplane_fp32/
```

Os pesos treinados ficam em:

```text
/content/drive/MyDrive/yolo_treinos/yolo26n_airplane_fp32/weights/
```

com os arquivos:

```text
best.pt
last.pt
```

## Parâmetros principais

| Parâmetro  | Descrição                                            |
| ---------- | ---------------------------------------------------- |
| `data`     | Caminho para o arquivo `data.yaml` do dataset        |
| `epochs`   | Número máximo de épocas                              |
| `imgsz`    | Tamanho das imagens usadas no treino                 |
| `batch`    | Tamanho do batch                                     |
| `device=0` | Usa a primeira GPU disponível                        |
| `patience` | Número de épocas sem melhora antes do early stopping |
| `cache`    | Define se as imagens serão cacheadas                 |
| `workers`  | Número de workers do dataloader                      |
| `project`  | Diretório onde os resultados serão salvos            |
| `name`     | Nome da execução de treino                           |

---

# Batch automático

Neste projeto, pode-se usar:

```python
batch=-1
```

Com isso, o Ultralytics calcula automaticamente um batch adequado para a GPU disponível.

Exemplo observado em ambiente Google Colab com GPU Tesla T4:

```text
AutoBatch: Using batch-size 21
```

Isso significa que o framework testou a memória disponível e escolheu automaticamente um valor seguro para o treinamento.

Caso deseje testar manualmente um batch maior:

```python
batch=32
```

Porém, se ocorrer erro de memória na GPU, recomenda-se voltar para:

```python
batch=-1
```

ou reduzir o valor manualmente.

---

# Cache

O parâmetro `cache` controla como as imagens são carregadas durante o treino.

## Cache em RAM

```python
cache=True
```

Carrega as imagens em memória RAM.

Vantagem:

* pode acelerar o treinamento.

Desvantagem:

* consome bastante RAM do sistema.

## Cache em disco

```python
cache="disk"
```

Cria cache no disco.

Vantagem:

* usa menos RAM do sistema.

Desvantagem:

* pode ser mais lento que cache em RAM.

## Sem cache

```python
cache=False
```

Lê as imagens diretamente do disco durante o treino.

Vantagem:

* usa menos RAM.

Desvantagem:

* pode deixar o treino mais lento.

Para Google Colab com pouca RAM disponível, uma configuração mais estável pode ser:

```python
cache="disk"
```

---

# Retomando um Treino Interrompido

Como o Google Colab pode encerrar o ambiente antes do fim do treinamento, recomenda-se salvar os resultados no Google Drive usando o parâmetro `project`.

Para retomar um treino interrompido, use o arquivo `last.pt`:

```python
from ultralytics import YOLO

model = YOLO(
    "/content/drive/MyDrive/yolo_treinos/yolo26n_airplane_fp32/weights/last.pt"
)

results = model.train(
    resume=True
)
```

O `resume=True` continua o treino a partir da última época salva.

Por exemplo:

```text
Treino interrompido na época 75/100
Retomada com resume=True
Continua a partir da época 76/100
```

Importante: o dataset precisa estar disponível no mesmo caminho usado no treino original. Caso o dataset estivesse em `/content`, será necessário copiá-lo novamente para o mesmo local antes de retomar.

Exemplo:

```python
!rm -rf /content/airplane
!cp -r /content/drive/MyDrive/datasets/airplane /content/airplane
```

Depois:

```python
model = YOLO(
    "/content/drive/MyDrive/yolo_treinos/yolo26n_airplane_fp32/weights/last.pt"
)

results = model.train(resume=True)
```

---

# Validação Inicial após o Treino

Após finalizar o treinamento, o modelo FP32 pode ser validado com:

```python
from ultralytics import YOLO

model_fp32 = YOLO(
    "/content/drive/MyDrive/yolo_treinos/yolo26n_airplane_fp32/weights/best.pt"
)

metrics = model_fp32.val(
    data="/content/airplane/data.yaml",
    imgsz=640,
    device=0,
    name="yolo26n_airplane_fp32_val"
)

print(metrics.results_dict)
```

As principais métricas avaliadas são:

* Precision;
* Recall;
* mAP50;
* mAP50-95.

---

# Status Atual

Nesta versão inicial do projeto, o pipeline contempla:

* download do dataset Aircraft Detection Model v14 via Roboflow;
* armazenamento do dataset no Google Drive;
* cópia do dataset para `/content` no Google Colab;
* treinamento do YOLO26n em FP32;
* salvamento dos pesos no Google Drive;
* retomada de treino interrompido com `last.pt`.

As próximas etapas do projeto incluirão:

* deterioração controlada do dataset;
* avaliação do modelo em imagens degradadas;
* exportação e quantização INT8;
* comparação entre modelo FP32 e INT8;
* análise dos resultados.
