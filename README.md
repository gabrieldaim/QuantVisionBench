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

# Degradação Controlada do Dataset

Após o treinamento inicial do modelo YOLO26n em FP32 sobre o dataset original, este projeto prevê a criação de versões degradadas do conjunto de validação e teste. O objetivo é avaliar a robustez do modelo diante de diferentes condições de perda de qualidade visual.

A degradação controlada do dataset permite medir como o desempenho do modelo varia quando as imagens apresentam problemas comuns em cenários reais, como borrão de movimento, perda de foco, compressão de imagem e baixa iluminação.

Importante: as degradações são aplicadas apenas sobre as imagens dos conjuntos de validação e teste. As anotações, isto é, os arquivos de labels com as bounding boxes, são mantidas inalteradas, pois as transformações utilizadas não modificam a posição espacial dos objetos nas imagens.

---

## Objetivo da Degradação

A etapa de degradação tem como objetivo responder perguntas como:

* O modelo mantém boa acurácia quando as imagens perdem nitidez?
* Qual tipo de degradação mais prejudica a detecção de aeronaves?
* O modelo é mais sensível a borrão, compressão, baixa iluminação ou perda de foco?
* Como o desempenho varia conforme a degradação se intensifica?
* O modelo continua viável em cenários de baixa qualidade visual?

Essa análise é importante porque, em aplicações reais, imagens podem ser capturadas em condições imperfeitas, especialmente em vídeos, câmeras de vigilância, transmissões comprimidas ou dispositivos com recursos limitados.

---

## Estratégia Experimental

A estratégia adotada consiste em treinar o modelo apenas uma vez no dataset original e, posteriormente, validá-lo em diferentes versões degradadas do dataset.

A lógica experimental é:

```text
Treino:
dataset original

Validação/Teste:
dataset original
dataset com Motion Blur
dataset com Gaussian Blur
dataset com JPEG Compression
dataset com Low Light
dataset com degradações combinadas
```

Dessa forma, é possível comparar diretamente a queda de desempenho entre o cenário original e os cenários degradados.

---

## Grupos de Degradação

As degradações foram organizadas em cinco grupos principais:

| Grupo            | Tipo                 | Objetivo                                                     |
| ---------------- | -------------------- | ------------------------------------------------------------ |
| Original         | Sem degradação       | Baseline de comparação                                       |
| Motion Blur      | Borrão de movimento  | Simular movimento da aeronave ou instabilidade da câmera     |
| Gaussian Blur    | Desfoque gaussiano   | Simular perda de foco e redução de nitidez                   |
| JPEG Compression | Compressão JPEG      | Simular perda de qualidade por compressão de imagem ou vídeo |
| Low Light        | Baixa iluminação     | Simular imagens escuras ou com baixa visibilidade            |
| Combined         | Degradação combinada | Simular um cenário realista com múltiplos problemas visuais  |

---

## Níveis de Degradação

Cada tipo de degradação é aplicado em três níveis:

| Nível  | Descrição         |
| ------ | ----------------- |
| Light  | Degradação leve   |
| Medium | Degradação média  |
| Heavy  | Degradação pesada |

A divisão em níveis permite observar a progressão da perda de desempenho conforme a qualidade visual das imagens piora.

Exemplo:

```text
motion_blur/light
motion_blur/medium
motion_blur/heavy
```

Essa organização permite avaliar não apenas se uma degradação afeta o modelo, mas também o quanto ela afeta conforme sua intensidade aumenta.

---

## Tipos de Degradação

### Motion Blur

O Motion Blur simula borrão de movimento. Esse tipo de degradação é relevante para detecção de aeronaves, pois imagens reais podem ser capturadas com a aeronave em alta velocidade, com a câmera em movimento ou a partir de vídeos com instabilidade.

Parâmetros sugeridos:

| Nível  | Kernel |
| ------ | -----: |
| Light  |      5 |
| Medium |     11 |
| Heavy  |     21 |

---

### Gaussian Blur

O Gaussian Blur simula perda de foco ou redução geral de nitidez da imagem. Ele é útil para medir a sensibilidade do modelo a imagens desfocadas ou com baixa definição.

Parâmetros sugeridos:

| Nível  | Kernel |
| ------ | -----: |
| Light  |      3 |
| Medium |      7 |
| Heavy  |     11 |

---

### JPEG Compression

A compressão JPEG simula perda de qualidade causada por compactação de imagem. Esse tipo de degradação é comum em vídeos, câmeras IP, transmissões com baixa largura de banda e imagens compartilhadas em sistemas web.

Parâmetros sugeridos:

| Nível  | Qualidade JPEG |
| ------ | -------------: |
| Light  |             70 |
| Medium |             40 |
| Heavy  |             20 |

Quanto menor o valor de qualidade, maior a compressão e maior a perda visual.

---

### Low Light

O Low Light reduz o brilho das imagens, simulando cenários de baixa iluminação, sombra, exposição inadequada ou perda de visibilidade.

Parâmetros sugeridos:

| Nível  | Fator de brilho |
| ------ | --------------: |
| Light  |            0.80 |
| Medium |            0.60 |
| Heavy  |            0.40 |

Quanto menor o fator, mais escura fica a imagem.

---

### Combined

O grupo Combined aplica múltiplas degradações na mesma imagem. Esse cenário é mais próximo de situações reais, nas quais a imagem pode estar simultaneamente escura, borrada, desfocada e comprimida.

Degradações aplicadas:

```text
Low Light + Motion Blur + Gaussian Blur + JPEG Compression
```

Parâmetros sugeridos:

| Nível  | Motion Blur | Gaussian Blur | JPEG Quality | Brilho |
| ------ | ----------: | ------------: | -----------: | -----: |
| Light  |           5 |             3 |           70 |   0.80 |
| Medium |          11 |             7 |           40 |   0.60 |
| Heavy  |          21 |            11 |           20 |   0.40 |

---

## Estrutura de Pastas

Os datasets degradados são organizados separadamente para manter clareza durante as validações.

Estrutura proposta:

```text
aircraft_benchmark/
│
├── original/
│   └── data.yaml
│
└── degraded/
    │
    ├── motion_blur/
    │   ├── light/
    │   ├── medium/
    │   └── heavy/
    │
    ├── gaussian_blur/
    │   ├── light/
    │   ├── medium/
    │   └── heavy/
    │
    ├── jpeg_compression/
    │   ├── light/
    │   ├── medium/
    │   └── heavy/
    │
    ├── low_light/
    │   ├── light/
    │   ├── medium/
    │   └── heavy/
    │
    └── combined/
        ├── light/
        ├── medium/
        └── heavy/
```

Cada cenário degradado possui sua própria estrutura de validação e teste:

```text
degraded/motion_blur/light/
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

## Preservação dos Labels

As degradações aplicadas neste projeto não alteram a geometria principal dos objetos na imagem. Por isso, os arquivos de anotação no formato YOLO são reutilizados sem modificação.

Ou seja, para cada imagem degradada, o label correspondente continua sendo o mesmo da imagem original.

Exemplo:

```text
Imagem original:
valid/images/exemplo.jpg

Label original:
valid/labels/exemplo.txt

Imagem degradada:
degraded/motion_blur/light/valid/images/exemplo.jpg

Label reutilizado:
degraded/motion_blur/light/valid/labels/exemplo.txt
```

Essa estratégia permite comparar os resultados de forma justa, pois o conteúdo semântico e as bounding boxes permanecem os mesmos, enquanto apenas a qualidade visual da imagem é alterada.

---

## Comparação Esperada

Após gerar os datasets degradados, o mesmo modelo treinado no dataset original será validado em cada cenário.

A comparação esperada seguirá o formato:

| Dataset                 | Precision | Recall | mAP50 | mAP50-95 | Queda mAP50-95 (%) | FPS | Latência média (ms) |
| ----------------------- | --------: | -----: | ----: | -------: | -----------------: | --: | ------------------: |
| Original                |         - |      - |     - |        - |                  - |   - |                   - |
| Motion Blur Light       |         - |      - |     - |        - |                  - |   - |                   - |
| Motion Blur Medium      |         - |      - |     - |        - |                  - |   - |                   - |
| Motion Blur Heavy       |         - |      - |     - |        - |                  - |   - |                   - |
| Gaussian Blur Light     |         - |      - |     - |        - |                  - |   - |                   - |
| Gaussian Blur Medium    |         - |      - |     - |        - |                  - |   - |                   - |
| Gaussian Blur Heavy     |         - |      - |     - |        - |                  - |   - |                   - |
| JPEG Compression Light  |         - |      - |     - |        - |                  - |   - |                   - |
| JPEG Compression Medium |         - |      - |     - |        - |                  - |   - |                   - |
| JPEG Compression Heavy  |         - |      - |     - |        - |                  - |   - |                   - |
| Low Light Light         |         - |      - |     - |        - |                  - |   - |                   - |
| Low Light Medium        |         - |      - |     - |        - |                  - |   - |                   - |
| Low Light Heavy         |         - |      - |     - |        - |                  - |   - |                   - |
| Combined Light          |         - |      - |     - |        - |                  - |   - |                   - |
| Combined Medium         |         - |      - |     - |        - |                  - |   - |                   - |
| Combined Heavy          |         - |      - |     - |        - |                  - |   - |                   - |

Essa avaliação permitirá identificar quais tipos de degradação causam maior impacto no desempenho do modelo e em quais condições o modelo deixa de ser confiável.

---

# Estratégia de Validação Experimental

Após a geração dos datasets degradados, o projeto realiza uma etapa de validação comparativa entre diferentes cenários de execução. O objetivo é avaliar não apenas a robustez do modelo diante das degradações visuais, mas também o impacto do ambiente computacional e da quantização INT8 sobre acurácia e desempenho.

A validação considera quatro cenários principais:

| Cenário  | Modelo        | Ambiente         | Objetivo                                   |
| -------- | ------------- | ---------------- | ------------------------------------------ |
| GPU FP32 | PyTorch FP32  | Execução com GPU | Baseline acelerada em GPU                  |
| GPU INT8 | TensorRT INT8 | Execução com GPU | Avaliar impacto da quantização INT8 em GPU |
| CPU FP32 | PyTorch FP32  | Execução sem GPU | Baseline em ambiente restrito              |
| CPU INT8 | OpenVINO INT8 | Execução sem GPU | Avaliar ganho da quantização INT8 em CPU   |

Essa divisão permite analisar separadamente:

* a perda de acurácia causada pelas degradações visuais;
* a diferença de desempenho entre GPU e CPU;
* o impacto da quantização INT8;
* o equilíbrio entre acurácia e velocidade de inferência.

---

## Execuções Repetidas

Cada cenário de validação é executado múltiplas vezes para reduzir o impacto de variações pontuais do ambiente de execução.

Neste projeto, cada combinação de modelo, ambiente e dataset é validada **3 vezes**.

Exemplo:

```text
GPU FP32 + Original → 3 execuções
GPU FP32 + Motion Blur Light → 3 execuções
GPU FP32 + Motion Blur Medium → 3 execuções
...
CPU INT8 + Combined Heavy → 3 execuções
```

A partir dessas execuções, são calculados valores médios e desvios padrão para as principais métricas.

---

## Métricas Coletadas

Durante a validação, são coletadas métricas de acurácia e desempenho.

### Métricas de Detecção

| Métrica   | Descrição                                                                |
| --------- | ------------------------------------------------------------------------ |
| Precision | Proporção de detecções corretas entre todas as detecções realizadas      |
| Recall    | Proporção de objetos reais corretamente detectados                       |
| mAP50     | Média da precisão considerando IoU de 0.50                               |
| mAP50-95  | Média da precisão considerando múltiplos limiares de IoU, de 0.50 a 0.95 |

### Métricas de Desempenho

| Métrica          | Descrição                                                            |
| ---------------- | -------------------------------------------------------------------- |
| Preprocess Time  | Tempo médio de pré-processamento por imagem                          |
| Inference Time   | Tempo médio de inferência por imagem                                 |
| Postprocess Time | Tempo médio de pós-processamento por imagem                          |
| Total Time       | Soma dos tempos de pré-processamento, inferência e pós-processamento |
| FPS              | Quantidade estimada de imagens processadas por segundo               |
| Elapsed Time     | Tempo total da execução de validação                                 |

Além disso, são registradas informações do ambiente, como:

* versão do Python;
* versão do PyTorch;
* disponibilidade de CUDA;
* nome da GPU utilizada;
* dispositivo solicitado na execução.

---

## Cálculo de Queda de Desempenho

Para medir o impacto das degradações visuais, cada resultado degradado é comparado com o resultado obtido no dataset original dentro do mesmo cenário.

A queda percentual de mAP50 é calculada como:

```text
queda_mAP50_% = ((mAP50_original - mAP50_degradado) / mAP50_original) * 100
```

A queda percentual de mAP50-95 segue a mesma lógica:

```text
queda_mAP50-95_% = ((mAP50-95_original - mAP50-95_degradado) / mAP50-95_original) * 100
```

Dessa forma, é possível identificar quais degradações causam maior impacto na acurácia do modelo.

---

## Cálculo de Speedup

Para comparar o ganho de desempenho entre os cenários, o projeto utiliza o cenário **CPU FP32** como referência.

O speedup é calculado como:

```text
speedup = FPS_cenario / FPS_CPU_FP32
```

Com isso, é possível observar quanto uma configuração é mais rápida ou mais lenta em relação à execução em CPU sem quantização.

Exemplo:

```text
speedup = 2.0
```

significa que o cenário avaliado foi aproximadamente duas vezes mais rápido que o CPU FP32.

---

## Organização dos Resultados

Os resultados são armazenados no Google Drive para evitar perda de dados ao encerrar o ambiente do Google Colab.

Estrutura utilizada:

```text
aircraft_benchmark_results/
│
├── reports/
│   │
│   ├── gpu_fp32/
│   │   ├── raw/
│   │   │   └── raw_results.csv
│   │   ├── summary/
│   │   │   ├── summary_results.csv
│   │   │   └── summary_results.xlsx
│   │   ├── plots/
│   │   └── ultralytics_val_runs/
│   │
│   ├── gpu_int8/
│   │   ├── raw/
│   │   ├── summary/
│   │   ├── plots/
│   │   └── ultralytics_val_runs/
│   │
│   ├── cpu_fp32/
│   │   ├── raw/
│   │   ├── summary/
│   │   ├── plots/
│   │   └── ultralytics_val_runs/
│   │
│   ├── cpu_int8/
│   │   ├── raw/
│   │   ├── summary/
│   │   ├── plots/
│   │   └── ultralytics_val_runs/
│   │
│   └── _comparative_4_scenarios/
│       ├── tables/
│       ├── plots/
│       └── comparative_report_4_scenarios.xlsx
```

---

## Arquivos Gerados

Para cada cenário, são gerados arquivos de resultados brutos e consolidados.

### Resultados Brutos

```text
raw_results.csv
```

Contém uma linha para cada execução individual.

Exemplo:

```text
GPU FP32 + Original + Run 1
GPU FP32 + Original + Run 2
GPU FP32 + Original + Run 3
GPU FP32 + Motion Blur Light + Run 1
...
```

### Resultados Consolidados

```text
summary_results.csv
summary_results.xlsx
```

Contêm as médias, desvios padrão, valores mínimos e máximos das métricas coletadas.

### Relatório Comparativo Final

```text
comparative_report_4_scenarios.xlsx
```

Consolida os quatro cenários em uma única planilha, permitindo comparação direta entre:

* GPU FP32;
* GPU INT8;
* CPU FP32;
* CPU INT8.

---

## Gráficos Gerados

O projeto gera gráficos comparativos para auxiliar a análise dos resultados.

Entre os principais gráficos estão:

| Gráfico                                  | Objetivo                                                  |
| ---------------------------------------- | --------------------------------------------------------- |
| mAP50-95 por cenário no dataset original | Comparar acurácia base entre FP32 e INT8                  |
| FPS por cenário no dataset original      | Comparar desempenho computacional                         |
| Latência por cenário                     | Avaliar custo de inferência por imagem                    |
| Trade-off FPS × mAP50-95                 | Analisar equilíbrio entre velocidade e acurácia           |
| Queda de mAP50-95 no nível pesado        | Identificar degradações mais prejudiciais                 |
| mAP50-95 por nível de degradação         | Observar a queda progressiva por intensidade              |
| Heatmap de queda de mAP50-95             | Visualizar rapidamente os impactos por degradação e nível |
| Speedup em relação ao CPU FP32           | Medir ganho de desempenho relativo                        |

Esses gráficos auxiliam na interpretação dos resultados e na identificação dos cenários mais eficientes.

---

## Interpretação Esperada

A análise final busca responder às seguintes perguntas:

* Qual degradação visual mais prejudica o modelo?
* O impacto da degradação é progressivo entre os níveis leve, médio e pesado?
* A quantização INT8 reduz significativamente a acurácia?
* O ganho de desempenho do INT8 compensa eventual perda de precisão?
* O modelo quantizado é mais adequado para execução em ambiente restrito?
* Qual cenário apresenta melhor equilíbrio entre mAP50-95 e FPS?

Essa etapa permite avaliar o comportamento do modelo sob uma perspectiva mais próxima de aplicações reais, considerando tanto a qualidade visual das imagens quanto as restrições computacionais do ambiente de execução.


