# Versión definitiva del proyecto TFM HAM10000

Esta versión fusiona lo mejor de `contenido_ismael` y `contenido_completo`.

## Criterio de fusión

Se toma como base `contenido_ismael` porque incluye una batería de modelos más completa:

- DINOv2 + centroides
- DINOv2 + MLP
- EfficientNetB0
- ConvNeXt-Tiny
- MobileNetV2
- ViT-B/16
- ResNet50
- DenseNet121
- InceptionV3

Y se incorpora la idea metodológica más importante de `contenido_completo`:

- particionado flexible,
- soporte few-shot,
- soporte porcentajes,
- y augmentación real asociada al pipeline de imágenes, no duplicación simple de filas.

## Mejoras principales

### 1. Split sin fuga de información

El nuevo `SplitManager` divide por `lesion_id`, no solo por imagen.  
Esto es esencial en HAM10000 porque una misma lesión puede tener varias imágenes. Si una imagen de la misma lesión cae en train y otra en test, el resultado queda artificialmente inflado.

### 2. Escenarios experimentales claros

En `config.py` se definen tres escenarios:

```text
few_shot_50
few_shot_100
stratified_70_15_15
```

Esto permite defender en el paper tanto la comparación con pocos datos como la comparación clásica con más datos.

### 3. Test común para todos los modelos

Para un mismo escenario y una misma semilla, todos los modelos reciben la misma partición.  
Esto hace que la comparación entre fine-tuning y embeddings sea metodológicamente justa.

### 4. Augmentación real

La augmentación no se realiza duplicando filas en el `SplitManager`.  
Se aplica en los transforms de PyTorch dentro de los runners de imagen:

- flips
- rotaciones
- color jitter
- normalización ImageNet

Esto evita inflar artificialmente el dataset con copias estáticas.

### 5. Métricas clínicas

El nuevo `MetricsGenerator` calcula:

- Accuracy
- Balanced accuracy
- Macro F1
- Weighted F1
- Recall por clase
- Precision por clase
- F1 por clase
- Falsos negativos por clase
- Intervalos de confianza Wilson
- Matrices de confusión

Para el paper, la métrica prioritaria debe ser la sensibilidad de melanoma y la tasa de falsos negativos de melanoma.

## Ejecución

### Ejecutar escenario por defecto

```bash
python main.py
```

### Ejecutar solo embeddings

```bash
python main.py --only-embeddings
```

### Ejecutar solo modelos de imagen

```bash
python main.py --only-images
```

### Ejecutar un escenario concreto

```bash
python main.py --scenario few_shot_100
```

```bash
python main.py --scenario stratified_70_15_15
```

### Ejecutar con una sola semilla para pruebas rápidas

```bash
python main.py --seeds 42
```

## Salidas generadas

```text
results/
├── table_1_experimental_plan.csv
├── table_2_performance.csv
├── table_3_false_negatives.csv
├── confusion_matrices/
├── splits/
└── cache/
```

## Uso recomendado para el TFM

La estructura experimental recomendada para el documento es:

1. Ejecutar `few_shot_50` para analizar comportamiento con pocos datos.
2. Ejecutar `few_shot_100` para ver si los embeddings escalan bien.
3. Ejecutar `stratified_70_15_15` para comparar con el protocolo clásico.
4. Repetir al menos con tres semillas: 42, 1337 y 2026.
5. Reportar media y desviación típica de:
   - balanced accuracy,
   - macro F1,
   - recall melanoma,
   - false negative rate melanoma.

## Advertencia metodológica

Si BiomedCLIP aparece en el paper como modelo evaluado, debe añadirse como extractor real en `feature_extractor.py`.  
Si no se implementa, debe mencionarse solo como trabajo futuro.
