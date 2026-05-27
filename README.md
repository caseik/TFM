# Integración de nuevos runners

Este paquete añade los modelos de los tres cuadernos manteniendo la nomenclatura separada por modelo:

```text
model_runners/
├── mobilenet_v2_runner.py
├── vit_b16_runner.py
├── resnet50_runner.py
├── densenet121_runner.py
└── inception_v3_runner.py
```

Además se incluye `_image_runner_utils.py`, con código común para dataset, transforms, WeightedRandomSampler, FocalLoss, entrenamiento, validación, checkpoint y predicción.

En el `main`, cada modelo se ejecuta directamente con:

```python
"runner": "mobilenet_v2"
"runner": "vit_b16"
"runner": "resnet50"
"runner": "densenet121"
"runner": "inception_v3"
```

Los runners devuelven siempre:

```python
{
    "y_true": [...],
    "y_pred": [...]
}
```

por lo que son compatibles con `MetricsGenerator`.
