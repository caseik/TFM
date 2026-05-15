| Modelo            | Variante | Train |  Val | Test | LoRA | Augment. |
| ----------------- | -------- | ----: | ---: | ---: | ---: | -------- |
| Embeddings        | MLP      |   350 |    0 | 9650 |    0 | No       |
| Embeddings        | SVM      |   350 |    0 | 9650 |    0 | No       |
| Fine-tuning       | CNN      |  8000 | 1000 | 1000 |    0 | Sí       |
| Embeddings + LoRA | MLP      |   350 |  500 | 9150 |  700 | Sí       |


| Modelo   | Acc. Global | Melanoma |   n | Nevus |    n |  BCC |   n | ... |
| -------- | ----------: | -------: | --: | ----: | ---: | ---: | --: | --- |
| Modelo A |         81% |      74% | 420 |   89% | 3800 |  77% | 210 | ... |
| Modelo B |         93% |      95% |  18 |   92% |   25 | 100% |   9 | ... |


| Modelo | MEL FN | MEL n | BCC FN | BCC n | AKIEC FN | AKIEC n |
| ------ | -----: | ----: | -----: | ----: | -------: | ------: |
