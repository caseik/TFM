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


| Model   | Global Acc. | 95% CI    | Melanoma | Mel CI | Mel n | Nevus | Nevus CI | Nevus n | ... |
| ------- | ----------: | --------- | -------: | ------ | ----: | ----: | -------- | ------: | --- |
| Model A |       81.0% | 80.2–81.8 |    74.0% | 69–78  |   420 | 89.0% | 88–90    |    3800 | ... |
| Model B |       93.0% | 81–99     |    95.0% | 73–99  |    18 | 92.0% | 75–98    |      25 | ... |


| Model   | MEL FN | MEL n | MEL FN Rate | MEL CI | BCC FN | BCC n | BCC CI |
| ------- | -----: | ----: | ----------: | ------ | -----: | ----: | ------ |
| Model A |    122 |   420 |       29.0% | 25–34  |    ... |   ... | ...    |
| Model B |      1 |    18 |        5.5% | 0–27   |    ... |   ... | ...    |
