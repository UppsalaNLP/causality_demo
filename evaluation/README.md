# Evaluation
Follow these steps to evaluate a new model for Swedish:

1) get the test set at https://github.com/UppsalaNLP/Swedish-Causality-Datasets/tree/master/Curated-Ranking-Data-Set

2) create a ranking with the model to evaluate (saved as 43\_query\_ranking\_model\_<modelpath>.json)

```
from create_eval_ranking import predict_annotated_queries
from complete_evaluation import load_dataset
_, _, data = load_dataset(path_to_test_set)
predict_annotated_queries(data, modelpath)
```

3) run evaluation

```
from complete_evaluation import load_dataset, evaluation_report
evaluation_report(path_to_test_set, edrc=True, ranking=path_to_model_ranking, outfile=result_file_name)
```
