# Data-prep function versions (src/helper)

Side-by-side view of the data-prep functions in the four helper files. Line counts are from `ast` at dev_refactor a6dc839; signatures were compared, function bodies were not diffed.

| function | files | lines | what differs | who calls which |
|---|---|---|---|---|
| readingData | utils, utils_from_coolturtles, utils_mse_crps, utils_pnn | 23 / 36 / 45 / 23 | mse_crps adds `rotation_years=None` | see below |
| preparingData | same | 84 / 101 / 210 / 97 | utils adds `date_time`, `verbose`, `val_date_time`; pnn adds `ind`; coolturtles and mse_crps take `input_structure` and `independent_year` | see below |
| creatingAdditionalColumns | same | 111 / 142 / 70 / 111 | coolturtles and mse_crps take `input_structure` as 2nd argument | see below |
| deletingMissingValues | same | 8 / 8 / 45 / 8 | mse_crps is 45 lines, others 8 | see below |
| reshaping | same | 37 / 43 / 48 / 37 | coolturtles and mse_crps take `input_structure` first | see below |
| splittingData | same | 41 / 48 / 33 / 41 | coolturtles adds `year_independent`; mse_crps takes year2 to year5 plus `year_independent` | see below |
| dataframe_checker | same | 7 / 7 / 10 / 7 | same signature in all four; mse_crps is 10 lines | see below |
| dateTimeRetriever | same | 13 / 23 / 8 / 13 | name differs: utils and pnn define `DateTimeRetriever`; coolturtles and mse_crps define `dateTimeRetriever` | see below |

Importers (static grep of `import` lines, file level):
- utils: esb/pipeline.py, esb/status.py, run_spaghetti_graphs.py, src/driver/tuner_models_retriever.py, src/driver/tuner_retriever.py, src/helper/pnn_to_csv.py, esb_03 tuner scripts
- utils_mse_crps: esb/predict.py, esb/_verify.py, esb/_phase5_tuner.py, src/driver/crps_mme_runner.py, operational_mse_crps_driver.py, pnn_mme_driver.py, tuner_retriever.py, src/helper/esb_years.py
- utils_pnn: src/driver/pnn_mme_driver.py only; utils_from_coolturtles: no importer found
