from unitxt.catalog import add_to_catalog
from unitxt.metrics import (
    BinaryMaxF1,
    F1Binary,
    F1Fast,
    F1MacroMultiLabel,
    F1MicroMultiLabel,
    F1Strings,
    F1Weighted,
    PrecisionBinary,
    RecallBinary,
)

metric = F1Fast(
    main_score="f1_macro", averages=["macro", "per_class"], ci_score_names=["f1_macro"]
)
add_to_catalog(metric, "metrics.f1_macro", overwrite=True)

metric = F1Fast(main_score="f1_micro", averages=["micro"])
add_to_catalog(metric, "metrics.f1_micro", overwrite=True)

metric = F1MacroMultiLabel()
add_to_catalog(metric, "metrics.f1_macro_multi_label", overwrite=True)

metric = F1MicroMultiLabel()
add_to_catalog(metric, "metrics.f1_micro_multi_label", overwrite=True)

metric = F1Binary()
add_to_catalog(metric, "metrics.f1_binary", overwrite=True)

metric = BinaryMaxF1()
add_to_catalog(metric, "metrics.max_f1_binary", overwrite=True)

metric = F1Weighted()
add_to_catalog(metric, "metrics.f1_weighted", overwrite=True)

metric = RecallBinary()
add_to_catalog(metric, "metrics.recall_binary", overwrite=True)

metric = PrecisionBinary()
add_to_catalog(metric, "metrics.precision_binary", overwrite=True)

metric = F1Strings()
add_to_catalog(metric, "metrics.f1_strings", overwrite=True)
