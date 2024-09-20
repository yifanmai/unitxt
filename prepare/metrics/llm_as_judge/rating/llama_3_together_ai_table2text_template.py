from unitxt import add_to_catalog
from unitxt.inference import TogetherAiInferenceEngine
from unitxt.llm_as_judge import LLMAsJudge

model_list = ["meta-llama/Llama-3-8b-chat-hf"]
template = "templates.response_assessment.rating.table2text_single_turn_with_reference"
task = "rating.single_turn_with_reference"

for model_id in model_list:
    inference_model = TogetherAiInferenceEngine(
        model_name=model_id, max_tokens=252
    )
    model_label = model_id.split("/")[1].replace("-", "_").replace(".", ",").lower()
    model_label = f"{model_label}_together_ai"
    template_label = template.split(".")[-1]
    metric_label = f"{model_label}_template_{template_label}"
    metric = LLMAsJudge(
        inference_model=inference_model,
        template=template,
        task=task,
        main_score=metric_label,
        prediction_type="str",
    )

    add_to_catalog(
        metric,
        f"metrics.llm_as_judge.rating.{model_label}_template_{template_label}",
        overwrite=True,
    )
