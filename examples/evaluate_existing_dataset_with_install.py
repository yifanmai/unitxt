from unitxt.api import evaluate, load_dataset
from unitxt.inference import CrossProviderInferenceEngine
from unitxt.text_utils import print_dict

# Use the Unitxt APIs to load the wnli entailment dataset using the standard template in the catalog for relation task with 2-shot in-context learning.
# We set loader_limit to 20 to limit reduce inference time.
dataset = load_dataset(
    card="cards.wnli",
    template="templates.classification.multi_class.relation.default",
    format="formats.chat_api",
    num_demos=2,
    demos_pool_size=10,
    loader_limit=20,
    split="test",
)
# loader=LoadFromDictionary(data=data,data_classification_policy=["public"]),

inference_model = CrossProviderInferenceEngine(
    model="llama-3-2-1b-instruct", provider="watsonx"
)
"""
We are using a CrossProviderInferenceEngine inference engine that supply api access to provider such as:
watsonx, bam, openai, azure, aws and more.

For the arguments these inference engines can receive, please refer to the classes documentation or read
about the the open ai api arguments the CrossProviderInferenceEngine follows.
"""
predictions = inference_model.infer(dataset)

evaluated_dataset = evaluate(predictions=predictions, data=dataset)

# Print results
print_dict(
    evaluated_dataset[0],
    keys_to_print=[
        "source",
        "prediction",
        "processed_prediction",
        "references",
        "score",
    ],
)
