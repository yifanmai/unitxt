from unitxt.catalog import add_to_catalog
from unitxt.templates import DialogFieldsData, DialogPairwiseChoiceTemplate

for to_shuffle in [True, False]:
    add_to_catalog(
        DialogPairwiseChoiceTemplate(
            dialog_fields=[
                DialogFieldsData(
                    dialog_field="reference_dialog",
                    assistant_role_label="### Reference answer:",
                    user_role_label="### User:",
                    system_role_label="### System:",
                ),
                DialogFieldsData(
                    dialog_field="dialog_a",
                    assistant_role_label="### Assistant A:",
                    user_role_label="### User:",
                    system_role_label="### System:",
                ),
                DialogFieldsData(
                    dialog_field="dialog_b",
                    assistant_role_label="### Assistant B:",
                    user_role_label="### User:",
                    system_role_label="### System:",
                ),
            ],
            turns_separator="\n\n",
            label_separator="\n",
            choice_a_field="dialog_a",
            choice_b_field="dialog_b",
            answer_field="winner",
            choice_a_label="A",
            choice_b_label="B",
            choice_tie_label="C",
            shuffle=to_shuffle,
            instruction="Please act as an impartial judge and evaluate the quality of the responses provided by two AI"
            " assistants to the user questions. Your evaluation should consider correctness and helpfulness."
            " You will be given reference answers, the assistant A's answers, the assistant B's answers."
            " Your job is to determine which assistant provides correct and helpful answers to the second"
            " user question. Begin your evaluation by comparing both assistants' answers with the reference"
            " answers. Identify and correct any mistakes. Avoid any position biases and ensure that the order"
            " in which the responses were presented does not influence your decision. Do not allow the length"
            " of the responses to influence your evaluation. Do not favor certain names of the assistants."
            " Be as objective as possible. After providing your explanation, output your final verdict by"
            ' strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is'
            ' better, and "[[C]]" for a tie.\n\n',
            input_format="<|The Start of Reference Answer|>\n\n"
            "{reference_dialog}\n\n"
            "<|The End of Reference Answer|>\n\n\n"
            "<|The Start of Assistant A's Conversation with User|>\n\n"
            "{dialog_a}\n\n"
            "<|The End of Assistant A's Conversation with User|>\n\n\n"
            "<|The Start of Assistant B's Conversation with User|>\n\n"
            "{dialog_b}\n\n"
            "<|The End of Assistant B's Conversation with User|>",
            output_format="[[{winner}]]",
            postprocessors=[
                r"processors.extract_mt_bench_label_judgment",
            ],
        ),
        f"templates.response_assessment.pairwise_comparison.mt_bench_multi_turn_with_reference{'_with_shuffling' if to_shuffle else ''}",
        overwrite=True,
    )
