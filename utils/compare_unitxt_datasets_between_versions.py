"""This file contain utilities for comparing actual datasets generated by unitxt.

Common use of these utilities is
after creating new version, in order to make sure that the generated datasets (examples) remains the same.
Common stages of comparison:
1. Create a file with many recipes to compare - using create_recipes_params_file function
2. Create examples from these recipes using version A, store the example in dir A.
Done by run create_examples_for_recipes_file
3. Create examples from these recipes using version B, store the example in dir B.
Done by run create_examples_for_recipes_file
4. Compare dir A and dir B using generate_diff_html (defined in a separate file).
"""

import concurrent.futures
import itertools
import json
import os
import os.path
import random
import threading

from unitxt import register_local_catalog
from unitxt.dataset_utils import fetch
from unitxt.standard import DatasetRecipe

DEMOS_POOL_SIZE = 100
TEST_SIZE = 100
recipes_params_file_name = "recipes_params.json"
SEED = 42
random.seed(SEED)


def get_all_artifacts_in_catalog_by_type(catalog_path, artifact_type):
    artifacts = []
    for root, _dirs, files in os.walk(os.path.join(catalog_path, artifact_type)):
        for file in files:
            artifact_abs_path = os.path.join(root, file)
            artifact_relative_path = os.path.relpath(artifact_abs_path, catalog_path)
            artifacts.append(
                artifact_relative_path.replace("/", ".").replace(".json", "")
            )
    return artifacts


"""
 This function create a file which contains many recipes for comparison. For each recipe within the catalogs list,
 which is also exist in the compare catalogs list, this function choose an augmentor, system prompt and format from the
 artifacts that defined both in the catalogs and compare catalogs list. In addition, for each recipe the function choose
 a template, and number of examples.
 Parameters:
    results_dir - path to the output file. The output file will be results_dir/ecipes_params.json
    catalogs - list of catalogs(paths) for iterating the cards, and selecting other artifacts.
    compare_catalogs - list of catalogs, since this function should creat recipes in order to compare datasets between
    two versions, the recipes should contain artifact that exist in both versions.
"""


def create_recipes_params_file(results_dir, catalogs, compare_catalogs):
    os.makedirs(results_dir, exist_ok=True)
    type2artifacts = {}
    compared_type2artifacts = {}
    for catalog_path in catalogs:
        register_local_catalog(catalog_path)
        for type in ["cards", "augmentors", "system_prompts", "formats"]:
            if type not in type2artifacts:
                type2artifacts[type] = []
            type2artifacts[type] += get_all_artifacts_in_catalog_by_type(
                catalog_path, artifact_type=type
            )

    for catalog_path in compare_catalogs:
        register_local_catalog(catalog_path)
        for type in ["cards", "augmentors", "system_prompts", "formats"]:
            if type not in compared_type2artifacts:
                compared_type2artifacts[type] = []
            art_type = type
            if type == "system_prompts":
                art_type = "instructions"
            compared_type2artifacts[type] += get_all_artifacts_in_catalog_by_type(
                catalog_path, artifact_type=art_type
            )

    compared_type2artifacts["system_prompts"] = [
        a.replace("instruction", "system_prompt")
        for a in compared_type2artifacts["system_prompts"]
    ]

    for type in ["cards", "augmentors", "system_prompts", "formats"]:
        type2artifacts[type] = [
            a for a in type2artifacts[type] if a in compared_type2artifacts[type]
        ]
        type2artifacts[type].sort()

    recipes = []
    for _i, card in enumerate(type2artifacts["cards"]):
        if "ansible" in card:
            continue
        try:
            card_obj = fetch(card)
            recipes.append(
                {
                    "card": card,
                    "template": random.choice(
                        [e.__id__ for e in card_obj.templates.items]
                    ),
                    "augmentor": random.choice(type2artifacts["augmentors"]),
                    "system_prompt": random.choice(type2artifacts["system_prompts"]),
                    "format": random.choice(type2artifacts["formats"]),
                    "num_demos": random.randint(0, 5),
                }
            )
        except:
            pass

    with open(os.path.join(results_dir, recipes_params_file_name), "w") as file:
        json.dump(recipes, file)


def generate_examples_for_configuration(
    card, template, system_prompt, format, num_demos, is_old_version
):
    system_prompt_field = "system_prompt"
    if is_old_version:
        system_prompt_field = "instruction"
        system_prompt = system_prompt.replace("system_prompt", "instruction")
        template = template.replace("with_context.no_intro", "context_no_intro")
        card = card.replace("almost_evil", "almostEvilML_qa_by_lang")
    inputs = {
        "card": card,
        "template": template,
        system_prompt_field: system_prompt,
        "format": format,
        "num_demos": num_demos,
        "demos_pool_size": DEMOS_POOL_SIZE,
        "loader_limit": 2 * TEST_SIZE + DEMOS_POOL_SIZE,
    }
    recipe = DatasetRecipe(**inputs)
    stream = recipe()
    return list(itertools.islice(stream["test"], TEST_SIZE))


def get_file_name(
    dir, card, template, system_prompt, format, num_demos, augmentor=None
):
    return os.path.join(
        dir, f"{card}__{template}__{system_prompt}__{format}__{num_demos}"
    )


def generate_examples_for_configuration_and_save_in_file(
    recipe, dir, compare_dir, results
):
    card = recipe["card"]
    template = recipe["template"]
    system_prompt = recipe["system_prompt"]
    format = recipe["format"]
    num_demos = recipe["num_demos"]
    results_key = get_file_name(dir="", **recipe)
    try:
        file_name = get_file_name(dir, card, template, system_prompt, format, num_demos)
        if os.path.isfile(file_name):
            results[results_key] = {
                "status": "skip",
                "reason": f"File already exists. Skipping {file_name}",
            }
            return
        if compare_dir is not None:
            compare_file_name = get_file_name(
                compare_dir, card, template, system_prompt, format, num_demos
            )
            if not os.path.isfile(compare_file_name):
                results[results_key] = {
                    "status": "skip",
                    "reason": "File doesn't exist for compare. Skipping {compare_file_name}",
                }
                return
        examples = generate_examples_for_configuration(
            card,
            template,
            system_prompt,
            format,
            num_demos,
            is_old_version=compare_dir is not None,
        )
        with open(file_name, "w") as file:
            for i, example in enumerate(examples):
                file.write(f"{i}:: {example['source']}")
            file.write("\n")
        results[results_key] = {"status": "success"}
        return
    except Exception as e:
        results[results_key] = {
            "status": "failure",
            "exception": e,
            "reproduce": "NNNresults = {" + f'"{results_key}": ' + " {}}NNN"
            "generate_examples_for_configuration_and_save_in_file(" + f"recipe={recipe}"
            # str(", ".join(f"{key}='{value}'" for key, value in recipe.items() if key != 'augmentor')) +
            f', dir="{dir}", compare_dir={compare_dir}, results=results)NNN'
            + "print(results)NNN",
        }
        return


def handle_recipe(recipe, dir, compare_dir, results):
    generate_examples_for_configuration_and_save_in_file(
        recipe, dir, compare_dir, results
    )
    print_status(results, dir)


lock = threading.Lock()


def print_status(results, dir):
    lock.acquire()
    try:
        with open(os.path.join(dir, "log.txt"), "w") as file:
            for i, (recipe_name, status) in enumerate(results.items()):
                file.write(
                    f"{i}: {recipe_name} -- {status}\n".replace("\\'", "'").replace(
                        "NNN", "\n"
                    )
                )
    finally:
        lock.release()


"""
    This function gets file with recipes configurations (usually generated by create_recipes_params_file) and generate
    few example from each recipe test, in order to compare the actual generated examples between different unitxt
    versions.
    Parameters:
        recipes_list_file: path to a json file with list of recipes.
        results_dir: path to the dir that will hold the result. This function will generate a separate file for each
                     recipe with it generated examples.
        compare_dir: many times, the comparison occurs between given version to a suggested one. Therefore, in case that
                     the given function resulted example doesn't contain file for a certion recipe, the compared result
                     should ignore it as well. The compare_dir is a path to the compared results dir.
        max_recipes: This parameter enables to restrict run to the only nth recipes. None means using the entire list.
        max_workers: Since generating examples for all the recipes can take long, and since most of the time is due to
                     I/O, this function runs multithreaded. this parameter defines the amount of workers it will use.
"""


def create_examples_for_recipes_file(
    recipes_list_file, results_dir, compare_dir, max_recipes=None, max_workers=100
):
    os.makedirs(results_dir, exist_ok=True)
    with open(recipes_list_file) as file:
        recipes = json.load(file)

    if max_recipes is not None:
        recipes = recipes[:max_recipes]

    results = {
        get_file_name(dir="", **recipe): {"recipe": recipe} for recipe in recipes
    }
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        {
            executor.submit(
                handle_recipe, recipe, results_dir, compare_dir, results
            ): recipe
            for recipe in recipes
        }
