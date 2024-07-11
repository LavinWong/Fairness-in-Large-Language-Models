import os
import medium_sized.extrinsic_bias.question_answering.io_utils as io
from medium_sized.extrinsic_bias.question_answering.io_utils import write_json, show_json
from transformers import (
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
from medium_sized.extrinsic_bias.question_answering.model_tweaks import adjust_tokenizer
from medium_sized.extrinsic_bias.question_answering.tokenizer_utils import get_tokenized_dataset
import medium_sized.extrinsic_bias.question_answering.tasks as tasks
from medium_sized.extrinsic_bias.question_answering import data_generation, run_lrqa

LABEL_TO_ID_DICT = {"A": 0, "B": 1, "C": 2, "D": 3}
CATEGORIES = [
    # 'Age',
    # 'Disability_status',
    'Gender_identity',
    # 'Nationality',
    # 'Physical_appearance',
    # 'Race_ethnicity',
    # 'Race_x_SES',
    # 'Race_x_gender',
    # 'Religion',
    # 'SES',
    # 'Sexual_orientation',
]

def run_experiment():
    print("------------Medium-sized LMs: Extrinsic bias - Natural Language Understanding - Question Answering task------------")
    print("------------Training model------------")
    data_generation.main()
    run_lrqa.main()
    input_data_path = "data/bbq/input_data"
    data_path = "data/bbq"
    for category in CATEGORIES:
        os.makedirs(os.path.join(data_path, category), exist_ok=True)
        data = io.read_jsonl(os.path.join(input_data_path, f"{category}.jsonl"))
        new_data = [
            {
                "context": example["context"],
                "query": " " + example["question"],
                "option_0": " " + example["ans0"],
                "option_1": " " + example["ans1"],
                "option_2": " " + example["ans2"],
                "label": example["label"],
            }
            for example in data
        ]
        io.write_jsonl(new_data, os.path.join(data_path, category, "validation.jsonl"))
        io.write_json({"num_choices": 3}, os.path.join(data_path, category, "config.json"))

        print("------------Evaluation task------------")

        training_args = TrainingArguments(output_dir="data/bbq", per_device_train_batch_size = 4, 
                                        per_device_eval_batch_size = 4, gradient_checkpointing = True)
        
        config = AutoConfig.from_pretrained(
            "checkpoint-last",
            cache_dir=None,
            revision="main",
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "checkpoint-last",
            cache_dir=None,
            use_fast=True,
            revision="main",
        )
        adjust_tokenizer(tokenizer)

        task = tasks.get_task(os.path.join(data_path, category))
        dataset_dict = task.get_datasets()

        model = AutoModelForMultipleChoice.from_pretrained(
            "checkpoint-last",
            from_tf=bool(".ckpt" in "checkpoint-last"),
            config=config,
            cache_dir="None",
            revision="main",
            torch_dtype=None,
        )

        tokenized_dataset_dict = get_tokenized_dataset(
            task=task,
            dataset_dict=dataset_dict,
            tokenizer=tokenizer,
            max_seq_length=256,
            padding_strategy="max_length",
            truncation_strategy="only_first",
            model_mode="mc",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset_dict.get("train"),
            eval_dataset=tokenized_dataset_dict.get("validation"),
            compute_metrics=task.compute_metrics,
            tokenizer=tokenizer,
        )

        print(tokenized_dataset_dict)

        validation_metrics = trainer.evaluate(eval_dataset=tokenized_dataset_dict["validation"])
        write_json(validation_metrics, os.path.join("medium_sized/extrinsic_bias/question_answering", f"{category}.json"))
        show_json(validation_metrics)



