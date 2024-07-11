import os
import torch
from dataclasses import dataclass
from transformers import (
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)

import medium_sized.extrinsic_bias.question_answering.tasks as tasks
from medium_sized.extrinsic_bias.question_answering.hf_utils import last_checkpoint_handling
from medium_sized.extrinsic_bias.question_answering.model_tweaks import adjust_tokenizer
from medium_sized.extrinsic_bias.question_answering.tokenizer_utils import get_tokenized_dataset

torch.cuda.empty_cache()

@dataclass
class ModelArgs:
    def __init__(self, model_name_or_path="roberta-base", model_mode="mc", use_fast_tokenizer=True, max_seq_length=256, config_name=None, padding_strategy="max_length", parallelize=False, truncation_strategy="only_first", overwrite_cache=False, torch_dtype_fp16=False, eval_phase="validation", predict_phases="test"):
        self.model_name_or_path = model_name_or_path
        self.model_mode = model_mode
        self.use_fast_tokenizer = use_fast_tokenizer
        self.max_seq_length = max_seq_length
        self.config_name = config_name
        self.padding_strategy = padding_strategy
        self.parallelize = parallelize
        self.truncation_strategy = truncation_strategy
        self.overwrite_cache = overwrite_cache
        self.torch_dtype_fp16 = torch_dtype_fp16
        self.eval_phase = eval_phase
        self.predict_phases = predict_phases
        self.cache_dir = None
        self.model_revision = "main"
        self.tokenizer_name = None


class TaskArgs:
    def __init__(self, task_name="custom", task_base_path=None):
        self.task_name = task_name
        self.task_base_path = task_base_path


def main():
    model_args = ModelArgs()
    task_args = TaskArgs()
    training_args = TrainingArguments(output_dir="", per_device_train_batch_size = 4, 
                                      per_device_eval_batch_size = 4, gradient_checkpointing = True)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
    )
    adjust_tokenizer(tokenizer)

    torch_dtype = torch.float16 if model_args.torch_dtype_fp16 else None
    model = AutoModelForMultipleChoice.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        torch_dtype=torch_dtype,
        # gradient_checkpointing=training_args.gradient_checkpointing,
    )

    task = tasks.get_task("data/bbq/train_dataset")
    dataset_dict = task.get_datasets()

    tokenized_dataset_dict = get_tokenized_dataset(
        task=task,
        dataset_dict=dataset_dict,
        tokenizer=tokenizer,
        max_seq_length=model_args.max_seq_length,
        padding_strategy=model_args.padding_strategy,
        truncation_strategy=model_args.truncation_strategy,
        model_mode=model_args.model_mode,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset_dict.get("train"),
        eval_dataset=tokenized_dataset_dict.get("validation"),
        compute_metrics=task.compute_metrics,
        tokenizer=tokenizer,
    )

    checkpoint = last_checkpoint_handling(training_args=training_args, model_args=model_args)
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    trainer.save_model(output_dir=os.path.join("data/bbq", "checkpoint-last"))
    # noinspection PyArgumentList
    trainer.log_metrics("train", train_result.metrics)
    # noinspection PyArgumentList
    trainer.save_metrics("train", train_result.metrics)
    # noinspection PyArgumentList
    trainer.save_state()