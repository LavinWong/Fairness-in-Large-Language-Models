import os
import datasets
import medium_sized.extrinsic_bias.question_answering.io_utils as io

LABEL_TO_ID_DICT = {"A": 0, "B": 1, "C": 2, "D": 3}


def main():
    data_path = "data/bbq/train_dataset"

    os.makedirs(data_path, exist_ok=True)
    dataset = datasets.load_dataset("race", name="all")
    for phase in dataset.keys():
        out_data = []
        for elem in dataset[phase]:
            assert len(elem["options"]) == 4
            out_data.append({
                "context": elem["article"],
                "query": elem["question"],
                "option_0": " " + elem["options"][0],
                "option_1": " " + elem["options"][1],
                "option_2": " " + elem["options"][2],
                "option_3": " " + elem["options"][3],
                "label": LABEL_TO_ID_DICT[elem["answer"]],
            })
        io.write_jsonl(out_data, os.path.join(data_path, f"{phase}.jsonl"))
    io.write_json({"num_choices": 4}, os.path.join(data_path, "config.json"))

