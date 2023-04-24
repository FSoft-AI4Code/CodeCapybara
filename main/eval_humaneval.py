import os
import fire
import json
import gzip
import subprocess
from pathlib import Path
from glob import glob

def eval_humaneval(prediction_dir,
                   output_dir="./output_dir", 
                   reference_path="./data/HumanEval.jsonl.gz"):
    references = []
    if reference_path.endswith(".jsonl.gz"):
        with gzip.open(reference_path, "rt") as f:
            references = [json.loads(line) for line in f]

    predictions = []
    for prediction_path in glob(f"{prediction_dir}/*.jsonl"):
        with open(prediction_path, "r") as f:
            predictions.extend([json.loads(line) for line in f])

    references = {ref["task_id"]: ref for ref in references}
    for idx, prediction in enumerate(predictions):
        task_id = prediction["task_id"]
        code = prediction["completion"]
        ref = references[task_id]
        prompt = ref["prompt"]
        # 1. remove the prefixed prompt
        code = code[len(prompt):]
        # 2. remove everything after "\n\n"
        code = code.split("\n\n")[0]
        # 3. remove everything after the "def "
        code = code.split("def ")[0]
        prediction["completion"] = code
        predictions[idx] = prediction

    os.makedirs(output_dir, exist_ok=True)
    postprocessed_prediction_path = Path(output_dir, "humaneval_postprocessed_prediction.jsonl")
    with open(postprocessed_prediction_path, "w") as f:
        for prediction in predictions:
            json.dump(prediction, f)
            f.write("\n")
    cmd = f"evaluate_functional_correctness {postprocessed_prediction_path}"
    result = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    for line in result.stdout:
        print(line.decode(), end="")

if __name__ == "__main__":
    fire.Fire(eval_humaneval)
