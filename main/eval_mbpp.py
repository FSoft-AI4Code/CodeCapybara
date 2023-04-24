import os
import fire
import json
import regex
import signal
import tempfile
import threading
import subprocess
import collections
from glob import glob
from tqdm import tqdm
from datetime import datetime
from functools import partial
from multiprocessing import Pool
import numpy as np
from utils.evaluation import estimate_pass_at_k

class Command(object):
    def __init__(self, cmd):
        self.cmd = cmd
        self.process = None

    def run(self, timeout):
        def target():
            self.process = subprocess.Popen(self.cmd, shell=True, preexec_fn=os.setsid)
            self.process.communicate()

        thread = threading.Thread(target=target)
        thread.start()

        thread.join(timeout)
        if thread.is_alive():
            os.killpg(self.process.pid, signal.SIGTERM)
            thread.join()
        return self.process.returncode

class MBPPGoogleDataset(object):
    def __init__(self, path="dataset/mbpp/mbpp.jsonl", mode="function_name"):
        raw_data = sorted(
            [json.loads(x) for x in open(path)], key=lambda x: x["task_id"]
        )
        for i, data_item in enumerate(raw_data):
            assert data_item["task_id"] == i + 1
        self.raw_data = collections.defaultdict()
        self.mode = mode
        # 374 for training, 100 heldout, 500 test
        self.raw_data["train"] = raw_data[:10] + raw_data[510:]
        self.raw_data["test"] = raw_data[10:510]
        # data for codex collector, in input-output-info format
        self.data = collections.defaultdict()
        for split in self.raw_data:
            self.data[split] = self.extract_data(self.raw_data[split], mode)

    @staticmethod
    def extract_data(raw_data, mode):
        if mode == "function_name":
            get_function_name = lambda test_example: regex.match(
                "assert [\(]*([^\(]+)\(", test_example
            ).group(1)
            info = [get_function_name(x["test_list"][0]) for x in raw_data]
        elif mode == "assertion":
            info = [x["test_list"][0] for x in raw_data]
        elif mode == "assertion-full":
            info = [x["test_list"] for x in raw_data]
        else:
            raise Exception(f"Mode {mode} not supported.")
        nls = [x["text"] for x in raw_data]
        codes = [x["code"] for x in raw_data]
        return list(zip(nls, codes, info))

def evaluate_one_mbpp(args, tempdir, references, timeout):
    i, item = args
    task_id = item["task_id"]
    ref = references[task_id - 10 - 1]
    test_cases = ref["test_list"]
    test_setups = ref["test_setup_code"]
    code = item["trg_prediction"]
    # write code to file
    with open(f"{tempdir.name}/code-{i}.py", "w") as fout:
        print(code, file=fout)
        print(test_setups, file=fout)
        for case in test_cases:
            print(case, file=fout)
        fout.close()
    command = Command(f"python {tempdir.name}/code-{i}.py >/dev/null 2>&1")
    execution_result = command.run(timeout=timeout) == 0
    return (task_id, execution_result)
    # return execution_result


""" dataset keys: src, trg_prediction, reference (only trg_prediction useful) """


def evaluate_google_mbpp(
    dataset,
    reference_path,
    split="test",
    timeout=10,
    num_procs=1,
    verbose=False,
):
    references = MBPPGoogleDataset(reference_path)
    tempdir = tempfile.TemporaryDirectory()
    passed_information = list()
    passed_information = collections.defaultdict(list)
    partial_evalutate_one = partial(
        evaluate_one_mbpp, tempdir=tempdir, references=references.raw_data[split], timeout=timeout
    )

    if num_procs > 1:
        with Pool(processes=num_procs) as pool:
            for result_json in tqdm(
                pool.imap(
                    partial_evalutate_one, list(enumerate(dataset))
                ),
                total=len(dataset),
                leave=False,
                disable=not verbose,
            ):
                passed_information[result_json[0]].append(result_json[1])
    else:
        for args in tqdm(
            list(enumerate(dataset)), disable=not verbose
        ):
            result_json = partial_evalutate_one(args)
            passed_information[result_json[0]].append(result_json[1])
    tempdir.cleanup()
    results = {task_id: {"num_samples": len(sample_passed_info), "num_correct": sum(sample_passed_info)} for task_id, sample_passed_info in passed_information.items()}
    return results

def postprocess(datapoint):
    prediction = datapoint.get("trg_prediction")
    prediction = prediction.split("###")[0].strip()
    prediction = prediction.split("if __name__")[0].strip()
    prediction = prediction.split("assert")[0].strip()
    prediction = "def ".join(prediction.split("def ")[:2]).strip()
    datapoint["trg_prediction"] = prediction

    return datapoint

def eval_mbpp(
        prediction_dir,
        reference_path="data/mbpp.jsonl",
        num_procs=8,
        ):
    dataset = []
    for filepath in glob("{}/*.jsonl".format(prediction_dir)):
        with open(filepath) as f:
            dataset.extend([json.loads(line) for line in f])

    dataset = list(map(postprocess, dataset))

    # statistics
    stats = collections.Counter([ex["task_id"] for ex in dataset])
    print(stats)

    results = evaluate_google_mbpp(dataset,
                                 reference_path,
                                 num_procs=num_procs,
                                 verbose=True)
    score_dict = collections.defaultdict(float)
    num_samples = [r["num_samples"] for r in results.values()]
    num_correct = [r["num_correct"] for r in results.values()]
    for k in (1, 10, 80, 100):
        scores = estimate_pass_at_k(num_samples, num_correct, k)
        score_dict[k] = float(np.mean(scores))

    print("Results:\n")
    for k, score in score_dict.items():
        print(f"Pass@{k} = {score*100:.2f}%\n")

if __name__ == "__main__":
    fire.Fire(eval_mbpp)
