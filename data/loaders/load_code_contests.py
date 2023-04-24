import datasets

def get_code_contests(tokenizer, max_seq_length):
    language_map = {1: "python2", 2: "C++", 3: "python3", 4: "java"}
    dataset_name = "deepmind/code_contests"
    dataset = datasets.load_dataset(dataset_name)
    train_dataset = dataset["train"]
    data = []
    for example in train_dataset:
        desc = example['description']
        desc_tokens = tokenizer.tokenize(desc)
        if len(desc_tokens) > max_seq_length: continue
        solutions = example['solutions']
        languages = set()
        for language, solution in zip(solutions["language"], solutions["solution"]):
            _example = dict(description=desc)
            if language in languages or language == 0:
                continue
            languages.add(language)
            # full_seq = desc + ' Output: ' + solution
            # if len(full_seq) > max_seq_length: continue

            _example["language"] = language_map[language]
            _example["solution"] = solution

            _example['dataset'] = 'code-contest'
            data.append(_example) 
    return data
