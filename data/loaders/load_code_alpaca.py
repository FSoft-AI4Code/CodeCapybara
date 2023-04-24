import json
def get_code_alpaca(path, tokenizer, max_length):
    with open(path) as f:
        data = json.load(f)
    filtered_data = []
    for line in data:
        if len(tokenizer.encode(line['instruction'])) > max_length: continue
        line['dataset'] = 'codealpaca'
        filtered_data.append(line)
    return filtered_data
        