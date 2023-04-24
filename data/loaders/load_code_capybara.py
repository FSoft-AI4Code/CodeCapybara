import jsonlines
def get_code_capybara(path, tokenizer, max_length):
    data = []
    with jsonlines.open(path) as f:
        for line in f:
            if 'gen_code' not in line: continue
            if len(tokenizer.encode(line['instruction'])) > max_length: continue
            line['dataset'] = 'codecapybara'
            data.append(line)
    return data
        