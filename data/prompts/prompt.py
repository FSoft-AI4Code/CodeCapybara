first_PROMPT = {
        "template": """
I have {num_code_snippets} code snippets. You are a software programming teacher and you want to have assignment problems for your students to implement with the code snippets as solutions, how would you generate such assignment questions given the snippets.
 
Here are the requirements: 
{requirements}

List of {num_code_snippets} code snippets:

        """,
        "requirements": {
            "r1": "The problem statements should clearly state the problem that the code is intended to solve, and the expected behavior of the code.",
            "repeat_words": "Try not to repeat the verb for each assignment statement to maximize diversity.",
            "diversity": "The language used for the assignment statement also should be diverse. For example, you should combine questions with imperative problem statements.",
            "gpt" :"A GPT language model should be able to complete the problem statements. For example, do not ask the assistant to create any visual or audio output. For another example, do not ask the assistant to wake you up at 5pm or set a reminder because it cannot perform any action. ",
            "english" :"The problem statements should be in English and not include code. ",
            "manner" :"The problem statements should be written in a clear and concise manner, with proper grammar and punctuation.",
            "length": "The problem statements should be concise, ideally at least {min_length} words and at most {max_length} words. ",
            "natural_language_reflection": "The problem statements must correctly reflect the corresponding code in natural language. ",
            "code_related" :"All tasks should be coding or programming related.",
            "terminology": "Use the appropriate terminology corresponding to each language in the problem statement. For example, List in Python corresponds to vector in C++, dictionary in python corresponds to hash in Perl. ",
            "programming_language": "The problem statements must mention the programming language of the code.",
            "input_output": "The problem statements may include input and output descriptions or not. ",
},
        "optional_requirements": ["input_output", "length"]
            }

second_PROMPT = {
        "template": """
I have a pair of a code problem assignments and its solution. However, the solution is not able to run because it lacks libraries that need to be imported as well as other dependency functions appeared within the solution. Write a full script containing all libraries and the dependency implementation that are required to run the solution code.

Here are the mandatory requirements:
{requirements}
- Problem:
{instruction}
- Solution: 
{code}

Here is the implementation:
        """,
        "requirements": {
            "r1": "The script should not contain the main function and testing code.",
            "r2": "Your answer should not include any sample usage and code comment.",
            "r3": "Only dependencies necessitated in the solution should be implemented.",
            "r4": "Your answer only contains libraries and dependencies implementation.",
            "r5": "Your answer should have a unique script.",
            "r6": "New functions and code elements in your answer should not be empty.",
            "r7": "Your answer do not include any explanation and note.",
    # "r1": "The script should not contain the main function.",
    # "r2": "Only dependencies appeared in the solution should be implemented.",
    # "r3": "Your answer only contains libraries and dependencies implementation, not the solution.",
    # "r4": "The solution should not be mentioned in your answer.",
    # "r5": "Your answer must only contain the code without any explaination.",
            }
        }

code_contests_prompts = [
                         '{problem_description}'
                         'Give me the {language} code solution for this problem.\n{problem_description}',
                         'Solve this problem in {language}.\n{problem_description}',
                         'Can you provide a {language} implementation for this algorithmic challenge? \n{problem_description}',
                         'Demonstrate a {language} solution for this code problem.\n{problem_description}',
                         'I need a {language} program to solve this competitive programming challenge.\n{problem_description}',
                         'Plea se write a {language} code to solve this algorithmic problem.\n{problem_description}',
                         'Show me how to solve this code challenge in {language}.\n{problem_description}',
                         'Python implementation needed for this competitive programming problem.\n{problem_description}',
                         'Can you code a solution in {language} for this problem?\n{problem_description}',
                         'I require a {language} script to solve this algorithmic task.\n{problem_description}',
                         'Please present a {language} solution for this code problem.\n{problem_description}',
                         'Python code for this competitive programming challenge please.\n{problem_description}',
                         'Could you show me a {language} impl ementation for this programming challenge?\n{problem_description}',
                         "I'm looking for a {language} program to solve this coding challenge.\n{problem_description}",
                         'Can you help me with a {language} solution for this algorithm?\n{problem_description}',
                         'Please code a {language} solution for this problem.\n{problem_description}',
                         'I need a {language} implementation for this code problem.\n{problem_description}',
                         'Provide me with a {language} code for this competitive programming problem.\n{problem_description}',
                         'Solve this algorithmic problem using {language}.\n{problem_description}',
                         "I'm seeking a {language} script to solve this code challeng e.\n{problem_description}",
                         'Please write a {language} program to solve this competitive programming challenge.\n{problem_description}',
                         'Could you show me a {language} solution for this code problem?\n{problem_description}',
                         'I need a {language} implementation for this algorithmic task.\n{problem_description}',
                         "I'm looking for a {language} code to solve this competitive programming challenge.\n{problem_description}",
                         'Can you present a {language} program to solve this algorithmic problem?\n{problem_description}',
                         'Python implementation required for this code problem.\n{problem_description}',
                        'Solve this programming challenge using py thon.\n{problem_description}',
                        'Can you code a {language} solution for this competitive programming challenge?\n{problem_description}',
                        'I require a {language} implementation to solve this algorithmic challenge.\n{problem_description}',
                        'Provide me with a {language} code to solve this coding problem.\n{problem_description}',
                        'Please write a {language} script to solve this programming challenge.\n{problem_description}',
                        'Python solution needed for this code problem.\n{problem_description}',
                        'Can you help me solve this competitive programming problem with a {language} implementation?\n{problem_description}',
                        "I'm seeking a {language} program to solve this algorithmic challenge.\n{problem_description}",
                        'Solve this code problem using {language} language.\n{problem_description}',
                        'Please present a {language} implementation for this programming challenge.\n{problem_description}',
                        'Could you show me how to solve this competitive programming challenge in {language}?\n{problem_description}',
                        'I need a {language} code to solve this algorithmic task. \n{problem_description}',
                        'Python solution required for this code problem.\n{problem_description}',
                        'Can you code a {language} implementation for this programming challenge?\n{problem_description}',
                        "I'm looking for a {language} program t o solve this coding problem.\n{problem_description}",
                        'Provide me with a {language} solution for this algorithmic challenge.\n{problem_description}',
                        'Please write a {language} code to solve this competitive programming problem.\n{problem_description}',
                        'Python implementation needed to solve this code problem.\n{problem_description}',
                        'Can you help me solve this programming challenge with a {language} implementation?\n{problem_description}',
                        'I require a {language} program to solve this competitive programming challenge.\n{problem_description}',
                        'Please show me a {language} solution for this algorithmic task.\n{problem_description}']


codealpaca_prompts = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Output:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Output:"
    ),
}