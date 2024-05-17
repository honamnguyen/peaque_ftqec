from typing import Dict, Callable

def run_test(test_cases: Dict, func: Callable, name: str, outfunc: Callable = lambda x: x) -> None:
    """
    Run test cases for a given function and print the results.

    Parameters:
    - test_cases:
        (dict) A dictionary containing the test cases where the keys are the inputs and the values are the expected outputs.
        (list) A list of tuples containing the test cases where the first element of each tuple is the input and the second element is the expected output.
    - func (function): The function to be tested.
    - name (str): The name of the test.
    - outfunc (function, optional): A function to transform the output and expected values before comparison. Defaults to the identity function.

    Returns:
    None
    """
    results = []
    failed_before = False

    # Accept different formats for test_cases
    if type(test_cases) is dict:
        test_cases = test_cases.items()
    elif type(test_cases) is list:
        test_cases = zip(*test_cases)

    for input, answer in test_cases:
        output = func(input)
        results.append(outfunc(output) == outfunc(answer))
        if results[-1] == False:
            if not failed_before:
                failed_before = True
                print(f'\n--- {name} failure log ----')
            print(f'\n  Failed with input {input}')
            print('     Expected:', answer)
            print('     Got:     ', output)
            print()
    print(f'{name} - Tests passed: {sum(results)}/{len(results)}')
