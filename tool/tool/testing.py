from typing import Dict, Callable

def run_test(test_cases: Dict, func: Callable, name: str, outfunc: Callable = lambda x: x) -> None:
    """
    Run test cases for a given function and print the results.

    Parameters:
    - test_cases (dict): A dictionary containing the test cases where the keys are the inputs and the values are the expected outputs.
    - func (function): The function to be tested.
    - name (str): The name of the test.
    - outfunc (function, optional): A function to transform the output and expected values before comparison. Defaults to the identity function.

    Returns:
    None
    """
    results = []
    failed_before = False
    for input, answer in test_cases.items():
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
