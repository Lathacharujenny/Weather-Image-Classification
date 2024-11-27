import json

def evaluate_test_data(test_gen, model, results_path):
    results = model.evaluate(test_gen)
    results_dict = {
        'Test Loss': results[0],
        'Test Accuracy': results[1]
    }

    with open(results_path, 'w') as file:
        json.dump(results_dict, file)

    