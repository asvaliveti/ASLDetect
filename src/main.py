from ModelRunner import ModelRunner
import json

NUM_EPOCHS = 400
PARAMS_FILE = "test_params.json"

if __name__ == "__main__":
    # Load runner parameters from JSON file
    with open("src/test_params.json", "r") as f:
        runner_params = json.load(f)

    print(f"Loaded {len(runner_params)} parameter sets from {PARAMS_FILE}")
    with open("runner_log.txt", "a") as log_file:
        for i, params in enumerate(runner_params):
            runner_id = f"model_{i+1}"
            lr = params.get("lr", 0.001)
            lr_decay = params.get("lr_decay", 0.9)
            batch_size = params.get("batch_size", 32)
            log_str = f"Running {runner_id} | lr={lr} | lr_decay={lr_decay} | batch_size={batch_size}"
            print(log_str)
            log_file.write(log_str + "\n")
            runner = ModelRunner(runner_id, NUM_EPOCHS, learning_rate=lr, lr_decay=lr_decay, batch_size=batch_size)
            runner.main_loop(NUM_EPOCHS)