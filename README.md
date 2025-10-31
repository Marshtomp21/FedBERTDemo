# **FedBERT Demo Version**

This project is dedicated to reproducing the core mechanisms of FedBERT.

### Paper
This is the demo version of the FedBERT model described in the paper:
https://dl.acm.org/doi/10.1145/3510033

### Difference Between Demo and the experiment in the Paper
* Instead of using Fairseq, this demo version uses the Huggingface Transformers library for easier usage.
* Parallel training strategy has not yet been implemented.
* Using the Flask framework to simulate network communication.

### To Do List
1. [x] Apply the framework to real datasets such as WikiText-103.
2. [x] Implement sequential training strategy.
3. [ ] Implement model validation and evaluation.
4. [ ] Implement parallel training strategy.

### Setup
* Ensure that local port 2778 is not in use.
* Start the server: Run `server.py` in a terminal.
* Start the client: Run `run_training.py` in another terminal.