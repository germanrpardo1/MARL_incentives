"""Schedule runs."""

import subprocess

if __name__ == "__main__":
    subprocess.run(["python", "scripts/qlearning_binary_state_exp_replay.py"])

    subprocess.run(["python", "scripts/qlearning_no_state_exp_replay.py"])
