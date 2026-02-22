from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import glob
import json

def create_learning_curves_exp1():

    dirs = glob.glob(f"exp/CartPole-v0_cartpole*/")
    fig, (axes1, axes2) = plt.subplots(1, 2, figsize=(15,8))

    axes1.set_xlabel("Train EnvstepsSoFar")
    axes1.set_ylabel("Eval Average Return")
    axes1.set_title("Average Return vs Environment Steps Small Batch")

    axes2.set_xlabel("Train EnvstepsSoFar")
    axes2.set_ylabel("Eval Average Return")
    axes2.set_title("Average Return vs Environment Steps Large Batch")

    for dir in dirs:
        log = Path(dir) / "log.csv"
        flags = Path(dir) / "flags.json"
        df = pd.read_csv(log)
        with open(flags) as f:
            label = json.load(f)["exp_name"]
        
        if "lb" in label:
            # plot in axes 2
            axes2.plot(df["Train_EnvstepsSoFar"], df["Eval_AverageReturn"], linewidth=1, label=label)
        else:
            # plot in axes 1
            axes1.plot(df["Train_EnvstepsSoFar"], df["Eval_AverageReturn"], linewidth=1, label=label)
    axes1.legend()
    axes2.legend()
    plt.tight_layout()
    plt.savefig("exp1_learning_curves_cartpole")
    plt.show()
    return fig

def create_learning_curves_exp2():
    dir = "exp/HalfCheetah-v4_cheetah_baseline_sd1_20260222_015008"
    log = Path(dir) / "log.csv"

    df = pd.read_csv(log)
    fig, (axes1, axes2) = plt.subplots(1, 2, figsize=(15,8))

    axes1.set_xlabel("Train EnvstepsSoFar")
    axes1.set_ylabel("Baseline Loss")
    axes1.set_title("Baseline Loss Learning Curve")

    axes2.set_xlabel("Train EnvstepsSoFar")
    axes2.set_ylabel("Eval Average Return")
    axes2.set_title("Eval Return Learning Curve")

    axes1.plot(df["Train_EnvstepsSoFar"], df["Baseline Loss"], linewidth=1)
    axes2.plot(df["Train_EnvstepsSoFar"], df["Eval_AverageReturn"], linewidth=1)

    plt.tight_layout()
    plt.savefig("exp2_learning_curves_cheetah")
    plt.show()
    return fig

    
if __name__ == "__main__":
    create_learning_curves_exp2()