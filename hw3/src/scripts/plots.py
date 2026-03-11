from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import glob
import re

CARTPOLE_DIR = Path("exp/CartPole-v1_dqn_sd1_20260307_165216") / "log.csv"
LUNARLANDER_DIR = Path("exp/LunarLander-v2_dqn_sd1_20260307_171419") / "log.csv"

def plot_learning_curve(logdir, name):
    #logdir = Path("exp/CartPole-v1_dqn_sd1_20260307_165216") / "log.csv"
    fig, (axes1) = plt.subplots(1, figsize=(10,8))

    axes1.set_xlabel("Environment Steps")
    axes1.set_ylabel("Eval Average Return")
    axes1.set_title(f"Average Return vs Environment Steps {name}")

    df = pd.read_csv(logdir)
    eval_data = df[df["Eval_AverageReturn"].notna()]

    axes1.plot(eval_data["step"], eval_data["Eval_AverageReturn"], linewidth=1)
    
    plt.tight_layout()
    plt.savefig(f"exp1_learning_curves_{name}")
    plt.show()
    return fig

def plot_learning_curve_2_5():
    fig, (axes1, axes2) = plt.subplots(1, 2, figsize=(10,8))

    axes1.set_xlabel("Environment Steps")
    axes1.set_ylabel("Average Return")
    axes1.set_title(f"Average Return vs Environment Steps MsPacman")
    axes2.set_xlabel("Environment Steps")
    axes2.set_ylabel("Average Return")
    axes2.set_title(f"Average Return vs Environment Steps LunarLander-v2")

    logdir = Path("exp/MsPacman_dqn_sd1_20260308_051143") / "log.csv"
    df = pd.read_csv(logdir)
    eval_data = df[df["Eval_AverageReturn"].notna()]
    train_data = df[df["Train_EpisodeReturn"].notna()]

    axes1.plot(eval_data["step"], eval_data["Eval_AverageReturn"], linewidth=1, zorder=5, label="eval_return")
    axes1.plot(train_data["step"], train_data["Train_EpisodeReturn"], linewidth=0.5, alpha = 0.5, label="train_return")
    axes1.legend()

    logdir = Path("exp/LunarLander-v2_dqn_sd1_20260307_171419") / "log.csv"
    df = pd.read_csv(logdir)
    eval_data = df[df["Eval_AverageReturn"].notna()]
    train_data = df[df["Train_EpisodeReturn"].notna()]

    axes2.plot(eval_data["step"], eval_data["Eval_AverageReturn"], linewidth=1, zorder=5, label="eval_return")
    axes2.plot(train_data["step"], train_data["Train_EpisodeReturn"], linewidth=0.5, alpha = 0.5, label="train_return")
    axes2.legend()

    plt.tight_layout()
    plt.savefig(f"learning_curves_2_5")
    plt.show()
    return fig

def plot_learning_curve_2_6():
    logdirs = glob.glob(f"exp/LunarLander-v2_dqn_lunarlander_v2_target*/")
    fig, (axes1) = plt.subplots(1, figsize=(10,8))

    axes1.set_xlabel("Environment Steps")
    axes1.set_ylabel("Average Return")
    axes1.set_title(f"Average Return vs Environment Steps Target Update Period Comparison")

    for logdir in logdirs:
        path = Path(logdir) / "log.csv"
        target_update_period = None
        match = re.search(r'target_update_period_(\d+)', logdir)
        if match:
            target_update_period = match.group(1)
        df = pd.read_csv(path)
        eval_data = df[df["Eval_AverageReturn"].notna()]
        axes1.plot(eval_data["step"], eval_data["Eval_AverageReturn"], linewidth=0.5, label= target_update_period)
    axes1.legend()

    plt.tight_layout()
    plt.savefig(f"learning_curves_2_6")
    plt.show()
    return fig

def plot_learning_curve_3_4():
    logdir = Path("exp/HalfCheetah-v4_sac_sd1_20260310_155926") / "log.csv"
    fig, (axes) = plt.subplots(1, figsize=(10,8))
    axes.set_xlabel("Environment Steps")
    axes.set_ylabel("Average Return")
    axes.set_title(f"Average Return vs Environment Steps SAC HalfCheetah-v4")
    df = pd.read_csv(logdir)
    eval_data = df[df["Eval_AverageReturn"].notna()]

    axes.plot(eval_data["step"], eval_data["Eval_AverageReturn"], linewidth=1)

    plt.tight_layout()
    plt.savefig(f"learning_curves_3_4")
    plt.show()
    return fig

def plot_learning_curve_3_5():
    logdir_1 = Path("exp/HalfCheetah-v4_sac_sd1_20260310_155926") / "log.csv"
    logdir_2 = Path("exp/HalfCheetah-v4_sac_autotune_sd1_20260311_061854") / "log.csv"

    fig, (axes1, axes2) = plt.subplots(1, 2, figsize=(15,8))
    axes1.set_xlabel("Environment Steps")
    axes1.set_ylabel("Average Return")
    axes1.set_title(f"Average Return vs Environment Steps SAC HalfCheetah-v4 Temperature Comparison")

    axes2.set_xlabel("Environment Steps")
    axes2.set_ylabel("Temperature")
    axes2.set_title(f"Temperature vs Environment Steps SAC HalfCheetah-v4 Auto-tune Temperature")

    df1 = pd.read_csv(logdir_1)
    eval_data_1 = df1[df1["Eval_AverageReturn"].notna()]
    df2 = pd.read_csv(logdir_2)
    eval_data_2 = df2[df2["Eval_AverageReturn"].notna()] 

    axes1.plot(eval_data_1["step"], eval_data_1["Eval_AverageReturn"], linewidth=1, label="fixed temperature")  
    axes1.plot(eval_data_2["step"], eval_data_2["Eval_AverageReturn"], linewidth=1, label="auto-tune temperature")
    axes1.legend()

    alpha_data = df2[df2["alpha"].notna()]
    axes2.plot(alpha_data["step"], alpha_data["alpha"], linewidth=1)

    plt.tight_layout()
    plt.savefig(f"learning_curves_3_5")
    plt.show()
    return fig

def plot_learning_curve_3_6():
    logdir1 = Path("exp/Hopper-v4_sac_singleq_sd1_20260311_001448") / "log.csv"
    logdir2 = Path("exp/Hopper-v4_sac_clipq_sd1_20260311_032258") / "log.csv"

    fig, (axes1, axes2) = plt.subplots(1, 2, figsize=(15,8))
    axes1.set_xlabel("Environment Steps")
    axes1.set_ylabel("Average Return")
    axes1.set_title(f"Average Return vs Environment Steps SAC Hopper-v4")

    axes2.set_xlabel("Environment Steps")
    axes2.set_ylabel("Q-value")
    axes2.set_title(f"Q-values vs Environment Steps SAC Hopper-v4")

    df1 = pd.read_csv(logdir1)
    eval_data_1 = df1[df1["Eval_AverageReturn"].notna()]
    q_values_1 = df1[df1["q_values"].notna()]
    df2 = pd.read_csv(logdir2)
    eval_data_2 = df2[df2["Eval_AverageReturn"].notna()]
    q_values_2 = df2[df2["q_values"].notna()]

    axes1.plot(eval_data_1["step"], eval_data_1["Eval_AverageReturn"], linewidth=1, label="single Q")
    axes1.plot(eval_data_2["step"], eval_data_2["Eval_AverageReturn"], linewidth=1, label="clipped double Q")
    axes1.legend()

    axes2.plot(q_values_1["step"], q_values_1["q_values"], linewidth=1, label="single Q")
    axes2.plot(q_values_2["step"], q_values_2["q_values"], linewidth=1, label="clipped double Q")
    axes2.legend()

    plt.tight_layout()
    plt.savefig(f"learning_curves_3_6")
    plt.show()
    return fig

if __name__ == "__main__":
    #plot_learning_curve(LUNARLANDER_DIR, "lunarlander")
    plot_learning_curve_3_5()
