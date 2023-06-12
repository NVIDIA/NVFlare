import os
import matplotlib.pyplot as plt


def plot_acc(path):

    log_path = os.path.join(path, "simulate_job", "app_site-1", "log.txt")
    acc = []

    with open(log_path, encoding='utf-8') as f:
        for line in f.readlines():
            str_split = line.split(' ')
            if len(str_split) > 5:
                if str_split[-2] == "train_accuracy:":
                    acc.append(float(str_split[-1]))

    print(acc)
    ep = [i*10 for i in range(len(acc))]
    plt.plot(ep, acc)
    plt.xlabel("Local training epoch")
    plt.ylabel("Training accuracy")
    plt.title("One-shot VFL")
    plt.savefig("figs/oneshotVFL_results.png")