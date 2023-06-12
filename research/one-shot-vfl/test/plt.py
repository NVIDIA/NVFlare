import matplotlib.pyplot as plt

acc = [0.2016, 0.5178, 0.7014, 0.8043, 0.8868, 0.9065, 0.9530, 0.9680, 0.9767, 0.9762]

plt.plot(acc)
plt.xlabel("Local training epoch")
plt.ylabel("Training accuracy")
plt.title("One-shot VFL")
plt.savefig("taing_acc.png")