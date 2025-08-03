import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np
import torch
import models

selected_item = None

FGSM_MNIST_STR = "FGSM with MNIST dataset"
FGSM_CIFAR10_STR = "FGSM with CIFAR-10 dataset"
PGD_MNIST_STR = "PGD with MNIST dataset"
PGD_CIFAR10_STR = "PGD with CIFAR-10 dataset"
DBP_IRIS_STR = "DBP with IRIS dataset"
DBP_WINE_STR = "DBP with WINE dataset"

CIFAR_RESULTS = ["airplane", "automobile", "bird", 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

COLS = 2


def create_user_interface():
    """
    this method creates the user interface for the program
    :return:
    """

    def get_item():
        """
        this method gets user's selection from the options
        :return:
        """
        global selected_item
        selection = listbox.curselection()
        if selection:
            selected_item = listbox.get(selection[0])
            window.destroy()

    # create a window for user selection
    window = tk.Tk()
    window.title("Hello")
    window.geometry('300x250+600+250')
    label = tk.Label(window, text="choose an option for display: ")
    label.pack(pady=5)
    # create options list and present it and in the window
    options = [FGSM_MNIST_STR, FGSM_CIFAR10_STR, PGD_MNIST_STR, PGD_CIFAR10_STR, DBP_IRIS_STR, DBP_WINE_STR]
    listbox = tk.Listbox(window, height=len(options), width=50)
    for option in options:
        listbox.insert(tk.END, option)
    listbox.pack(pady=5)
    # create selection button
    select_button = tk.Button(window, text="select", command=get_item)
    select_button.pack(pady=5)
    result = tk.Label(window, text="")
    result.pack(pady=10)
    window.mainloop()
    return selected_item


def show_results(images, is_mnist):
    """
    this method represents the images before and after the attack to the user
    :return:
    """
    n = len(images)
    fig, axes = plt.subplots(n, COLS, figsize=(10, 11))
    for i in range(n):
        epsilon = models.EPSILONS[i]
        data, orig_label, adv_data, final_label = images[i][0:4]
        data = dim_helper(data, is_mnist)
        adv_data = dim_helper(adv_data, is_mnist)
        orig_label = orig_label[0].item()
        orig_label = orig_label if is_mnist else CIFAR_RESULTS[orig_label]
        final_label = final_label[0].item()
        final_label = final_label if is_mnist else CIFAR_RESULTS[final_label]
        ax1 = axes[i, 0]
        ax2 = axes[i, 1]
        ax1.axis("off")
        ax2.axis("off")
        ax1.imshow(data, cmap='gray' if is_mnist else None)
        ax1.set_title(f"Original", fontsize=9)
        ax2.imshow(adv_data, cmap='gray' if is_mnist else None)
        ax2.set_title(f"Adversarial", fontsize=9)
        row_tite = f"ε= {epsilon:.2f}, Original Label: {orig_label}, Predicted: {final_label}"
        fig.text(0.5, ax1.get_position().y1 + 0.015, row_tite, ha='center', va='bottom', fontsize=10)
    plt.subplots_adjust(wspace=0.3, hspace=0.7)
    plt.show()


def dim_helper(img, if_mnist):
    """
    this is a helper method which fixes image's dim
    :param img:
    :param if_mnist:
    :return:
    """
    if isinstance(img, torch.Tensor):
        if img.ndim == 4:
            img = img[0]
        img = img.detach().cpu().numpy()
    if if_mnist:
        img = np.transpose(img, (1, 2, 0))
    else:
        if img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
    return img


def plot_decision_boundary(model, x, y, title):
    """
    this method is a helper method for showing the SVM model
    :param model:
    :param x:
    :param y:
    :param title:
    :return:
    """
    h = .02
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.contourf(xx, yy, z, alpha=0.3)
    plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='k')
    plt.title(title)


def show_SVM(model, x_train, y_train, poisoned_model, x_poisoned, y_poisoned):
    """
    this method shows the SVM model results in the plot
    :param model:
    :param x_train:
    :param y_train:
    :param poisoned_model:
    :param x_poisoned:
    :param y_poisoned:
    :return:
    """
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plot_decision_boundary(model, x_train, y_train, "Original SVM")
    plt.subplot(1, 2, 2)
    plot_decision_boundary(poisoned_model, x_poisoned, y_poisoned, "Poisoned SVM")
    plt.show()
