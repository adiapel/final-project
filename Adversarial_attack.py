import torch
import torch.nn.functional as F
import GUI
import models
import numpy as np

ALPHA = 0.01
ITERS = 20


def fgsm_attack(image, epsilon, data_grad, is_mnist):
    """
    this function creates the FGSM attack on an image
    :param image: an image to be changed
    :param epsilon:
    :param data_grad: the gradient
    :return:
    """
    perturbed_image = image + epsilon * data_grad.sign()
    if is_mnist:
        return torch.clamp(perturbed_image, 0, 1)
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1).to(image.device)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1).to(image.device)
    clamp_min = (0 - mean) / std
    clamp_max = (1 - mean) / std
    return torch.max(torch.min(perturbed_image, clamp_max), clamp_min)


def pgd_attack(model, images, labels, epsilon, alpha=ALPHA, iters=ITERS):
    """
    this function creates the PGD attack on an image
    :param model:
    :param images:
    :param labels:
    :param alpha:
    :param iters:
    :return:
    """
    original_images = images.clone().detach()
    for i in range(iters):
        images.requires_grad = True
        output = model(images)
        loss = F.cross_entropy(output, labels)
        model.zero_grad()
        loss.backward()
        adversarial_images = images + alpha * images.grad.sign()
        eta = torch.clamp(adversarial_images - original_images, min=-epsilon, max=epsilon)
        images = torch.clamp(original_images + eta, min=0, max=1).detach()
    return images


def adversarial_attack(is_mnist, is_fgsm):
    """
    this function creates the model using the requested dataset and uses FGSM/PGD attack
    :param
    :param is_mnist: if the data set is MNIST or CIFAR-10
    :return:
    """
    images = []
    if is_mnist:
        model, test_loader = models.create_MNIST_model()
    else:
        model, test_loader = models.create_CIFAR_10_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    data_iter = iter(test_loader)
    for i in range(len(models.EPSILONS)):
        image, label = next(data_iter)
        image, label = image.to(device), label.to(device)
        image.requires_grad = True
        output = model(image)
        init_pred = output.argmax(dim=1)
        # compute loss
        loss = F.cross_entropy(output, label)
        model.zero_grad()
        loss.backward()
        if is_fgsm:
            perturbed_data = fgsm_attack(image, models.EPSILONS[i], image.grad, is_mnist)
        else:
            perturbed_data = pgd_attack(model, image, label, models.EPSILONS[i])
        output = model(perturbed_data)
        final_pred = output.argmax(dim=1)
        if not is_mnist:
            image = un_normalized(image)
            perturbed_data = un_normalized(perturbed_data)
        images.append([image, init_pred, perturbed_data, final_pred])
    GUI.show_results(images, is_mnist)


def un_normalized(image):
    """
    this method un-normalize the data
    :param image:
    :return:
    """
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])
    image = image[0].detach().cpu().numpy().transpose((1, 2, 0))
    image = image * std + mean
    return np.clip(image, 0, 1)
