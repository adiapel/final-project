import GUI
import Adversarial_attack
import DBP

if __name__ == '__main__':
    """
    main method of the program
    """
    selected_program = GUI.create_user_interface()
    if selected_program == GUI.FGSM_MNIST_STR:
        Adversarial_attack.adversarial_attack(True, True)
    if selected_program == GUI.FGSM_CIFAR10_STR:
        Adversarial_attack.adversarial_attack(False, True)
    if selected_program == GUI.PGD_MNIST_STR:
        Adversarial_attack.adversarial_attack(True, False)
    if selected_program == GUI.PGD_CIFAR10_STR:
        Adversarial_attack.adversarial_attack(False, False)
    if selected_program == GUI.DBP_IRIS_STR:
        DBP.create_model(True)
    if selected_program == GUI.DBP_WINE_STR:
        DBP.create_model(False)
    else:
        exit()
