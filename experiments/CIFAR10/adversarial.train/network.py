import config
from base_model.cifar_resnet18 import cifar_resnet18

def create_network():
    net = cifar_resnet18(num_class = 10, expansion = 2)

    return net

if __name__ == '__main__':
    net = create_network()
    print(net)
