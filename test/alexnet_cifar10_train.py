import alexnet_cifar10.input as input

if __name__ == '__main__':
    train_images, train_labels, test_images, test_labels = input.load_cifar10(
        '/Users/Zhang/Research/Deep Learning Dataset/CIFAR/cifar-10-batches-py')

    print 0
