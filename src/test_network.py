'''
Created on Sep 26, 2016

@author: zhaoxm
'''
import mnist_loader
import network
import network2

if __name__ == '__main__':
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = network.Network([784, 30, 10])
    net.SGD(training_data, 30, 10, 0.5, test_data=test_data)
    print "cross begin:"
    net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
    net.large_weight_initializer()
    net.SGD(training_data, 30, 10, 0.5, evaluation_data=test_data,monitor_evaluation_accuracy=True)
    '''
    Epoch 0: 7927 / 10000
Epoch 1: 8246 / 10000
Epoch 2: 8363 / 10000
Epoch 3: 8433 / 10000
Epoch 4: 9023 / 10000
Epoch 5: 9092 / 10000
Epoch 6: 9145 / 10000
Epoch 7: 9188 / 10000
Epoch 8: 9203 / 10000
Epoch 9: 9213 / 10000
Epoch 10: 9234 / 10000
Epoch 11: 9245 / 10000
Epoch 12: 9267 / 10000
Epoch 13: 9266 / 10000
Epoch 14: 9286 / 10000
Epoch 15: 9314 / 10000
Epoch 16: 9293 / 10000
Epoch 17: 9318 / 10000
Epoch 18: 9315 / 10000
Epoch 19: 9342 / 10000
Epoch 20: 9342 / 10000
Epoch 21: 9346 / 10000
Epoch 22: 9357 / 10000
Epoch 23: 9351 / 10000
Epoch 24: 9361 / 10000
Epoch 25: 9347 / 10000
Epoch 26: 9377 / 10000
Epoch 27: 9370 / 10000
Epoch 28: 9375 / 10000
Epoch 29: 9377 / 10000
cross begin:
Epoch 0 training complete
Accuracy on evaluation data: 9127 / 10000

Epoch 1 training complete
Accuracy on evaluation data: 9248 / 10000

Epoch 2 training complete
Accuracy on evaluation data: 9365 / 10000

Epoch 3 training complete
Accuracy on evaluation data: 9384 / 10000

Epoch 4 training complete
Accuracy on evaluation data: 9415 / 10000

Epoch 5 training complete
Accuracy on evaluation data: 9454 / 10000

Epoch 6 training complete
Accuracy on evaluation data: 9448 / 10000

Epoch 7 training complete
Accuracy on evaluation data: 9456 / 10000

Epoch 8 training complete
Accuracy on evaluation data: 9498 / 10000

Epoch 9 training complete
Accuracy on evaluation data: 9488 / 10000

Epoch 10 training complete
Accuracy on evaluation data: 9497 / 10000

Epoch 11 training complete
Accuracy on evaluation data: 9517 / 10000

Epoch 12 training complete
Accuracy on evaluation data: 9511 / 10000

Epoch 13 training complete
Accuracy on evaluation data: 9504 / 10000

Epoch 14 training complete
Accuracy on evaluation data: 9533 / 10000

Epoch 15 training complete
Accuracy on evaluation data: 9515 / 10000

Epoch 16 training complete
Accuracy on evaluation data: 9471 / 10000

Epoch 17 training complete
Accuracy on evaluation data: 9512 / 10000

Epoch 18 training complete
Accuracy on evaluation data: 9520 / 10000

Epoch 19 training complete
Accuracy on evaluation data: 9512 / 10000

Epoch 20 training complete
Accuracy on evaluation data: 9533 / 10000

Epoch 21 training complete
Accuracy on evaluation data: 9515 / 10000

Epoch 22 training complete
Accuracy on evaluation data: 9536 / 10000

Epoch 23 training complete
Accuracy on evaluation data: 9514 / 10000

Epoch 24 training complete
Accuracy on evaluation data: 9519 / 10000

Epoch 25 training complete
Accuracy on evaluation data: 9514 / 10000

Epoch 26 training complete
Accuracy on evaluation data: 9530 / 10000

Epoch 27 training complete
Accuracy on evaluation data: 9507 / 10000

Epoch 28 training complete
Accuracy on evaluation data: 9531 / 10000

Epoch 29 training complete
Accuracy on evaluation data: 9526 / 10000
    '''