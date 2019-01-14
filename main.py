import numpy as np
import sys, os

sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from PIL import Image
import pickle
	
def _init_image():
	# it will take several minutes for first time
	(x_train, t_train), (x_test, t_test) = \
		load_mnist(flatten=True, normalize=False) 
	image = x_train[0]
	label = t_train[0]
	print(label)
	print(image.shape)
	image = image.reshape(28,28)
	return image
	
def _show_image(image):
	pil_image = Image.fromarray(np.uint8(image))
	pil_image.show()	

	
def _get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

	
def _init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

	
def _predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1
    z1 = _sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = _sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = _softmax(a3)

    return y

	
def _sigmoid(x): 
	return 1 / (1 + np.exp(-x))

	
def _identity_function(v):
	return v
	
	
def _softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 
    x = x - np.max(x) # to avoid overflow
    return np.exp(x) / np.sum(np.exp(x))

	
def _show_prediction():
	x, t = _get_data()
	network = _init_network()
	accuracy_cnt = 0
	for i in range(len(x)):
		y = _predict(network, x[i])
		p = np.argmax(y) # get the highest value in the array
		if p == t[i]:
			accuracy_cnt += 1
	print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
	

if __name__ == '__main__':
	image = _init_image()
	#_show_image(image)
	_show_prediction()
