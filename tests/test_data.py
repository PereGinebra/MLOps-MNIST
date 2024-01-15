from corrupt_mnist.data.dataset import mnist
from numpy import unique

def test_data():
    train, test = mnist()
    assert len(train) == 30000, 'Unexpected size for training dataset'
    assert len(test) == 5000, 'Unexpected size for test dataset'
    assert [(image.shape, type(label.item())) for image, label in train] == [((28,28),int)]*len(train), 'Unexpected shape for some training datapoint'
    assert [(image.shape, type(label.item())) for image, label in test] == [((28,28),int)]*len(test), 'Unexpected shape for some testing datapoint'
    assert unique([label for _, label in train]).tolist() == list(range(10)), 'Missing representation from some label or wrong label set'