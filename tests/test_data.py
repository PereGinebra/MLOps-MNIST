from corrupt_mnist.data.dataset import mnist
from numpy import unique

def test_data():
    train, test = mnist()
    assert len(train) == 30000
    assert len(test) == 5000
    assert [(image.shape, type(label.item())) for image, label in train] == [((28,28),int)]*len(train)
    assert [(image.shape, type(label.item())) for image, label in test] == [((28,28),int)]*len(test)
    assert unique([label for _, label in train]).tolist() == list(range(10))