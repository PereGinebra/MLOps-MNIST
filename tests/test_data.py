import os.path

import pytest

from corrupt_mnist.data.dataset import mnist
from numpy import unique

@pytest.mark.skipif(not os.path.exists('data/processed/train_images.pt'), reason="Train image file not found")
@pytest.mark.skipif(not os.path.exists('data/processed/train_target.pt'), reason="Train label file not found")
@pytest.mark.skipif(not os.path.exists('data/processed/test_images.pt'), reason="Test image file not found")
@pytest.mark.skipif(not os.path.exists('data/processed/test_target.pt'), reason="Test label file not found")
def test_data():
    train, test = mnist()
    assert len(train) == 50000, 'Unexpected size for training dataset'
    assert len(test) == 5000, 'Unexpected size for test dataset'
    assert [(image.shape, type(label.item())) for image, label in train] == [((28,28),int)]*len(train), 'Unexpected shape for some training datapoint'
    assert [(image.shape, type(label.item())) for image, label in test] == [((28,28),int)]*len(test), 'Unexpected shape for some testing datapoint'
    assert unique([label for _, label in train]).tolist() == list(range(10)), 'Missing representation from some label or wrong label set'