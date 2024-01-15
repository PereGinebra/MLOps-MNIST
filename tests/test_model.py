from corrupt_mnist.models.model import SimpleCNN
import torch

def test_model():
    model = SimpleCNN()
    assert model(torch.rand([1,28,28])).shape == (1,10), 'Unexpected shape for output from a batch of a single image'
    assert model(torch.rand([32,28,28])).shape == (32,10), 'Unexpected shape for output from a batch of 32 images'