from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends._nnapi.prepare as prepare
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from pathlib import Path
from torch.utils.mobile_optimizer import optimize_for_mobile
import yaml

print(torch.version.__version__)
print(torch.__file__)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.dropout1(x)
        x = torch.reshape(x, (1, -1))
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def save_model(
        model,
        model_state_path,
        model_non_optimized_path,
        model_path,
        model_mobile_path,
        model_ops_yaml_path):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    model_state_path = script_dir + '/' + model_state_path
    print(model_state_path)
    Path(os.path.dirname(model_state_path)).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_state_path)
    model_script = torch.jit.script(model)
    model_non_optimized_path = script_dir + '/' + model_non_optimized_path
    torch.jit.save(model_script, model_non_optimized_path)
    model_script_opt = optimize_for_mobile(model_script)
    model_path = script_dir + '/' + model_path
    print(model_path)
    Path(os.path.dirname(model_path)).mkdir(parents=True, exist_ok=True)
    torch.jit.save(model_script_opt, model_path)
    model_mobile_path = script_dir + '/' + model_mobile_path
    print(model_mobile_path)
    model_script_opt._save_for_lite_interpreter(model_mobile_path)

    ops = torch.jit.export_opnames(model_script_opt)
    # FIXME: remove it after fix
    ops.append('aten::set_.source_Storage')
    model_ops_yaml_path = script_dir + '/' + model_ops_yaml_path
    with open(model_ops_yaml_path, 'w') as output:
        yaml.dump(ops, output)


def make_nnapi_model(model):
    input_float = torch.zeros(1, 1, 28, 28)
    input_tensor = input_float

    # Many NNAPI backends prefer NHWC tensors, so convert our input to channels_last,
    # and set the "nnapi_nhwc" attribute for the converter.
    input_tensor = input_tensor.contiguous(memory_format=torch.channels_last)
    input_tensor.nnapi_nhwc = True

    # Trace the model.  NNAPI conversion only works with TorchScript models,
    # and traced models are more likely to convert successfully than scripted.
    with torch.no_grad():
        traced = torch.jit.trace(model, input_tensor)
    nnapi_model = torch.backends._nnapi.prepare.convert_model_to_nnapi(traced, input_tensor)

    return nnapi_model



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 1)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--skip-training', action='store_true', default=False,
                        help='Skip training: no loading dataset, no training, no testing')
    parser.add_argument('--seed',
                        type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval',
                        type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model-state',
                        type=str, default='output/mnist_state.pt',
                        help='model state path')
    parser.add_argument('--save-model-non-optimized',
                        type=str, default='output/mnist_non_optimized.pt',
                        help='model non optimized path')
    parser.add_argument('--save-model',
                        type=str, default='output/mnist.pt',
                        help='model path')
    parser.add_argument('--save-model-mobile',
                        type=str, default='output/mnist.ptl',
                        help='mobile model path')
    parser.add_argument('--save-model-ops',
                        type=str, default='output/mnist_ops.yaml',
                        help='model ops path')
    parser.add_argument('--save-quantized-model-state',
                        type=str, default='output/mnist_quantized_state.pt',
                        help='quantized model state path')
    parser.add_argument('--save-quantized-model-non-optimized',
                        type=str, default='output/mnist_quantized_non_optimized.pt',
                        help='quantized model non optimized path')
    parser.add_argument('--save-quantized-model',
                        type=str, default='output/mnist_quantized.pt',
                        help='quantized model path')
    parser.add_argument('--save-quantized-model-mobile',
                        type=str, default='output/mnist_quantized.ptl',
                        help='mobile quantized model path')
    parser.add_argument('--save-quantized-model-ops',
                        type=str, default='output/mnist_quantized_ops.yaml',
                        help='quantized model ops path')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    model = Net().to(device)

    if not args.skip_training:
        train_kwargs = {'batch_size': args.batch_size}
        test_kwargs = {'batch_size': args.test_batch_size}
        if use_cuda:
            cuda_kwargs = {'num_workers': 1,
                           'pin_memory': True,
                           'shuffle': True}
            train_kwargs.update(cuda_kwargs)
            test_kwargs.update(cuda_kwargs)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset1 = datasets.MNIST(
            '../dataset',
            train=True,
            download=True,
            transform=transform)
        dataset2 = datasets.MNIST(
            '../dataset',
            train=False,
            transform=transform)
        train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
        test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        for epoch in range(1, args.epochs + 1):
            print("Epoch {}/{}".format(epoch, args.epochs + 1))
            train(args, model, device, train_loader, optimizer, epoch)
            test(model, device, test_loader)
            scheduler.step()

    model.eval()
    save_model(
        model,
        args.save_model_state,
        args.save_model_non_optimized,
        args.save_model,
        args.save_model_mobile,
        args.save_model_ops)

    nnapi_model = make_nnapi_model(model)
    nnapi_model_script = torch.jit.script(nnapi_model)
    script_dir = os.path.dirname(os.path.realpath(__file__))
    nnapi_model_path = script_dir + "/output/mnist-nnapi.pt"
    nnapi_model_script.save(nnapi_model_path)
    nnapi_model_mobile_path = script_dir + "/output/mnist-nnapi.ptl"
    nnapi_model._save_for_lite_interpreter(nnapi_model_mobile_path)

    model_quantized = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.Conv2d},
        dtype=torch.qint8)

    save_model(
        model_quantized,
        args.save_quantized_model_state,
        args.save_quantized_model_non_optimized,
        args.save_quantized_model,
        args.save_quantized_model_mobile,
        args.save_quantized_model_ops)


if __name__ == '__main__':
    main()
