**References:**  
1. Ian Pointer. Programming PyTorch for Deep Learning
2. Eli Stevens. Deep Learning with PyTorch
3. Pointer. Programming PyTorch for Deep Learning. 2020  

* **Freezing model**  
`for name, param in transfer_model.nammed_parameters():`  
`   if ("bn" not in name):`  # excluding Batch normalization layers for example  
`       param.requires_grad = False`  

* **Inception/ GoogLeNet**  
Run different convolutions (for different type resolutions) and then concatenate them.

* **Adaptive layers**  
For working with inputs with different shapes Adaptive Pooling layers may be used.

* **Use nn.Sequential()**  
Use this to create a chain of layers in order to break your model into more logical arrangements.  
`class CNNNet(nn.Module):`  
`def __init__(self, num_classes=2):`  
`super(CNNNet, self).__init__()`  
`self.features = nn.Sequential(`  
`nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)`  
`nn.ReLU()`  
`nn.MaxPool2d(kernel_size=3, stride=2)`  
`nn.Conv2d(64, 192, kernel_size=5, padding=2)`  
`nn.ReLU()`  
`nn.MaxPool2d(kernel_size=3, stride=2)`  
`nn.Conv2d(192, 384, kernel_size=3, padding=1)`  
`nn.ReLU()`  
`nn.Conv2d(384, 256, kernel_size=3, padding=1)`  
`nn.ReLU()`  
`nn.Conv2d(256, 256, kernel_size=3, padding=1)`  
`nn.ReLU()`  
`nn.MaxPool2d(kernel_size=3, stride=2)`  
`)`  
`self.avgpool = nn.AdaptiveAvgPool2d((6, 6))`  
`self.classifier = nn.Sequential(`  
`nn.Dropout()`  
`nn.Linear(256 * 6 * 6, 4096)`  
`nn.ReLU()`  
`nn.Dropout()`  
`nn.Linear(4096, 4096)`  
`nn.ReLU()`  
`nn.Linear(4096, num_classes)`  
`)`  
`def forward(self, x):`  
`x = self.features(x)`  
`x = self.avgpool(x)`  
`x = torch.flatten(x, 1)`  
`x = self.classifier(x)`  
`return x`  

* **L2 Regularization:**  
`l2_lambda = 0.9`  
`l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())`  
`loss = loss + l2_lambda * l2_norm`  
Replace pow(2.0) with abs() for L1 regularization.  
Optimizer in PyTorch already has a weight_decay parameter, that corresponds to 2 * lambda, that directly performs weight decay during the update.  

* **Skip Connections**  
`def forward(self, x):`  
`out = F.max_pool2d(torch.relu(self.conv1(x)), 2)`  
`out1 = out`  
`out = F.max_pool2d(torch.relu(self.conv2(out)), 2)`  
`out = F.max_pool2d(torch.relu(self.conv3(out + out1)), 2)`  
`out = out.view(-1, 8 * 8 * self.n_chans1 // 2)`  
`out = torch.tanh(self.fc1(out))`  
`out = self.fc2(out)`  
`return out`  
A skip connection, or a sequence of skip connections in a deep network, creates a direct path from the deeper parameters to the loss. This makes their contribution to the gradient of the loss more direct, as partial derivatives of the loss with respect to those parameters have a change not to be multiplied by a long chain of other operations.  

* **Very Deep Models**  
`class ResBlock(nn.Module):`  
`def __init__(self, n_chans):`  
`super(ResBlock, self).__init__()`  
`self.conv = nn.Conv2d(n_chans, n_chans, kernel_size=3, padding=1)`  
`self.batch_norm = nn.BatchNorm2d(num_features=n_chans)`  
`def forward(self, x):`  
`out = self.conv(x)`  
`out = self.batch_norm(out)`  
`out = torch.relu(out)`  
`return out + x`  
`class Net(nn.Module):`  
`def __init__(self, n_chans1=32, n_blocks=10):`  
`super(Net, self).__init__()`  
`self.n_chans1 = n_chans1`  
`self.conv1 = nn.Conv2d(3, n_chans1, kernel_size=3, padding=1)`  
`self.resblocks = nn.Sequential(* [ResBlock(n_chans=n_chans1)] * n_blocks)`  
`self.fc1 = nn.Linear(8 * 8 * n_chans1, 32)`  
`self.fc2 = nn.Linear(32, 2)`  
`def forward(self, x):`  
`out = F.max_pool2d(torch.relu(self.conv1(x)), 2)`  
`out = self.resblocks(out)`  
`out = F.max_pool2d(out, 2)`  
`out = out.view(-1, 8 * 8 * self.n_chans1)`  
`out = torch.tanh(self.fc1(out))`  
`out = self.fc2(out)`  
`return out`  
`model = Net(n_chans1=32, n_blocks=100)`  

* **Start Small and Get Bigger**  
[3, page 83] On example of classifying pictures:  
1. Start with small pictures' size 64x64.  
2. Train a model.
3. Take the same pictures but with size 128x128.  
4. Again train model but not from scratch, using learned parameters from previous training step.  

* **Ensemble Neural Networks**  
`models_ensemble = [models.resnet50().to(device), models.resnet50().to(device)]`  
`predictions = [F.softmax(m(torch.rand(1,3,224,244).to(device))) for m in models_ensemble]`  
`avg_prediction = torch.stack(predictions).mean(0).argmax()`  

* **Differential Learning Rates**  
`optim.SGD([`  
                `{'params': model.base.parameters()},`  
                `{'params': model.classifier.parameters(), 'lr': 1e-3}`  
            `], lr=1e-2, momentum=0.9)`  


* **Flame graph**  
[3, page 143]

* **Tensorboard**  
Run in command line: `tensorboard --logdir=runs`  

* **Gradient Checkpointing**  
[3, page 152]  

* **Mixup**  
[3, page 179]  



