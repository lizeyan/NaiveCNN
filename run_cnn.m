clear

model = Network();
model.add(Convolution('conv1', 7, 1, 8, 1, 0.1));
model.add(Relu('relu1'));
model.add(Pooling('pool1', 2, 0));  % output shape: 14 x 14 x 4 x N
model.add(Convolution('conv2', 7, 8, 16, 1, 0.1));
model.add(Relu('relu2'));
model.add(Pooling('pool2', 2, 0)); % output shape: 7 x 7 x 4 x N
model.add(Linear('fc3', 64 * 4, 10, 0.1));
loss = SoftmaxLoss('loss');

% load data
if ~exist('data/mnist.mat', 'file')
	get_mnist('data')
end
load('data/mnist.mat');
train_data = mnist.train_data / 255;
test_data = mnist.test_data / 255;
train_label = mnist.train_label;
test_label = mnist.test_label;

update.learning_rate = 0.0001;
update.weight_decay = 0.005;
update.momentum = 0.90;

solver.update = update;
solver.shuffle = true;
solver.batch_size = 64;
solver.display_freq = 100;
solver.max_iter = 50000;
solver.test_freq = 1000;
solver.snapshot_freq = 100000000;
tic
solve_cnn(model, loss, train_data, train_label, ...
      test_data, test_label, solver);
toc
