# people_politics.py
# predict politics type from sex, age, state, income
# PyTorch 1.12.1-CPU Anaconda3-2020.02  Python 3.7.6
# Windows 10/11 

import numpy as np
import torch as T
device = T.device('cpu')  # apply to Tensor or Module

# -----------------------------------------------------------

class PeopleDataset(T.utils.data.Dataset):
  # sex  age    state    income   politics
  # -1   0.27   0  1  0   0.7610   2
  # +1   0.19   0  0  1   0.6550   0
  # sex: -1 = male, +1 = female
  # state: michigan, nebraska, oklahoma
  # politics: conservative, moderate, liberal

  def __init__(self, src_file):
    all_xy = np.loadtxt(src_file, usecols=range(0,7),
      delimiter="\t", comments="#", dtype=np.float32)
    tmp_x = all_xy[:,0:6]   # cols [0,6) = [0,5]
    tmp_y = all_xy[:,6]     # 1-D

    self.x_data = T.tensor(tmp_x, 
      dtype=T.float32).to(device)
    self.y_data = T.tensor(tmp_y,
      dtype=T.int64).to(device)  # 1-D

  def __len__(self):
    return len(self.x_data)

  def __getitem__(self, idx):
    preds = self.x_data[idx]
    trgts = self.y_data[idx] 
    return preds, trgts  # as a Tuple

# -----------------------------------------------------------

class Net(T.nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.hid1 = T.nn.Linear(6, 10)  # 6-(10-10)-3
    self.hid2 = T.nn.Linear(10, 10)
    self.oupt = T.nn.Linear(10, 3)

    T.nn.init.xavier_uniform_(self.hid1.weight)
    T.nn.init.zeros_(self.hid1.bias)
    T.nn.init.xavier_uniform_(self.hid2.weight)
    T.nn.init.zeros_(self.hid2.bias)
    T.nn.init.xavier_uniform_(self.oupt.weight)
    T.nn.init.zeros_(self.oupt.bias)

  def forward(self, x):
    z = T.tanh(self.hid1(x))
    z = T.tanh(self.hid2(z))
    z = T.log_softmax(self.oupt(z), dim=1)  # NLLLoss() 
    return z

# -----------------------------------------------------------

def accuracy(model, ds):
  # assumes model.eval()
  # item-by-item version
  n_correct = 0; n_wrong = 0
  for i in range(len(ds)):
    X = ds[i][0].reshape(1,-1)  # make it a batch
    Y = ds[i][1].reshape(1)  # 0 1 or 2, 1D
    with T.no_grad():
      oupt = model(X)  # logits form

    big_idx = T.argmax(oupt)  # 0 or 1 or 2
    if big_idx == Y:
      n_correct += 1
    else:
      n_wrong += 1

  acc = (n_correct * 1.0) / (n_correct + n_wrong)
  return acc

# -----------------------------------------------------------

# def accuracy_quick(model, dataset):
  # assumes model.eval()
#   X = dataset[0:len(dataset)][0]
  # Y = T.flatten(dataset[0:len(dataset)][1])
#   Y = dataset[0:len(dataset)][1]
#   with T.no_grad():
#     oupt = model(X)
  # (_, arg_maxs) = T.max(oupt, dim=1)
#   arg_maxs = T.argmax(oupt, dim=1)  # argmax() is new
#   num_correct = T.sum(Y==arg_maxs)
#   acc = (num_correct * 1.0 / len(dataset))
#   return acc.item()

# -----------------------------------------------------------

def main():
  # 0. get started
  print("\nBegin People predict politics type ")
  T.manual_seed(1)
  np.random.seed(1)
  
  # 1. create DataLoader objects
  print("\nCreating People Datasets ")

  train_file = ".\\Data\\people_train.txt"
  train_ds = PeopleDataset(train_file)  # 200 rows

  test_file = ".\\Data\\people_test.txt"
  test_ds = PeopleDataset(test_file)    # 40 rows

  bat_size = 10
  train_ldr = T.utils.data.DataLoader(train_ds,
    batch_size=bat_size, shuffle=True)

# -----------------------------------------------------------

  # 2. create network
  print("\nCreating 6-(10-10)-3 neural network ")
  net = Net().to(device)
  net.train()

# -----------------------------------------------------------

  # 3. train model
  max_epochs = 1000
  ep_log_interval = 100
  lrn_rate = 0.01

  loss_func = T.nn.NLLLoss()  # assumes log_softmax()
  optimizer = T.optim.SGD(net.parameters(), lr=lrn_rate)

  print("\nbat_size = %3d " % bat_size)
  print("loss = " + str(loss_func))
  print("optimizer = SGD")
  print("max_epochs = %3d " % max_epochs)
  print("lrn_rate = %0.3f " % lrn_rate)

  print("\nStarting training ")
  for epoch in range(0, max_epochs):
    # T.manual_seed(epoch+1)  # checkpoint reproducibility
    epoch_loss = 0  # for one full epoch

    for (batch_idx, batch) in enumerate(train_ldr):
      X = batch[0]  # inputs
      Y = batch[1]  # correct class/label/politics

      optimizer.zero_grad()
      oupt = net(X)
      loss_val = loss_func(oupt, Y)  # a tensor
      epoch_loss += loss_val.item()  # accumulate
      loss_val.backward()
      optimizer.step()

    if epoch % ep_log_interval == 0:
      print("epoch = %5d  |  loss = %10.4f" % \
        (epoch, epoch_loss))

      # save training checkpoint
      # dt = time.strftime("%Y_%m_%d-%H_%M_%S")
      # fn = ".\\Log\\" + str(dt) + "-" + \
      #  str(epoch) + "_checkpoint.pt"

      # info_dict = {
      #   'epoch' : epoch,
      #   'torch_random_state' : T.random.get_rng_state(),
      #   'numpy_random_state' : np.random.get_state(),
      #   'net_state' : net.state_dict(),
      #   'optimizer_state' : optimizer.state_dict() 
      # }
      # T.save(info_dict, fn)

      # TODO: check validation loss/accuracy
  print("Training done ")

# -----------------------------------------------------------

  # 4. evaluate model accuracy
  print("\nComputing model accuracy ")
  net.eval()
  acc_train = accuracy(net, train_ds)  # item-by-item
  print("Accuracy on training data = %0.4f" % acc_train)
  acc_test = accuracy(net, test_ds) 
  print("Accuracy on test data = %0.4f" % acc_test)

  # 5. make a prediction
  print("\nPredicting politics for M  30  oklahoma  $50,000: ")
  X = np.array([[-1, 0.30,  0,0,1,  0.5000]], dtype=np.float32)
  X = T.tensor(X, dtype=T.float32).to(device) 

  with T.no_grad():
    logits = net(X)  # do not sum to 1.0
  probs = T.exp(logits)  # sum to 1.0
  probs = probs.numpy()  # numpy vector prints better
  np.set_printoptions(precision=4, suppress=True)
  print(probs)

  # 6. save model (state_dict approach)
  print("\nSaving trained model state ")
  fn = ".\\Models\\people_model.pt"
  T.save(net.state_dict(), fn)

  # model = Net()  # requires class definition
  # model.load_state_dict(T.load(fn))
  # use model to make prediction(s)

  print("\nEnd People predict politics demo ")

if __name__ == "__main__":
  main()
