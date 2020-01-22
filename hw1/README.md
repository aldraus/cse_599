# CSE 490/599 G1 Homework 1 #

Welcome friends,

For the first assignment we'll be diving right in to Neural Networks: What do they know? Do they know things? Let's find out.
We're going to be implementing a standard "multi-layer perceptron" aka Linear layers plus some non-linearities. 
In this homework you will learn the ins and outs of how backprop actually works i.e. how gradients flow through the network.
You will train a classifier on MNIST, a common image dataset on hand-written numbers. 
In the PyTorch part, you will visualize some of these numbers if you have never seen them before. 
It's actually pretty impressive that we can make a neural net that reads handwriting.

We have provided some tests for checking your implementation. The tests are intentionally missing some cases, but feel free to write more tests yourself.
To run the tests from the outermost directory, simply run
```bash
pytest tests/hw1_tests
```
Or to run them for an individual file (for example test_linear_layer), run
```bash
pytest tests/hw1_tests/test_linear_layer.py
```

## Rules ##
1. You may not use PyTorch or any other deep learning package in parts 1-5 of the homework. Only Numpy and Numba are allowed. Functions like numpy.matmul are fine to use.
1. You may only modify the files we mention (those in the [submit.sh](submit.sh) script). We will not grade files outside of these.
1. You may not change the signatures or return types of `__init__`, `forward`, `backward`, or `step` in the various parts or you will fail our tests. You may add object fields (e.g. `self.data`) or helper functions within the files.
1. Undergrads partners only need to turn in a single homework, but you must put both partner's NetIDs in partners.txt comma separated on a single line.
    Example: `studenta,studentb`
1. You may talk with others about the homework, but you must implement by yourself (except partners).
1. Those not working with a partner should leave partners.txt blank.
1. Feel free to test as you go and modify whatever you want, but we will only grade the files from [submit.sh](submit.sh).

## 1. Layers ##

If you check out [nn/layers/layer.py](nn/layers/layer.py) you will see a lot of complicated python-foo. You can pretty much ignore all of it.
Importantly though, all of your neural network based operations will inherit from `Layer`. This helps create and track the computation graph. 
When you overload `Layer` you will need to implement a `forward` and `backward` function. 
There is also a `parent` which helps create the graph. For this homework you can ignore that field. It has been taken care of by the `SequentialLayer`.

The `forward` function should take an input array and return a new array (not in place). 

The `backward` function will take the partial gradients from the above layer, update the gradients of any parameters affected by the layer, and return the new gradients with respect to the input.
For this homework, each backward you implement should return a single array.
 
You can optionally also implement `initialize` and `selfstr`.

## 2. Parameters ##

These are special data holders for the weights (and biases) of the network. They will help you keep the forward weights and backward weight gradients straight.
Look at [nn/parameter.py](nn/parameter.py) to see what they hold. For forward passes, you will need to access the `param.data` field, and for backward, you will need `param.grad`.
Note that calling `param.grad = ...` actually does the `+=` operation in order to accumulate gradients (this will become more useful in later homeworks).

## 3. Writing your first Layers ##

### 3.1 Linear Layer ###

Open [nn/layers/linear_layer.py](nn/layers/linear_layer.py). Implementing the linear layer should be pretty straightforward. 
Implement both the `forward` and `backward` function. You should not include a nonlinearity here (that will be somewhere else). 
Also take a look at `selfstr`. This includes some extra information that will print when you call `print` on the layer or the network. 
You don't need to change that, but you might want to do similar things in other layers.

To update the gradients of a parameter, just do `param.grad = newval`. No need to change the parameter itself here. That will be done in the optimization step.

You can expect the `LinearLayer` to take as input a 2D array of `(batch X features)` and it should return `(batch X output channels)`

### 3.2 ReLU Layer ###

ReLU is a pretty simple operation, but we will implement it in two different ways. DO NOT implement ReLU in place (you can use in-place operations, but you should not in-place modify the input data itself). 
Since the gradient is undefined at 0 but is 0 for values less than 0, we will define the gradient at 0 to be 0 for simplicity.

ReLU Layers (and all the non-linearities you implement) should accept arrays of arbitrary shape.

### 3.2.1 Numpy ReLU ###
Fill in the code for the ReLU function using standard Numpy operations. Your code should work on matrices of any shape.

### 3.2.2 Numba ReLU ###
Numba is a useful way of writing custom functions over things like matrices. They may not seem necessary for this homework, but you will need them for the next one.
They also may make this homework easier for certain parts.

Fill in the code for the Numba version using standard loops. Your code should again work on matrices of any shape. 
You might notice that the first time the Numba version runs it takes a bit longer (to compile), but after that it should be quite fast. The Numba version may even run faster than the Numpy version, so that's kind of cool.
Since you will likely be looping over the matrix, you may find the various Numpy `flatten` and `reshape` functions useful as a preprocessing step, but be sure the output shape matches the input shape.

## 4. Writing your first Loss Layer and SGD Updates ##
Congratulations, you now have two pieces which you can combine to make a fully functioning neural network. Now we need to make a way to update the weights.

### 4.1 Softmax Cross Entropy Loss Forward ###
Open [nn/layers/losses/softmax_cross_entropy_loss_layer.py](nn/layers/losses/softmax_cross_entropy_loss_layer.py).
Implement the forward pass. To avoid underflow/overflow, you should first subtract the max of each row. 
Because we are using the Softmax function, we can prove that these two inputs should give equivalent results.
```math
\pi_i
= \frac{ \exp(x_i - b + b) }{ \sum_{j=1}^n \exp(x_j - b + b) }
= \frac{ \exp(x_i - b) \exp(b) }{ \sum_{j=1}^n \exp(x_j - b) \exp(b) }
= \frac{ \exp(x_i - b) }{ \sum_{j=1}^n \exp(x_j - b) }
```

By combining the Softmax and the Cross Entropy, we can actually implement a more stable loss as well. First we will implement Log Softmax (n is the size of the label dimension).
```math
\begin{aligned}
\log\left(\frac{e^{x_j}}{\sum_{i=1}^{n} e^{x_i}}\right) &= \log(e^{x_j}) - \log\left(\sum_{i=1}^{n} e^{x_i}\right) \\
&= x_j - \log\left(\sum_{i=1}^{n} e^{x_i}\right)
\end{aligned}
```

Finally, we can implement the Cross Entropy of the Softmax with the label.
```math
H(p,q) = -\sum_{i=1}^n p(i) log(q(x_i)) 
```
Where $`p(i)`$ is the label probability and $`log(q(x_i))`$ is the Log Softmax of the inputs. Since the probabilities are actually input as target integers, the probabilities will be a one-hot encoding of those targets. 
Alternatively, you can use the target integers as indices from the Log Softmax array. Finally, be sure to implement both `mean` and `sum` reduction.

For the first homework, you can expect the input to be 2D (batch x class) and the label to be 1D (batch). However you will get bonus points if you correctly implement it for arbitrary dimensions (warning, harder than it sounds). 
Hint: use numpy moveaxis and reshape to put the class dimension at the end and convert it to (batch x class), then undo after the computations.

### 4.2 Softmax Cross Entropy Loss Backward ###
Since the output of the forward function should be a float, the backward won't take any arguments. Instead, you should use some class variables to store relavent values from the forward pass in order to use them in the backward pass.
With some [fancy math](https://www.ics.uci.edu/~pjsadows/notes.pdf), we can show that the gradient of the loss wrt the logits is actually quite simple.
```math
\frac{\partial L}{\partial x_i} = q(x_i) - p(i)
```
Where $`p(i)`$ is the label probability and $`q(x_i)`$ is the Softmax of the inputs. Remember to scale the loss appropriately if the reduction was mean.
 

### 4.3 SGD Update ###
Open [nn/optimizers/sgd_optimizer.py](nn/optimizers/sgd_optimizer.py).
Recall that each `Parameter` has its own `data` and `grad` variables. Based on the other parts you wrote, the gradients should already be ready inside the `Parameter`. Now we just have to use them to update the weights.

Our normal SGD update with learning rate η is:

```math 
w \Leftarrow w - \eta * \frac{\partial L}{\partial w}
```

With this done, we can now train our first neural network! Open [hw1/main.py](hw1/main.py). We have already provided code to train and test a simple three layer neural network.
Have a look at the pretty standard training loop. First, you get the data. Then you call the forward function on the network to get its outputs. Finally, you zero the previous gradients, call backward on the network, and update the weights.
You can run it by calling
```bash
cd hw1
python main.py
```
After 1 epoch, you should see about 70% test accuracy. After 10 epochs, you should see about 90% accuracy.


## 5. Improvements ##
We can apply numerous improvements over the simple neural network from earlier. After implementing each improvement, you will need to modify [hw1/main.py](hw1/main.py) to use the new network or optimizer.

### 5.1 Momentum SGD ###
Our normal SGD update with learning rate η is:

```math 
w \Leftarrow w - \eta * \frac{\partial L}{\partial w}
```

With weight decay λ we get:
```math 
w \Leftarrow w - \eta * \left( \frac{\partial L}{\partial w} + \lambda w \right)
```

With momentum we first calculate the amount we'll change by as a scalar of our previous change to the weights plus our gradient and weight decay:
    
```math 
\Delta w \Leftarrow m * \Delta w_{prev} +  \left( \frac{\partial L}{\partial w} + \lambda w \right)
```

Then we apply that change:
   ```math 
w \Leftarrow w - \eta \Delta w
``` 

The MomentumSGDOptimizer class will need to keep a history of the previous changes in order to compute the new ones. 

Using momentum should give significantly faster and better convergence. After a single epoch, you should see about 90% accuracy. After 10 epochs, you should see about 95% accuracy.

### 5.2 Leaky ReLU Layer ###
Implement the Leaky ReLU function `LeakyReLU(x) = x if x > 0, slope * x if x <= 0`. We will define the gradient at 0 to be like the negative half. 
(Note, we know we changed this from what it originally said. If you already implemented the version we originally said, you will pass our tests, but this new version is more correct).

You may see LeakyReLU defined in other places as `LeakyReLU(x) = max(x, slope * x)` but what happens if slope is > 1 or negative? Be careful of this trap in your implementation.
You can implement this using either Numpy or Numba, whichever you find easier. 

### 5.3 Parameterized ReLU (PReLU) Layer ###
Implement the PReLU function where the leaky slope is a learned parameter. Again, we will define the gradient at 0 to be like the negative half. 
(Note, we know we changed this from what it originally said. If you already implemented the version we originally said, you will pass our tests, but this new version is more correct).


For more information, see https://arxiv.org/pdf/1502.01852.pdf
Note: The paper also lists a momentum rule which is different from the one we expect you to implement. You should ignore the one in the paper, but also Momentum computations will be taken care of in the optimizer, not in the PReLU layer itself.

PReLU can be either one value per channel (which we will assume is dimension 1 of the input) or one slope for the entire layer.
Thus, the `size` input will be an integer >= 1. 

You can implement this using either Numpy or Numba, whichever you find easier. 
If you implement using Numba, we recommend not using the `parallel=True` flag to ensure that your gradient computations do not overwrite each other. 
However we will give +1 extra credit (and +1 deep learning street cred) if you submit a parallel version.

## 6. PyTorch ##
Navigate to: https://colab.research.google.com/

and upload the iPython notebook provided: `homework1_colab.ipynb`

Complete the notebook to train a PyTorch model on the MNIST dataset.

## 7. Short answer ##
Answer these questions and save them in a file named `hw1/short_answer.pdf`.
1. Play around with different Leaky ReLU slopes. What is the best slope you could find? What happens if you set the slope > 1? What about slope < 0. Theoretically, what happens if you set slope = 1?
2. Set PReLU to take 1 slope per layer. After 20 epochs, what were your PReLU slopes? Does this correspond with what you found in question 1?
3. If you add more layers and more epochs, what accuracy can you reach? Can you get to 99%? What is your best network layout?

## Turn it in ##

First `cd` to the `hw1` directory. Then run the `submit.sh` script by running:

```bash
bash submit.sh
```

This will create the file `submit.tar.gz` in your directory with all the code you need to submit. The command will check to see that your files have changed relative to the version stored in the `git` repository. If it hasn't changed, figure out why, maybe you need to download your ipynb from google?

Submit `submit.tar.gz` in the file upload field for Homework 1 on Canvas.
