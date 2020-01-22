# CSE 490/599 G1 Homework 2 #

Welcome friends,

For the second assignment we'll be looking at Convolutional Neural Networks (CNNs), the things that make computer vision work. 
In this homework you will implement your own Convolution and Max Pool layers, and eventually will even create a ResNet style layer.
You will again be working on MNIST, but now you should be able to get up to 99% accuracy! 
We're also going to work on CIFAR, and even ImageNet, more challenging datasets on real images.

We have provided some tests for checking your implementation. Feel free to write more. But the tests for Conv and MaxPool should be sufficient for those layers.
To run the tests from the outermost directory, simply run
```bash
pytest tests/hw2_tests
```
Or to run them for an individual file (for example test_conv_layer), run
```bash
pytest tests/hw2_tests/test_conv_layer.py
```

## Rules ##
1. You may not use PyTorch or any other deep learning package in parts 1-5 of the homework. Only Numpy and Numba are allowed. Functions like numpy.matmul are fine to use.
1. You may not use prewritten convolution functions like `scipy.ndimage.convolve` or `cv2.filter2D`. I also think if you did use them, they wouldn't cover all the cases like padding and stride that we need, so it's best just to ignore them.
1. You may only modify the files we mention (those in the [submit.sh](submit.sh) script). We will not grade files outside of these.
1. You may not change the signatures or return types of `__init__`, `forward`, `backward`, or `step` (`forward_numba` and `backward_numba` are ok to modify as you see fit) in the various parts or you will fail our tests. You may add object fields (e.g. `self.data`) or helper functions within the files.
1. Undergrads partners only need to turn in a single homework, but you must put both partner's NetIDs in partners.txt comma separated on a single line.
    Example: `studenta,studentb`
1. You may talk with others about the homework, but you must implement by yourself (except partners).
1. Those not working with a partner should leave partners.txt blank.
1. Feel free to test as you go and modify whatever you want, but we will only grade the files from [submit.sh](submit.sh).

## 1. Convolutions ##

If you check out [nn/layers/conv_layer.py](nn/layers/conv_layer.py) you will see a pretty empty file. You should implement the convolution algorithm as described in class. 
We strongly recommend using Numba for this part of the homework. This is why we had you use Numba in the last homework.
To make everyone's lives easier, we are only asking for you to cover a few cases of convolutions. If you want to do more, you're more than welcome, but as soon as you pass the provided tests, you should be good to go.
Specifically, you can make the following assumptions/simplifications:
1. You will "technically" be implementing the cross-correlation, but everyone calls it convolution.
1. There is one bias per output channel, similar to the `LinearLayer`.
1. The input to the layer will always be 4-Dimensional `(batch, input_channels, height, width)`.
1. The output should be `(batch, output_channels, output_height, output_width)`
1. Convolutional kernels will always be square.
1. Padding is defined as how many rows to add to ALL 4 sides of the input images. e.g. If an input has shape `(2, 3, 100, 100)` and there is padding of 3, the padded input should be of shape `(2, 3, 106, 106)`.
1. You should always use the padding rule `padding = (kernel_size - 1) // 2`, and should always pad with zeros. This rule is nice because if your stride is 1 and the kernel-size is odd, the output will be the same shape as the input.
1. When computing output sizes, you should discard incomplete rows if the stride puts it over the edge. e.g. `(2, 3, 5, 5)` input, 3 kernels of size 2x2 and stride of 2 should result in an output of shape `(2, 3, 2, 2)`.
1. You can expect sane sizes of things and don't have to explicitly error check. e.g. The kernel size will never be larger than the input size or larger than the stride.

A hint on using Numba for this: You can use Numba's `prange` to parallelize certain loops, but you should use the standard python built in `range` for spots which need accumulation (`+=`). 
You can probably use `prange` only in the `forward` pass, but will probably need nested `range` and `prange` in the `backward`.

[This blog post](https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1) has some pretty gifs which may help your understanding.

## 2. Max Pooling ##
This should be very similar to convolution except instead of multiplying, you do a max. The same rules and expectations apply as in the conv layer.

## 3 Flatten Layer ##
We will need this to be able to feed the output of a convolution layer (which outputs `b x c x h x w`)into a linear layer (which expects `b x c`).
This should be very straightforward to implement. All you need to do is reshape the output (have a look at the np.reshape) function.
The `backward` should also be quite straightforward. You can do both `forward` and `backward` in place if you choose.


## 4. Training the network ##
With this done, and using our code from last time, we can now train a full convolutional network! Open [hw2/main.py](hw2/main.py). We have already provided code to train and test.
The network here is actually a very famous network called LeNet. If you would have written this in 1995, you'd be up for a Turing award this year.

Much of the training code will also look familiar from last time.
You can run it by calling
```bash
cd hw2
python main.py
```
After 1 epoch, you should see about 95% test accuracy. After 20 epochs, you should see about 98-99% accuracy.


## 5. Improvements ##
We're now going to implement a version of a ResNet (residual network) which should help our accuracy. You can read more about ResNets [here](https://arxiv.org/pdf/1512.03385.pdf).
This should get us up to 99% accuracy! 

### 5.1 Add Layer ###
The first thing we'll need is a layer to add two (or `N`) tensors together. Up until now, all of the layers have taken a single tensor as input. 
Some of the layers (like `LinearLayer` have parameters, but no layers have had multiple inputs. 

The `forward` pass should take `N` tensors and add them together. You can assume all tensors have exactly the same shape.

The `backward` pass should compute the gradient with respect to each input and return `N` gradient tensors in a tuple. 


### 5.2 ResNetBlock ###
Now to actually make the ResNet block. You can think of this as a mini network which will be used multiple times in your larger network. 
Through the magic of graphs and the backpropagation algorithm, we will only have to write a `forward` for this block and the `backward` can be figured out automatically.
But to get the graph to be happy (AKA to specify the graph properly), we will  have to do some minor bookkeeping.


Firstly, notice that `ResNetBlock` is a subclass of `LayerUsingLayer`. This means all the operations we will do on the data will be using `Layer`s rather than numpy/numba/python functions.
By doing this, we construct a computation graph which can be executed forward and which can be backpropagated through. This is actually something you've seen and used before.
Both `SequentialLayer` and `Network` are also subclasses of `LayerUsingLayer`s. That's why you didn't have to write explicit `backward` functions for them.

Have a look at [nn/layers/sequential_layer.py](nn/layers/sequential_layer.py) as a reference for another subclass of `LayerUsingLayer`.
A `SequentialLayer` applies operations to data sequentially with each `Layer` using the previous `Layer`'s output as its input. 
The nice thing about the sequential layer is we don't have to specify and call all the layers individually.
But there is also some bookkeeping involved. `LayerUsingLayer`s must implement the `final_layer` property to work. We will also need to start using the `parent` argument that we have previously been ignoring.
For `ResNetBlock` we will have to do similar things.

<img src="../readme_assets/images/layer_using_layer.png">

Each graph here is a DAG, and some of them have sub-graphs. Think of a `LayerUsingLayer` as a mini-graph within the overall graph. 
These sub-graphs need to specify certain properties to be able to work well with the full graph.
Specifically, they will need to deal with their input (parent) nodes and their output (final) nodes. 

When we create a node, we will need to specify its `parent`. 
All `Layer`s have an optional final `parent` argument which we will use for this.

Within the logic of a `LayerUsingLayer` we will also need to specify its `final_layer`.

But one nice property is that we can treat sub-graphs in the graph as a single unit. 
This means rather than worry about setting the parent for each individual node in a subgraph, we only need to set the parent of the subgraph itself.
So when we make a `SequentialLayer`, we don't have to set the parent of each sub-node (A,B,C in graph 2), we only have to set the parent of the entire `SequentialLayer` block.


Now let's get into the `ResNetBlock`.
<img src="../readme_assets/images/resnet_block.png">

The `ResNetBlock` has two branches from the input that are eventually combined again. 
The primary path takes the data, applies a Convolution -> ReLU -> Convolution pipeline.
The residual path then adds the original input back to the output of the primary path.
Finally, a ReLU is performed at the end. This is what we need to do in this layer (we are omitting Batch Norm for this assignment.)

So now let's actually implement the thing.



#### 5.2.1 \_\_init\_\_ ####
In `__init__` we need to create the `Layer`s to do those operations listed above. You may find it useful to make the primary path using a `SequentialLayer`.

Make sure to provide the parents of the layers to their constructors (all layers take a final optional `parent` argument in their constructors).

**IMPORTANT** when you assign parents, use `self.parent` instead of `parent`. To see why, look at the `LayerUsingLayer` constructor. It creates a `DummyLayer` placeholder if the actual input is `None`.

#### 5.2.2 forward ####
Now we must actually do the `forward` pass of the data through the ResNet block. It should take in a single data array and return a new data array with the operations applied.
If it seems like you just specified the graph flow in the `__init__` function and now you have to do it again, that's sort of correct and a bit of a design quirk of our library.*


#### 5.2.3 final_layer ####
`final_layer` is pretty straightforward. This is the very last thing done by a `LayerUsingLayer` (in the graph, it is the edge that exits the final node and doesn't go into anything).
In `SequentialLayer` we recursively search for the final layer from the `layers` list in case those layers too are `LayerUsingLayer`s.

Here we simply need to return a reference to the last operation. In this case it should be the ReLU after the add.

### 5.3 Testing it out ###
In [hw2/main.py](hw2/main.py), change out the MNistNetwork for the MNistResNetwork. You should now get up to 99% accuracy on MNIST! Congratulations.



## 6. PyTorch ##
Navigate to: https://colab.research.google.com/

and upload the iPython notebook provided: `homework2_colab.ipynb`

Complete the notebook to train a PyTorch model on CIFAR and ImageNet.

## 7. Short answer ##
Answer these questions and save them in a file named `hw1/short_answer.pdf`.
1. See if you can improve the MNistResNetwork architecture using more ResNetBlocks. What's the highest accuracy you achieve? What is the architecture (you can paste the output from `print(network))`.
2. Do you get any improvement using a different non-linearity? Be sure to change it back to ReLU before you turn in your final code.
3. Can you come up with an architecture which gets even higher accuracy? Again, include the output from `print(network)`.

## Turn it in ##

First `cd` to the `hw2` directory. Then run the `submit.sh` script by running:

```bash
bash submit.sh
```

This will create the file `submit.tar.gz` in your directory with all the code you need to submit. The command will check to see that your files have changed relative to the version stored in the `git` repository. If it hasn't changed, figure out why, maybe you need to download your ipynb from google?

Submit `submit.tar.gz` in the file upload field for Homework 1 on Canvas.


*On why we have this design quirk: We thought it would be nice to work directly with Numpy arrays rather than wrap them in some custom tensor object which could itself keep track of its parents and children and so on. This is how PyTorch works. 
But it also means you must redefine even the most simple functions such as add, multiply, exp, which are already built into Numpy on the Tensor wrapper. This is why in PyTorch you must call `torch.exp` instead of `np.exp` to add a node in the graph.
To give you the flexibility of using any Numpy functions without having to write all these wrapped functions ourselves, we opted for the alternative approach.
