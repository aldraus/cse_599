#!/usr/bin/env bash
rm -rf submit/
mkdir -p submit

prepare () {
    cp $1 submit/
}

echo "Creating tarball..."
prepare ../nn/layers/losses/softmax_cross_entropy_loss_layer.py
prepare ../nn/layers/leaky_relu_layer.py
prepare ../nn/layers/linear_layer.py
prepare ../nn/layers/prelu_layer.py
prepare ../nn/layers/relu_layer.py
prepare ../nn/optimizers/sgd_optimizer.py
prepare ../nn/optimizers/momentum_sgd_optimizer.py
prepare homework1_colab.ipynb
prepare short_answer.pdf
prepare partners.txt

tar cvzf submit.tar.gz submit
rm -rf submit/
echo "Done. Please upload submit.tar.gz to Canvas."
