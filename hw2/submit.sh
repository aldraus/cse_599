#!/usr/bin/env bash
rm -rf submit/
mkdir -p submit

prepare () {
    cp $1 submit/
}

echo "Creating tarball..."
prepare ../nn/layers/block_layers/res_net_block.py
prepare ../nn/layers/add_layer.py
prepare ../nn/layers/conv_layer.py
prepare ../nn/layers/flatten_layer.py
prepare ../nn/layers/max_pool_layer.py
prepare homework2_colab.ipynb
prepare homework2_colab.pdf
prepare weights.pt
prepare short_answer.pdf
prepare partners.txt

tar cvzf submit.tar.gz submit
rm -rf submit/
echo "Done. Please upload submit.tar.gz to Canvas."
