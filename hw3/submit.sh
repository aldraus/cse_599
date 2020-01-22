#!/usr/bin/env bash
cd hw1
rm -rf submit/
mkdir -p submit

prepare () {
    cp $1 submit/
}

echo "Creating tarball..."
prepare homework3_colab.ipynb
prepare short_answer.pdf
prepare partners.txt

tar cvzf submit.tar.gz submit
rm -rf submit/
cd ..
echo "Done. Please upload submit.tar.gz to Canvas."
