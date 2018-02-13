#!/bin/bash

if [[ -d /wdata/output/predictions/ && $(ls /wdata/output/predictions/) != "" ]]
then
	echo "REMOVING OLD PREDICTION FILES: $(ls /wdata/output/predictions/)"
	rm /wdata/output/predictions/*
fi

if [[ -d /wdata/input/test_data/ && $(ls /wdata/input/test_data/) != "" ]]
then
	echo "REMOVING OLD TEST FILES: $(ls /wdata/input/test_data/)"
	rm -r /wdata/input/test_data/*
fi

echo "PREPROCESSING TEST DATASET"
python iarpa/runBaseline.py --prepare True --path $1


if (( $(ls /wdata/working/cnn_checkpoint_weights/weights.final* |wc -l) == 0 ))
then
	echo "DOWNLOADING MODELS"
	curl -c /tmp/cookies "https://drive.google.com/uc?export=download&id=1iDfrmen7wv6w47DsmdtScwIOW4x9OJDQ" > /tmp/tmp.html
	curl -L -b /tmp/cookies "https://drive.google.com$(cat /tmp/tmp.html | grep -Po 'uc-download-link" [^>]* href="\K[^"]*' | sed 's/\&amp;/\&/g')" > models.tgz
	tar -xzvf models.tgz -C /wdata/working/
	rm models.tgz
fi

echo "GENERATING PREDICTIONS 1/12"
if [ -f /wdata/working/cnn_checkpoint_weights/weights.final.v1.densenet.p01.hdf5 ]
then
	python iarpa/runBaseline.py --test True --num_gpus 4 --algorithm densenet --database v1 --load_weights /wdata/working/cnn_checkpoint_weights/weights.final.v1.densenet.p01.hdf5 --prefix p01
fi

echo "GENERATING PREDICTIONS 2/12"
if [ -f /wdata/working/cnn_checkpoint_weights/weights.final.v1.resnet50.p01.hdf5 ]
then
	python iarpa/runBaseline.py --test True --num_gpus 4 --algorithm resnet50 --database v1 --load_weights /wdata/working/cnn_checkpoint_weights/weights.final.v1.resnet50.p01.hdf5 --prefix p02
fi

echo "GENERATING PREDICTIONS 3/12"
if [ -f /wdata/working/cnn_checkpoint_weights/weights.final.v2.densenet.p02.hdf5 ]
then
	python iarpa/runBaseline.py --test True --num_gpus 4 --algorithm densenet --database v2 --load_weights /wdata/working/cnn_checkpoint_weights/weights.final.v2.densenet.p02.hdf5 --prefix p03
fi

echo "GENERATING PREDICTIONS 4/12"
if [ -f /wdata/working/cnn_checkpoint_weights/weights.final.v3.densenet.p03.hdf5 ]
then
	python iarpa/runBaseline.py --test True --num_gpus 4 --algorithm densenet --database v3 --load_weights /wdata/working/cnn_checkpoint_weights/weights.final.v3.densenet.p03.hdf5 --prefix p04
fi

echo "GENERATING PREDICTIONS 5/12"
if [ -f /wdata/working/cnn_checkpoint_weights/weights.final.v1.densenet.p04.hdf5 ]
then
	python iarpa/runBaseline.py --test True --num_gpus 4 --algorithm densenet --database v1 --load_weights /wdata/working/cnn_checkpoint_weights/weights.final.v1.densenet.p04.hdf5 --prefix p05
fi

echo "GENERATING PREDICTIONS 6/12"
if [ -f /wdata/working/cnn_checkpoint_weights/weights.final.v1.densenet.p05.hdf5 ]
then
	python iarpa/runBaseline.py --test True --num_gpus 4 --algorithm densenet --database v1 --load_weights /wdata/working/cnn_checkpoint_weights/weights.final.v1.densenet.p05.hdf5 --prefix p06
fi

echo "GENERATING PREDICTIONS 7/12"
if [ -f /wdata/working/cnn_checkpoint_weights/weights.final.v3.resnet50.p02.hdf5 ]
then
	python iarpa/runBaseline.py --test True --num_gpus 4 --algorithm resnet50 --database v3 --load_weights /wdata/working/cnn_checkpoint_weights/weights.final.v3.resnet50.p02.hdf5 --prefix p07
fi

echo "GENERATING PREDICTIONS 8/12"
if [ -f /wdata/working/cnn_checkpoint_weights/weights.final.v3.resnet50.p04.hdf5 ]
then
	python iarpa/runBaseline.py --test True --num_gpus 4 --algorithm resnet50 --database v3 --load_weights /wdata/working/cnn_checkpoint_weights/weights.final.v3.resnet50.p04.hdf5 --prefix p08
fi

echo "GENERATING PREDICTIONS 9/12"
if [ -f /wdata/working/cnn_checkpoint_weights/weights.final.v2.resnet50.p03.hdf5 ]
then
	python iarpa/runBaseline.py --test True --num_gpus 4 --algorithm resnet50 --database v2 --load_weights /wdata/working/cnn_checkpoint_weights/weights.final.v2.resnet50.p03.hdf5 --prefix p09
fi

echo "GENERATING PREDICTIONS 10/12"
if [ -f /wdata/working/cnn_checkpoint_weights/weights.final.v3.densenet.p06.hdf5 ]
then
	python iarpa/runBaseline.py --test True --num_gpus 4 --algorithm densenet --database v3 --load_weights /wdata/working/cnn_checkpoint_weights/weights.final.v3.densenet.p06.hdf5 --prefix p10
fi

echo "GENERATING PREDICTIONS 11/12"
if [ -f /wdata/working/cnn_checkpoint_weights/weights.final.v1.densenet.p07.hdf5 ]
then
	python iarpa/runBaseline.py --test True --num_gpus 4 --algorithm densenet --database v1 --load_weights /wdata/working/cnn_checkpoint_weights/weights.final.v1.densenet.p07.hdf5 --prefix p11
fi

echo "GENERATING PREDICTIONS 12/12"
if [ -f /wdata/working/cnn_checkpoint_weights/weights.final.v2.densenet.p08.hdf5 ]
then
	python iarpa/runBaseline.py --test True --num_gpus 4 --algorithm densenet --database v2 --load_weights /wdata/working/cnn_checkpoint_weights/weights.final.v2.densenet.p08.hdf5 --prefix p12
fi

iarpa/fusion $(ls /wdata/output/predictions/*clas* 2> /dev/null) > $2.txt

