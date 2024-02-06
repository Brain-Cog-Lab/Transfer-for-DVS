python main_transfer.py --model Transfer_VGG_SNN --node-type LIFNode --source-dataset CALTECH101 --target-dataset NCALTECH101 --step 10 --batch-size 120 --act-fun QGateGrad --device 4 --seed 42 --num-classes 101 --traindata-ratio 1.0 --smoothing 0.0 --domain-loss --domain-loss-coefficient 0.0 --TET-loss&
PID1=$!; 
python main_transfer.py --model Transfer_VGG_SNN --node-type LIFNode --source-dataset CALTECH101 --target-dataset NCALTECH101 --step 10 --batch-size 120 --act-fun QGateGrad --device 6 --seed 42 --num-classes 101 --traindata-ratio 1.0 --smoothing 0.0 --domain-loss --domain-loss-coefficient 0.5 --TET-loss&
PID2=$!; 
wait ${PID1} && wait ${PID2}

python main_transfer.py --model Transfer_VGG_SNN --node-type LIFNode --source-dataset CALTECH101 --target-dataset NCALTECH101 --step 10 --batch-size 120 --act-fun QGateGrad --device 4 --seed 1024 --num-classes 101 --traindata-ratio 1.0 --smoothing 0.0 --domain-loss --domain-loss-coefficient 0.0 --TET-loss&
PID1=$!; 
python main_transfer.py --model Transfer_VGG_SNN --node-type LIFNode --source-dataset CALTECH101 --target-dataset NCALTECH101 --step 10 --batch-size 120 --act-fun QGateGrad --device 6 --seed 1024 --num-classes 101 --traindata-ratio 1.0 --smoothing 0.0 --domain-loss --domain-loss-coefficient 0.5 --TET-loss&
PID2=$!; 
wait ${PID1} && wait ${PID2}

python main_transfer.py --model Transfer_VGG_SNN --node-type LIFNode --source-dataset CALTECH101 --target-dataset NCALTECH101 --step 10 --batch-size 120 --act-fun QGateGrad --device 4 --seed 90210 --num-classes 101 --traindata-ratio 1.0 --smoothing 0.0 --domain-loss --domain-loss-coefficient 0.0 --TET-loss&
PID1=$!; 
python main_transfer.py --model Transfer_VGG_SNN --node-type LIFNode --source-dataset CALTECH101 --target-dataset NCALTECH101 --step 10 --batch-size 120 --act-fun QGateGrad --device 6 --seed 90210 --num-classes 101 --traindata-ratio 1.0 --smoothing 0.0 --domain-loss --domain-loss-coefficient 0.5 --TET-loss&
PID2=$!; 
wait ${PID1} && wait ${PID2}



python main_transfer.py --model Transfer_ResNet18 --node-type LIFNode --source-dataset RGBCEPDVS --target-dataset CEPDVS --step 6 --batch-size 120 --act-fun QGateGrad --device 4 --seed 42 --num-classes 20 --domain-loss --domain-loss-coefficient 0.0 --TET-loss --smoothing 0.0 --event-size 48 --train-portion 0.5 --DVS-DA --epochs 200&
PID1=$!; 
python main_transfer.py --model Transfer_ResNet18 --node-type LIFNode --source-dataset RGBCEPDVS --target-dataset CEPDVS --step 6 --batch-size 120 --act-fun QGateGrad --device 6 --seed 42 --num-classes 20 --domain-loss --domain-loss-coefficient 0.5 --TET-loss --smoothing 0.0 --event-size 48 --train-portion 0.5 --DVS-DA --epochs 200&
PID2=$!; 
wait ${PID1} && wait ${PID2}


python main_transfer.py --model Transfer_ResNet18 --node-type LIFNode --source-dataset RGBCEPDVS --target-dataset CEPDVS --step 6 --batch-size 120 --act-fun QGateGrad --device 4 --seed 1024 --num-classes 20 --domain-loss --domain-loss-coefficient 0.0 --TET-loss --smoothing 0.0 --event-size 48 --train-portion 0.5 --DVS-DA --epochs 200&
PID1=$!; 
python main_transfer.py --model Transfer_ResNet18 --node-type LIFNode --source-dataset RGBCEPDVS --target-dataset CEPDVS --step 6 --batch-size 120 --act-fun QGateGrad --device 6 --seed 1024 --num-classes 20 --domain-loss --domain-loss-coefficient 0.5 --TET-loss --smoothing 0.0 --event-size 48 --train-portion 0.5 --DVS-DA --epochs 200&
PID2=$!; 
wait ${PID1} && wait ${PID2}


python main_transfer.py --model Transfer_ResNet18 --node-type LIFNode --source-dataset RGBCEPDVS --target-dataset CEPDVS --step 6 --batch-size 120 --act-fun QGateGrad --device 4 --seed 90210 --num-classes 20 --domain-loss --domain-loss-coefficient 0.0 --TET-loss --smoothing 0.0 --event-size 48 --train-portion 0.5 --DVS-DA --epochs 200&
PID1=$!; 
python main_transfer.py --model Transfer_ResNet18 --node-type LIFNode --source-dataset RGBCEPDVS --target-dataset CEPDVS --step 6 --batch-size 120 --act-fun QGateGrad --device 6 --seed 90210 --num-classes 20 --domain-loss --domain-loss-coefficient 0.5 --TET-loss --smoothing 0.0 --event-size 48 --train-portion 0.5 --DVS-DA --epochs 200&
PID2=$!; 
wait ${PID1} && wait ${PID2}


python main_transfer.py --model Transfer_VGG_SNN --node-type LIFNode --source-dataset CALTECH101 --target-dataset NCALTECH101 --step 10 --batch-size 120 --act-fun QGateGrad --device 4 --seed 42 --num-classes 101 --traindata-ratio 1.0 --smoothing 0.0 --domain-loss --domain-loss-coefficient 0.5 --TET-loss --no-use-hsv --no-sliding-training&
PID1=$!; 
python main_transfer.py --model Transfer_VGG_SNN --node-type LIFNode --source-dataset CALTECH101 --target-dataset NCALTECH101 --step 10 --batch-size 120 --act-fun QGateGrad --device 6 --seed 42 --num-classes 101 --traindata-ratio 1.0 --smoothing 0.0 --domain-loss --domain-loss-coefficient 0.5 --TET-loss --no-use-hsv&
PID2=$!; 
wait ${PID1} && wait ${PID2}


python main_transfer.py --model Transfer_VGG_SNN --node-type LIFNode --source-dataset CALTECH101 --target-dataset NCALTECH101 --step 10 --batch-size 120 --act-fun QGateGrad --device 4 --seed 42 --num-classes 101 --traindata-ratio 1.0 --smoothing 0.0 --domain-loss --domain-loss-coefficient 0.5 --TET-loss --no-use-hsv --regularization&
PID1=$!;
python main_transfer.py --model Transfer_VGG_SNN --node-type LIFNode --source-dataset CALTECH101 --target-dataset NCALTECH101 --step 10 --batch-size 120 --act-fun QGateGrad --device 6 --seed 42 --num-classes 101 --traindata-ratio 1.0 --smoothing 0.0 --domain-loss --domain-loss-coefficient 0.5 --TET-loss --no-sliding-training --regularization&
PID2=$!; 
wait ${PID1} && wait ${PID2}


python main_transfer.py --model Transfer_ResNet18 --node-type LIFNode --source-dataset RGBCEPDVS --target-dataset CEPDVS --step 6 --batch-size 120 --act-fun QGateGrad --device 4 --seed 42 --num-classes 20 --domain-loss --domain-loss-coefficient 0.5 --TET-loss --smoothing 0.0 --event-size 48 --train-portion 0.5 --DVS-DA --epochs 200 --no-use-hsv --regularization&
PID1=$!; 
python main_transfer.py --model Transfer_ResNet18 --node-type LIFNode --source-dataset RGBCEPDVS --target-dataset CEPDVS --step 6 --batch-size 120 --act-fun QGateGrad --device 6 --seed 42 --num-classes 20 --domain-loss --domain-loss-coefficient 0.5 --TET-loss --smoothing 0.0 --event-size 48 --train-portion 0.5 --DVS-DA --epochs 200 --no-sliding-training --regularization&
PID2=$!; 
wait ${PID1} && wait ${PID2}