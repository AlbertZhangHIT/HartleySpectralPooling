# for cifar
#CUDA_VISIBLE_DEVICES=0 python cifar_resnet.py --epoch 160 --batch-size 128 --lr 0.1 --momentum 0.9 --wd 1e-4 -ct 10

#CUDA_VISIBLE_DEVICES=0 python cifar_resnet_spectral.py --epoch 160 --batch-size 128 --lr 0.1 --momentum 0.9 --wd 1e-4 -ct 10

# for mnist
#CUDA_VISIBLE_DEVICES=0 python mnist_resnet.py --epoch 15 --batch-size 100 --lr 0.001

#CUDA_VISIBLE_DEVICES=0 python mnist_resnet_spectral.py --epoch 15 --batch-size 100 --lr 0.001

#CUDA_VISIBLE_DEVICES=0 python mnist_simplenet.py --epoch 15 --batch-size 100 --lr 0.001

#CUDA_VISIBLE_DEVICES=0 python mnist_simplenet_spectral.py --epoch 15 --batch-size 100 --lr 0.001