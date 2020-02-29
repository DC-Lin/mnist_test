# mnist_test
mnist手写字体做auto-encoding，百次最终MSE
![mseloss](https://github.com/DC-Lin/mnist_test/blob/master/MSElossOfae.png)
对encoding的输出提取显示
![encodingfeature](https://github.com/DC-Lin/mnist_test/blob/master/vae.png)
左边为原图片，右边为decoding后的图片
![OPvsDP](https://github.com/DC-Lin/mnist_test/blob/master/vae1.png)
最后手写一个numpy的神经网络进行mnist字体的识别，在学习率1e-3下，最终的损失如图
![SGD](https://github.com/DC-Lin/mnist_test/blob/master/numpy_mnist_loss.png)
