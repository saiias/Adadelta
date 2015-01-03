##AdaDelta

Logistic Regression using AdaGrad and AdaDelta

### Dependency

- [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page)

sample.cpp needs [a9a](http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/) datasets

### sample code

```
g++ -I /path/to/eigin/ -std=c++11 -o sample Adagrad.cpp Adadelta.cpp LR.cpp sample.cpp
./sample
```

###reference

- [Adagrad](http://www.ark.cs.cmu.edu/cdyer/adagrad.pdf)
- [Adadelta](http://www.matthewzeiler.com/pubs/googleTR2012/googleTR2012.pdf)


