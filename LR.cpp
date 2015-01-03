#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include "LR.hpp"

#define EIGEN_MPL2_ONLY
using namespace Eigen;
using namespace std;

double LR::sigma(const MatrixXd& _x, int i){
  return sigmoid(_x.row(i).dot(w));
}

double LR::sigmoid(double z){
  return 1.0 / (1.0 + exp(-z));
}
  

LR::LR(int _N,int _d,MatrixXd _x,VectorXd _label,double _C,double _eta,int iter):
  N(_N),
  d(_d),
  X(_x),
  label(_label),
  C(_C),
  eta(_eta),
  iteration(iter),
  w(VectorXd::Random(d))
{}

double LR::Acc(vector<double>& pred,VectorXd &l){
  int t =0;
  double loss =0;
  for(int i = 0; i< pred.size();i++){
    loss += (l(i) - pred[i]) * (l(i) - pred[i]);
    int s = pred[i] > 0.5 ? 1 : 0;
    if(s == l(i)){
      t++;
    }
  }
  return (double)t/pred.size();
}

void LR::train(){
  for(int iter = 0; iter< iteration; iter++){
    for(int i = 0; i < N; i++){
      double pred = sigma(X,i);
      for(int idx = 0; idx < d; idx++){
        if(X(i,idx) != 0){
          w(idx) -= eta*((pred - label(i)) * X(i,idx) + C * w(idx));
        }
      }
    }
    vector<double> ret;
    predict(X,label,ret);
    iterscores.push_back(Acc(ret,label));
    eta *= 0.9;
  }
}
void LR::predict(MatrixXd& _x,VectorXd& _l,vector<double>& ret){
  for(int i = 0; i < _x.rows(); i++){
    ret.push_back(sigma(_x,i));
  }
}

