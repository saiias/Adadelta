#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include "Adadelta.hpp"

#define EIGEN_MPL2_ONLY
using namespace Eigen;
using namespace std;

Adadelta::Adadelta(int _N,int _d,MatrixXd _x,VectorXd _label,double _C,double _rho,double _eps,int iter):
  N(_N),
  d(_d),
  C(_C),
  rho(_rho),
  eps(_eps),
  X(_x),
  label(_label),
  E(VectorXd::Zero(d)),
  Edx(VectorXd::Zero(d)),
  iteration(iter),
  w(VectorXd::Random(d))
{}

double Adadelta::Acc(vector<double>& pred,VectorXd &l){
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

double Adadelta::rms(double z){
  return sqrt(z + eps);
}

void Adadelta::train(){
  for(int iter = 0; iter< iteration; iter++){
    for(int i = 0; i < N; i++){
      double pred = sigma(X,i);
      for(int idx = 0; idx < d; idx++){
        if(X(i,idx) != 0){
          double grad = (pred - label(i)) * X(i,idx) + C * w(idx);
          E[idx] = rho * E[idx] + (1.0 - rho) * grad *grad;
          double deltax = -(rms(Edx(idx))/rms(E(idx))) * grad;
          Edx[idx] = rho * Edx[idx] + (1.0 - rho) * deltax * deltax;
          w(idx) += deltax;
        }
      }
    }
    vector<double> ret;
    predict(X,label,ret);
    iterscores.push_back(Acc(ret,label));
  }
}

double Adadelta::sigma(const MatrixXd& _x, int i){
  return sigmoid(_x.row(i).dot(w));
}

double Adadelta::sigmoid(double z){
  return 1.0 / (1.0 + exp(-z));
}

void Adadelta::predict(MatrixXd& _x,VectorXd& _l,vector<double>& ret){
  for(int i = 0; i < _x.rows(); i++){
    ret.push_back(sigma(_x,i));
  }
}

