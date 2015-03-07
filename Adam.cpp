#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include "Adam.hpp"

#define EIGEN_MPL2_ONLY
using namespace Eigen;
using namespace std;

Adam::Adam(int _N,int _d,MatrixXd _x,VectorXd _label,double _C,double _alpha,double _beta1,double _beta2,double _epsilon,double _lambda,int iter):
  N(_N),
  d(_d),
  alpha(_alpha),
  beta1(_beta1),
  beta2(_beta2),
  epsilon(_epsilon),
  lambda(_lambda),
  C(_C),
  X(_x),
  m(VectorXd::Zero(d)),
  v(VectorXd::Zero(d)),
  label(_label),
  iteration(iter),
  w(VectorXd::Zero(d))
{}

double Adam::Acc(vector<double>& pred,VectorXd &l){
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

void Adam::train(){
  for(int iter = 0; iter< iteration; iter++){
    for(int i = 0; i < N; i++){
      double pred = sigma(X,i);
      for(int idx = 0; idx < d; idx++){
        if(X(i,idx) != 0){
          double tbeta = 1-(1-beta1) * pow(lambda,iter);
          double grad = (pred - label(i)) * X(i,idx)+ C * w(idx);
          m[idx] = tbeta * grad + (1-tbeta) * m[idx];
          v[idx] = beta2*pow(grad,2) + (1-beta2) * v[idx];
          double hat_m = m[idx]/(1-pow((1-beta1),iter+1));
          double hat_v = v[idx]/(1-pow((1-beta2),iter+1));
          w(idx) -= alpha*hat_m/(sqrt(hat_v) + epsilon);
        }
      }
    }
    vector<double> ret;
    predict(X,label,ret);
    iterscores.push_back(Acc(ret,label));
  }
}

double Adam::sigma(const MatrixXd& _x, int i){
  return sigmoid(_x.row(i).dot(w));
}

double Adam::sigmoid(double z){
  return 1.0 / (1.0 + exp(-z));
}

void Adam::predict(MatrixXd& _x,VectorXd& _l,vector<double>& ret){
  for(int i = 0; i < _x.rows(); i++){
    ret.push_back(sigma(_x,i));
  }
}

