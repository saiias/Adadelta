#ifndef ADAM_HPP
#define ADAM_HPP

#define EIGEN_MPL2_ONLY

#include<Eigen/Dense>
#include<vector>

using namespace Eigen;
using namespace std;

class Adam{
private:
  int N;
  int d;
  double alpha;
  double beta1;
  double beta2;
  double epsilon;
  double lambda;
  double C;
  MatrixXd X;
  VectorXd m;
  VectorXd v;
  VectorXd label;
  int iteration;
  double sigma(const MatrixXd& _x, int i);
  double sigmoid(double z);
  double rms(double z);


public:
  VectorXd w;
  vector<double> iterscores;
  Adam(int _N,int _d,MatrixXd _x,VectorXd _label,double _C,double _alpha,double _beta1,double _beta2,double _epsilon,double _lambda,int iter);
  double Acc(vector<double>& pred,VectorXd &l);
  void train();
  void predict(MatrixXd& _x,VectorXd& _l,vector<double>& ret);
};  

#endif
