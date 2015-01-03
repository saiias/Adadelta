#ifndef ADADELTA_HPP
#define ADADELTA_HPP

#define EIGEN_MPL2_ONLY

#include <Eigen/Dense>
#include <vector>

using namespace Eigen;
using namespace std;
class Adadelta{
private:
  int N;
  int d;
  double C;
  double rho;
  double eps;
  MatrixXd X;
  VectorXd label;
  VectorXd E;
  VectorXd Edx;
  int iteration;
  double sigma(const MatrixXd& _x, int i);
  double sigmoid(double z);
  double rms(double z);


public:
  VectorXd w;
  vector<double> iterscores;
  Adadelta(int _N,int _d,MatrixXd _x,VectorXd _label,double _C,double _rho,double _eps,int iter);
  double Acc(vector<double>& pred,VectorXd &l);

  void train();
  void predict(MatrixXd& _x,VectorXd& _l,vector<double>& ret);
};

#endif
