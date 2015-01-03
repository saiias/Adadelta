#ifndef LR_HPP__
#define LR_HPP__

#define EIGEN_MPL2_ONLY

#include <Eigen/Dense>
#include <vector>

using namespace Eigen;
using namespace std;

class LR{
private:
  int N;
  int d;
  MatrixXd X;
  VectorXd label;  
  double C;
  double eta;
  int iteration;  
  double sigma(const MatrixXd& _x, int i);
  double sigmoid(double z);

public:
  VectorXd w;
  vector<double> iterscores;
  LR(int _N,int _d,MatrixXd _x,VectorXd _label,double _C,double _eta,int iter);
  double Acc(vector<double>& pred,VectorXd &l);
  void train();
  void predict(MatrixXd& _x,VectorXd& _l,vector<double>& ret);
};

#endif
