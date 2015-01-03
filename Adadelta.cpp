#include<iostream>
#include<Eigen/Dense>
#include<cmath>
#include<vector>
#include<sstream>
#include<fstream>
#include<iostream>
#include<string>

#define DEBUG(x) cout<<"line"<<__LINE__<<":"<<#x" == "<<x<<endl
#define EIGEN_MPL2_ONLY
using namespace Eigen;
using namespace std;

class Adadelta{
public:
  double C;
  double rho;
  double eps;
  MatrixXd X;
  VectorXd label;
  VectorXd w;
  VectorXd E;
  VectorXd Edx;
  int iteration;
  int N;
  int d;
  vector<double> iterscores;
  Adadelta(int _N,int _d,MatrixXd _x,VectorXd _label,double _C,double _rho,double _eps,int iter){
    N =_N;
    d = _d;
    X = _x;
    label = _label;
    C = _C;
    rho = _rho;
    eps = _eps;
    w = VectorXd::Random(d);
    E = VectorXd::Zero(d);
    Edx = VectorXd::Zero(d);
    iteration = iter;
  }

  double Acc(vector<double>& pred,VectorXd &l){
    int t =0;
    double loss =0;
    for(int i = 0; i< pred.size();i++){
      loss += (l(i) - pred[i]) * (l(i) - pred[i]);
      int s = pred[i] > 0.5 ? 1 : 0;
      if(s == l(i)){
        t++;
      }
    }
    cout<<"Accuracy: "<<(double)t/pred.size()<<endl;
    // cout<<"Error: "<<loss<<endl;
    return (double)t/pred.size();
  }
  double rms(double z){
    return sqrt(z + eps);
  }

  void train(){
    for(int iter = 0; iter< iteration; iter++){
      cout<<"Iteration:"<<iter<<endl;
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

  double sigma(const MatrixXd& _x, int i){
    return sigmoid(_x.row(i).dot(w));
  }

  double sigmoid(double z){
    return 1.0 / (1.0 + exp(-z));
  }

  void predict(MatrixXd& _x,VectorXd& _l,vector<double>& ret){
    for(int i = 0; i < _x.rows(); i++){
      ret.push_back(sigma(_x,i));
    }
  }
};

void read(string filepath, MatrixXd& x,VectorXd& l){
  ifstream ifs(filepath);
  if(ifs.fail()){
    cerr << "File do not exist."<<endl;
    exit(0);
  }
  string s;
  int index = 0;
  while(getline(ifs,s)){
    stringstream ss(s);
    int t;
    ss>>t;
    if(t>0){
      l(index) = 1;
    }else{
      l(index) = 0;
    }
    string feature;
    while(ss>>feature){
      string::size_type idx = feature.find(":");
      int n = atoi(feature.substr(0,idx).c_str());
      double value = atof(feature.substr(idx+1).c_str());
      x(index,n-1) = value;
    }
    ++index;
  }
}

int main()
{
  MatrixXd x = MatrixXd::Zero(32561,123);
  VectorXd label(32561);
  MatrixXd x_test = MatrixXd::Zero(16281,123);
  VectorXd label_test(16281);
  read("a9a",x,label);
  read("a9a.t",x_test,label_test);

  Adadelta lr = Adadelta(x.rows(),x.cols(),x,label,0.01,0.95,0.0000001,300);
  lr.train();
  vector<double> scores;
  lr.predict(x_test,label_test,scores);
  cout <<"Accuracy:"<< lr.Acc(scores,label_test) <<endl;
  
  ofstream iofs("adadelta_iters.txt");
  for(double d : lr.iterscores){
    iofs<<d<<endl;
  }

  ofstream wofs("adadelta_weight.txt");
  for(int i = 0; i< lr.w.rows();i++){
    wofs<<lr.w(i)<<endl;
  }
}
