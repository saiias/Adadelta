#include <iostream>
#include <Eigen/Dense>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include "LR.hpp"
#include "Adadelta.hpp"
#include "Adagrad.hpp"
#include "Adam.hpp"



#define EIGEN_MPL2_ONLY
using namespace Eigen;
using namespace std;

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
  int iteration = 250;
  read("a9a",x,label);
  read("a9a.t",x_test,label_test);
  // LR lr = LR(x.rows(),x.cols(),x,label,0.01,0.1,iteration);
  // lr.train();
  // vector<double> scores;
  // lr.predict(x_test,label_test,scores);
  // cout<<"=== LR ==="<<endl;
  // cout <<"Accuracy:"<< lr.Acc(scores,label_test) <<endl;
  
  // ofstream iofs("iters.txt");
  // for(double d : lr.iterscores){
  //   iofs<<d<<endl;
  // }

  // ofstream wofs("weight.txt");
  // for(int i = 0; i< lr.w.rows();i++){
  //   wofs<<lr.w(i)<<endl;
  // }

  Adam adam = Adam(x.rows(),x.cols(),x,label,0.01,0.002,0.1,0.001,0.000000001,0.00000001,iteration);
  adam.train();
  vector<double> adamscores;
  adam.predict(x_test,label_test,adamscores);
  cout<<"=== Adam ==="<<endl;
  cout <<"Accuracy:"<< adam.Acc(adamscores,label_test) <<endl;

  ofstream adamiofs("adamiters.txt");
  for(double d : adam.iterscores){
    adamiofs<<d<<endl;
  }

  ofstream adamwofs("adamweight.txt");
  for(int i = 0; i< adam.w.rows();i++){
    adamwofs<<adam.w(i)<<endl;
  }
  

  // Adadelta addlr = Adadelta(x.rows(),x.cols(),x,label,0.01,0.95,0.0000001,iteration);
  // addlr.train();
  // vector<double> addscores;
  // addlr.predict(x_test,label_test,addscores);
  // cout<<"=== Adadelta=="<<endl;
  // cout <<"Accuracy:"<< lr.Acc(addscores,label_test) <<endl;
  
  // ofstream addiofs("adadelta_iters.txt");
  // for(double d : addlr.iterscores){
  //   addiofs<<d<<endl;
  // }

  // ofstream addwofs("adadelta_weight.txt");
  // for(int i = 0; i< addlr.w.rows();i++){
  //   addwofs<<lr.w(i)<<endl;
  // }
  
  // Adagrad adglr = Adagrad(x.rows(),x.cols(),x,label,0.01,0.1,iteration);
  // adglr.train();
  // vector<double> adgscores;
  
  // adglr.predict(x_test,label_test,adgscores);
  // cout<<"=== Adagrad ==="<<endl;
  // cout <<"Accuracy:"<< adglr.Acc(adgscores,label_test) <<endl;
  
  // ofstream adgiofs("adagrad_iters.txt");
  // for(double d : adglr.iterscores){
  //   adgiofs<<d<<endl;
  // }

  // ofstream adgwofs("adagrad_weight.txt");
  // for(int i = 0; i< adglr.w.rows();i++){
  //   adgwofs<<adglr.w(i)<<endl;
  // }  
}
