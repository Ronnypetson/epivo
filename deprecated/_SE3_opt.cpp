#include <iostream>
#include <Eigen/Dense>
 
using namespace Eigen;
using namespace std;
 
int main()
{
  MatrixXd m(2,2);
  m(0,0) = 3;
  m(1,0) = 2.5;
  m(0,1) = -1;
  m(1,1) = m(1,0) + m(0,1);
  cout << "Here is the matrix m:\n" << m << endl;
  VectorXd v(2);
  v(0) = 4;
  v(1) = v(0) - 1;
  cout << "Here is the vector v:\n" << v << endl;
}
