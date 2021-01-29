#include <iostream>
#include <Eigen/Dense>

#include <Eigen/src/Core/util/DisableStupidWarnings.h>

// tolerance for chekcing number of iterations
#define LM_EVAL_COUNT_TOL 4/3

#include <unsupported/Eigen/NonLinearOptimization>

using std::sqrt;
using namespace Eigen; // ::DenseFunctor

// Generic functor
template<typename _Scalar, int NX = Eigen::Dynamic, int NY = Eigen::Dynamic>
struct Functor
{
typedef _Scalar Scalar;
enum {
    InputsAtCompileTime = NX,
    ValuesAtCompileTime = NY
};
typedef Eigen::Matrix<Scalar,InputsAtCompileTime,1> InputType;
typedef Eigen::Matrix<Scalar,ValuesAtCompileTime,1> ValueType;
typedef Eigen::Matrix<Scalar,ValuesAtCompileTime,InputsAtCompileTime> JacobianType;

int m_inputs, m_values;

Functor() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}
Functor(int inputs, int values) : m_inputs(inputs), m_values(values) {}

int inputs() const { return m_inputs; }
int values() const { return m_values; }

};

struct misra1d_functor : Functor<double>
{
    misra1d_functor(void) : Functor<double>(2,14) {}
    static const double x[14];
    static const double y[14];
    int operator()(const VectorXd &b, VectorXd &fvec)
    {
        assert(b.size() == 2);
        assert(fvec.size() == 14);
        for(int i=0; i<14; i++) {
            fvec[i] = b[0]*b[1]*x[i]; // /(1.+b[1]*x[i]) - y[i];
        }
        return 0;
    }
    int df(const VectorXd &b, MatrixXd &fjac)
    {
        assert(b.size()==2);
        assert(fjac.rows()==14);
        assert(fjac.cols()==2);
        for(int i=0; i<14; i++) {
            // double den = 1.+b[1]*x[i];
            fjac(i,0) = b[1]*x[i]; // / den;
            fjac(i,1) = b[0]*x[i]; // *(den-b[1]*x[i])/den/den;
        }
        return 0;
    }
};
const double misra1d_functor::x[14] = { 77.6E0, 114.9E0, 141.1E0, 190.8E0, 239.9E0, 289.0E0, 332.8E0, 378.4E0, 434.8E0, 477.3E0, 536.8E0, 593.1E0, 689.1E0, 760.0E0};
const double misra1d_functor::y[14] = { 10.07E0, 14.73E0, 17.94E0, 23.93E0, 29.61E0, 35.18E0, 40.02E0, 44.82E0, 50.76E0, 55.05E0, 61.01E0, 66.40E0, 75.47E0, 81.78E0};

void testNistMisra1d(void)
{
  const int n=2;
  int info;

  VectorXd x(n);

  /*
   * First try
   */
  x << 500.0, 2.0;
  // do the computation
  misra1d_functor functor;
  LevenbergMarquardt<misra1d_functor> lm(functor);
  info = lm.minimize(x);
  std::cout << info << std::endl;
  std::cout << x << std::endl;

  // check return value
  //VERIFY_IS_EQUAL(info, 1);
  //VERIFY_IS_EQUAL(lm.nfev(), 9);
  //VERIFY_IS_EQUAL(lm.njev(), 7);
  // check norm^2
  //VERIFY_IS_APPROX(lm.fvec().squaredNorm(), 5.6419295283E-02);
  // check x
  //VERIFY_IS_APPROX(x[0], 4.3736970754E+02);
  //VERIFY_IS_APPROX(x[1], 3.0227324449E-04);

  /*
   * Second try
   */

  x << 450.0, 3.0;

  // do the computation
  info = lm.minimize(x);
  std::cout << info << std::endl;
  std::cout << x << std::endl;
  VectorXd y(14);
  functor(x, y);
  std::cout << y << std::endl;

  // check return value
  //VERIFY_IS_EQUAL(info, 1);
  //VERIFY_IS_EQUAL(lm.nfev(), 4);
  //VERIFY_IS_EQUAL(lm.njev(), 3);
  // check norm^2
  //VERIFY_IS_APPROX(lm.fvec().squaredNorm(), 5.6419295283E-02);
  // check x
  //VERIFY_IS_APPROX(x[0], 4.3736970754E+02);
  //VERIFY_IS_APPROX(x[1], 3.0227324449E-04);
}

int main(int argc, char *argv[]){
    testNistMisra1d();
    // fazer exemplo artificial com o sophus
}
