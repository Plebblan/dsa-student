#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
using namespace std;

//#include "sformat/fmt_lib.h"
#include "xtensor_lib.h"
void manual_vector(){
  xt::xarray<double> x = {1, 3, 5, 7};
  cout << "Simple tensor(vector):" << endl;
  cout << x << endl;
}
void manual_matrix(){
  xt::xarray<double> X = {
    {1, 2, 3},
    {4, 5, 6},
    {7, 8, 9}
  };
  cout << "Simple tensor(matrix):" << endl;
  cout << X << endl;
}

void gen_vector(){
  xt::xarray<double> x = xt::arange(10);
  cout << "Simple tensor (a vector):" << endl;
  cout << x << endl;
}
void gen_matrix(){
  xt::xarray<double> X = xt::random::randn<double>({3, 4});
  cout << "Generated tensor(matrix):" << endl;
  cout << X << endl;
}

int main(int argc, char** argv) {
    manual_vector();
    manual_matrix();
    gen_vector();
    gen_matrix();


    return 0;
}
