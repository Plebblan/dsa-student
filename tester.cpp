#include <iostream>
#include <iomanip>
#include <sstream>
using namespace std ;

#include "ann/funtions.h"
#include "ann/xtensor_lib.h"
#include "ann/dataset.h"
//#include "ann/dataloader.h"
#include "ann/dataloaderr.h"

string shape2str(xt::svector<unsigned long>vec){
    stringstream ss;
    ss << "(";
    for(int idx=0; idx < vec.size(); idx++){
        ss << vec[idx] << ", ";
    }
    string res = ss.str();
    if(vec.size() > 1) res = res.substr(0, res.rfind(','));
    else res = res.substr(0, res.rfind(' '));
    return res + ")";
}

int main ( int argc , char ** argv ) {
int nsamples = 10;
xt :: xarray <double > X = xt :: random :: randn < double >({ nsamples , 10}) ;
xt :: xarray <double > T = xt :: random :: randn < double >({ nsamples , 5}) ;
TensorDataset < double , double > ds (X , T ) ;

DataLoader < double , double > loader (& ds , 3, true , false ) ;
/*for (auto batch: loader) {
cout<< shape2str(batch.getData().shape())<< endl ;
cout<< shape2str(batch.getLabel().shape())<< endl ;
}*/ cout<<1;
return 0;

}

