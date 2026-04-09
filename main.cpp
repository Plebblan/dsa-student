#include <iostream>
#include <iomanip>
#include <sstream>
using namespace std ;

#include "ann/funtions.h"
#include "ann/xtensor_lib.h"
#include "ann/dataset.h"
//#include "ann/dataloader.h"
#include "ann/dataloaderr.h"

int positive_index(int idx, int size){
    if(idx < 0) return idx = size + idx;
    return idx;
}

xt::xarray<double> softmax(xt::xarray<double> X, int axis){
    xt::svector<unsigned long> shape = X.shape();
    axis = positive_index(axis, shape.size());
    shape[axis] = 1;

    xt::xarray<double> Xmax = xt::amax(X, axis);
    X = xt::exp(X - Xmax.reshape(shape));
    xt::xarray<double> SX = xt::sum(X, -1); SX = SX.reshape(shape);
    X = X/SX;

    return X;
}

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
int nsamples = 5;
//xt :: xarray <double > X = xt :: random :: randn < double >({ nsamples , 3}) ;
//xt:: xarray <double> Y= xt :: random :: randn < double >({ nsamples , 3}) ;
    //xt::xarray<double> X={{1,2,3},
                          //{4,5,6}};
    xt::xarray<double> Y={4,5,6};
    //xt::linalg::outer(Y,Y);
    //cout<<xt::linalg::dot(Y,X);
    int m_nAxis=Y.shape().size()-1;
    xt:: xarray <double> max_val = xt::amax(Y, {m_nAxis});
    xt:: xarray <double> stabilized = Y - xt::expand_dims(max_val, m_nAxis);
    xt:: xarray <double> exp_vals = xt::exp(stabilized);
    xt:: xarray <double> sum_exp = xt::sum(exp_vals, {m_nAxis});
    //if (m_trainable) m_aCached_Y=exp_vals / sum_exp.reshape({sum_exp.size(), 1});
    cout<< exp_vals / sum_exp.reshape({sum_exp.size(), 1});
return 0;

}
