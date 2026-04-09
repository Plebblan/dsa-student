/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cppFiles/file.h to edit this template
 */

/*
 * File:   dataset.h
 * Author: ltsach
 *
 * Created on September 2, 2024, 3:59 PM
 */

#ifndef DATASET_H
#define DATASET_H
#include "ann/xtensor_lib.h"
using namespace std;

template<typename DType, typename LType>
class DataLabel{
private:
    xt::xarray<DType> data;
    xt::xarray<LType> label;
public:
    DataLabel(xt::xarray<DType> data,  xt::xarray<LType> label):
    data(data), label(label){
    }
    xt::xarray<DType> getData() const{ return data; }
    xt::xarray<LType> getLabel() const{ return label; }
};

template<typename DType, typename LType>
class Batch{
private:
    xt::xarray<DType> data;
    xt::xarray<LType> label;
public:
    Batch(xt::xarray<DType> data,  xt::xarray<LType> label):
    data(data), label(label){
    }
    virtual ~Batch(){}
    xt::xarray<DType>& getData(){return data; }
    xt::xarray<LType>& getLabel(){return label; }
};


template<typename DType, typename LType>
class Dataset{
private:
public:
    Dataset(){};
    virtual ~Dataset(){};

    virtual int len()=0;
    virtual DataLabel<DType, LType> getitem(int index)=0;
    virtual xt::svector<unsigned long> get_data_shape()=0;
    virtual xt::svector<unsigned long> get_label_shape()=0;
    virtual int get_data_size() =0;
    virtual int get_data_dimension() =0;
    virtual int get_label_size() =0;
    virtual int get_label_dimension() =0;
};

//////////////////////////////////////////////////////////////////////
template<typename DType, typename LType>
class TensorDataset: public Dataset<DType, LType>{
private:
    xt::xarray<DType> data;
    xt::xarray<LType> label;
    xt::svector<unsigned long> data_shape, label_shape;

public:
    /* TensorDataset:
     * need to initialize:
     * 1. data, label;
     * 2. data_shape, label_shape
    */
    TensorDataset(xt::xarray<DType> data, xt::xarray<LType> label) : data(data), label(label){
        data_shape=data.shape();
        label_shape=label.shape();
    }
    /* len():
     *  return the size of dimension 0
    */
    int len(){
        return data.shape(0);
    }

    /* getitem:
     * return the data item (of type: DataLabel) that is specified by index
     */
    DataLabel<DType, LType> getitem(int index){
        if (label.dimension()==0)
        {
            DataLabel<DType,LType> dataItem(xt::view(data,index),label);
            return dataItem;
        }
        DataLabel<DType, LType> dataItem(xt::view(data,index),xt::view(label,index));
        return dataItem;
    }

    xt::svector<unsigned long> get_data_shape(){
        return data_shape;
    }
    xt::svector<unsigned long> get_label_shape(){
        return label_shape;
    }

    int get_data_size() {
        return data.size();
    }
    int get_data_dimension() {
        return data.dimension();
    }
    int get_label_size() {
        return label.size();
    }
    int get_label_dimension() {
        return label.dimension();
    }
};



#endif /* DATASET_H */

