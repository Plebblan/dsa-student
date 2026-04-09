/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cppFiles/file.h to edit this template
 */

/*
 * File:   dataloader.h
 * Author: ltsach
 *
 * Created on September 2, 2024, 4:01 PM
 */

#ifndef DATALOADER_H
#define DATALOADER_H
#include "ann/xtensor_lib.h"
#include "ann/dataset.h"
//#include "list/listheader.h"

using namespace std;

template<typename DType, typename LType>
class DataLoader{
public:

private:
    Dataset<DType, LType>* ptr_dataset;
    int batch_size;
    bool shuffle;
    bool drop_last;
    xt::xarray<unsigned long> indexlist;
    int batchcount;
    Batch<DType,LType> **batchlist;
    int m_seed;
public:
    DataLoader(Dataset<DType, LType>* ptr_dataset,
            int batch_size,
            bool shuffle=true,
            bool drop_last=false,
            int seed=-1){
        if (ptr_dataset->get_data_dimension()==0&&ptr_dataset->get_data_size()==1) throw;
        this->batch_size=batch_size;
        this->shuffle=shuffle;
        this->drop_last=drop_last;
        this->m_seed=seed;
        indexlist=xt::arange(0,ptr_dataset->len(),1);
        this->ptr_dataset=ptr_dataset;
        batchcount=ptr_dataset->len()/batch_size;
        //if (!drop_last&&ptr_dataset->len()%batch_size!=0) batchcount++;
        if (this->shuffle&&this->m_seed>=0) xt::random::seed(m_seed);
        if (this->shuffle) xt::random::shuffle(indexlist);
        batchlist=new Batch<DType,LType>*[batchcount];

        for (int l = 0; l < batchcount; l++)
        {
        int k=batch_size;
        if (l == batchcount-1&&!drop_last) k=k+ptr_dataset->len()%batch_size;

        int smth1=ptr_dataset->getitem(0).getData().size();
        xt::xarray<DType> data=xt::arange(0,k*smth1,1);
        xt::xarray<DType> data_shape=xt::adapt(ptr_dataset->get_data_shape());
        data_shape(0)=k;
        data.reshape(data_shape);
        for (int i=0;i<k;i++)
            {
            int index=i+l*batch_size;
            DataLabel<DType,LType> item=ptr_dataset->getitem(indexlist(index));
            xt::view(data,i)=item.getData();
            }

        xt::xarray<LType> label;
        if (ptr_dataset->get_label_dimension()!=0||ptr_dataset->get_label_size()!=1)
        {
            int smth2=ptr_dataset->getitem(0).getLabel().size();
            label=xt::arange(0,k*smth2,1);
            xt::xarray<LType> label_shape=xt::adapt(ptr_dataset->get_label_shape());
            label_shape(0)=k;
            label.reshape(label_shape);
            for (int i=0;i<k;i++)
                {
                int index=i+l*batch_size;
                DataLabel<DType,LType> item=ptr_dataset->getitem(indexlist(index));
                    xt::view(label,i)=item.getLabel();
                }
        }
        batchlist[l]= new Batch<DType,LType>(data,label);
        }

}
    virtual ~DataLoader(){delete [] batchlist;}

    /////////////////////////////////////////////////////////////////////////
    // The section for supporting the iteration and for-each to DataLoader //
    /// START: Section                                                     //
    /////////////////////////////////////////////////////////////////////////

    class Iterators;

    Iterators begin() {
        Iterators iterator(0,batchlist);
        return iterator;
    }

    Iterators end() {
        Iterators iterator(batchcount,batchlist);
        return iterator;
    }

public:
    class Iterators
    {
    protected:
        int batchindex;
        Batch<DType,LType> **batchlist;
    public:
        Iterators(int batchindex,Batch<DType,LType> **batchlist)
        :batchindex(batchindex),batchlist(batchlist)
        {}

        bool operator!=(const Iterators &iterator) {
            return batchindex!=iterator.batchindex;
        }

        Batch<DType,LType> &operator*() {
            return *batchlist[batchindex];
        }

        Iterators &operator=(const Iterators &iterator) {
            this->batchindex=iterator.batchindex;
            this->batchlist=iterator.batchlist;
        }

        Iterators &operator++() {
            this->batchindex++;
            return *this;
        }

        Iterators operator++(int) {
            Iterators iterator=*this;
            ++*this;
            return iterator;
        }
    };

    /////////////////////////////////////////////////////////////////////////
    // The section for supporting the iteration and for-each to DataLoader //
    /// END: Section                                                       //
    /////////////////////////////////////////////////////////////////////////
};


#endif /* DATALOADER_H */


