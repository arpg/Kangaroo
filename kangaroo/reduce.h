#pragma once

#include <kangaroo/Image.h>

namespace roo {

template<typename To, typename UpType, typename Ti>
void BoxHalf( Image<To> out, const Image<Ti> in);

template<typename To, typename UpType, typename Ti>
void BoxHalfIgnoreInvalid( Image<To> out, const Image<Ti> in);

template<typename To, typename UpType, typename Ti>
inline void BoxReduce( Image<To> out, Image<Ti> in_temp, Image<To> temp, int level)
{
    const int w = in_temp.w;
    const int h = in_temp.h;

    // in_temp has size (w,h)
    // out has size (w>>l,h>>l)
    // temp has at least size (w/2,h/2)

    Image<Ti>* t[] = {&in_temp, &temp};

    for(int l=0; l < (level-1); ++l ) {
        BoxHalf<To,UpType,Ti>(
            t[(l+1) % 2]->SubImage(w >> (l+1), h >> (l+1) ),
            t[l % 2]->SubImage(w >> l, h >> l)
        );
    }

    BoxHalf<To,UpType,Ti>(out, t[(level+1)%2]->SubImage(w >> (level-1), h >> (level-1) ) );
}

template<typename T, unsigned Levels, typename UpType>
inline void BoxReduce(Pyramid<T,Levels> pyramid)
{
    // pyramid.imgs[0] has size (w,h)
    const int w = pyramid.imgs[0].w;
    const int h = pyramid.imgs[0].h;

    // Downsample from pyramid.imgs[0]
    for(int l=1; l<Levels && (w>>l > 0) && (h>>l > 0); ++l) {
        BoxHalf<T,UpType,T>(pyramid.imgs[l], pyramid.imgs[l-1]);
    }
}

template<typename T, unsigned Levels, typename UpType>
inline void BoxReduceIgnoreInvalid(Pyramid<T,Levels> pyramid)
{
    // pyramid.imgs[0] has size (w,h)
    const int w = pyramid.imgs[0].w;
    const int h = pyramid.imgs[0].h;

    // Downsample from pyramid.imgs[0]
    for(int l=1; l<Levels && (w>>l > 0) && (h>>l > 0); ++l) {
        BoxHalfIgnoreInvalid<T,UpType,T>(pyramid.imgs[l], pyramid.imgs[l-1]);
    }
}

template<typename T, unsigned Levels, typename UpType>
inline void BlurReduce(Pyramid<T,Levels> pyramid, Image<T> temp1, Image<T> temp2)
{
    // TODO: Make better

    // pyramid.imgs[0] has size (w,h)
    const int w = pyramid.imgs[0].w;
    const int h = pyramid.imgs[0].h;

    // Downsample from pyramid.imgs[0], blurring into temporary, and using BoxHalf.
    for(int l=1; l<Levels && (w>>l > 0) && (h>>l > 0); ++l) {
        const int parent = l-1;
        const int parentw = w >> parent;
        const int parenth = h >> parent;
        Blur( temp1.SubImage(parentw,parenth), pyramid.imgs[l-1], temp2.SubImage(parentw,parenth));
        BoxHalf<T,UpType,T>(pyramid.imgs[l], temp1.SubImage(parentw,parenth) );
    }
}

}
