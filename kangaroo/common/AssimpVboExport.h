#pragma once

#include <assimp/cexport.h>
#include <assimp/scene.h>
#include "AssimpMissing.h"

#include <kangaroo/Image.h>

inline aiFace* MakeAssimpImageFaces(size_t w, size_t h)
{
    aiFace* faces = new aiFace[(w-1)*(h-1)];
    aiFace* cf = faces;
    for(int x=0; x<w-1; ++x)
    {
        for(int y=0; y<h-1; ++y)
        {
            cf->mNumIndices = 4;
            cf->mIndices = new unsigned int[4];
            cf->mIndices[0] = y*w+x;
            cf->mIndices[1] = y*w+x+w;
            cf->mIndices[2] = y*w+x+w+1;
            cf->mIndices[3] = y*w+x+1;
            cf++;
        }
    }
    return faces;
}

inline aiVector3D* MakeAssimpVertices(roo::Image<float4, roo::TargetHost> vbo)
{
    aiVector3D* vertices = new aiVector3D[vbo.Area()];
    for(int x=0; x < vbo.w; ++x)
    {
        for(int y=0; y < vbo.h; ++y)
        {
            const float4 v = vbo(x,y);
            vertices[vbo.w*y+x] = aiVector3D(v.x, v.y, v.z);
        }
    }
    return vertices;
}

inline aiColor4D* MakeAssimVerticesColor(roo::Image<unsigned char, roo::TargetHost> cbo)
{
    aiColor4D* colors = new aiColor4D[cbo.Area()];
    for(int x=0; x < cbo.w; ++x)
    {
        for(int y=0; y < cbo.h; ++y)
        {
            const float c = cbo(x,y) / 255.0f;
            colors[cbo.w*y+x] = aiColor4D(c,c,c,1);
        }
    }
    return colors;
}

inline aiMesh* MakeAssimpMeshFromVbo(roo::Image<float4, roo::TargetHost> vbo)
{
    aiMesh* mesh = new aiMesh();
    mesh->mPrimitiveTypes = aiPrimitiveType_POLYGON;

    mesh->mNumVertices = vbo.Area();
    mesh->mNumFaces = (vbo.w-1)*(vbo.h-1);

    mesh->mVertices = MakeAssimpVertices(vbo);
    mesh->mFaces = MakeAssimpImageFaces(vbo.w, vbo.h);

    return mesh;
}

inline aiMesh* MakeAssimpMeshFromVboCbo(roo::Image<float4, roo::TargetHost> vbo, roo::Image<unsigned char, roo::TargetHost> img)
{
    aiMesh* mesh = new aiMesh();
    mesh->mPrimitiveTypes = aiPrimitiveType_POLYGON;

    mesh->mNumVertices = vbo.Area();
    mesh->mNumFaces = (vbo.w-1)*(vbo.h-1);

    mesh->mVertices = MakeAssimpVertices(vbo);
    mesh->mFaces = MakeAssimpImageFaces(vbo.w, vbo.h);

    mesh->mColors[0] = MakeAssimVerticesColor(img);

    return mesh;
}
