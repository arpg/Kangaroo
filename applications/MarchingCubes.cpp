// Marching cubes ASSIMP exporter based on Marching Cubes Example Program
// by Cory Bloyd with additional source from Paul Bourke (public domain)
// http://paulbourke.net/geometry/polygonise/
//
// Marching Cubes Example Program
// by Cory Bloyd (corysama@yahoo.com)
//
// A simple, portable and complete implementation of the Marching Cubes
// and Marching Tetrahedrons algorithms in a single source file.
// There are many ways that this code could be made faster, but the
// intent is for the code to be easy to understand.
//
// For a description of the algorithm go to
// http://astronomy.swin.edu.au/pbourke/modelling/polygonise/
//
// This code is public domain.
//

#include "stdio.h"
#include "math.h"
#include <kangaroo/Sdf.h>

#include "MarchingCubes.h"
#include "MarchingCubesTables.h"

#include <assimp/cexport.h>
#include <assimp/scene.h>
#include "common/AssimpMissing.h"

namespace Gpu
{

struct GLvector
{
    GLfloat fX;
    GLfloat fY;
    GLfloat fZ;
};

GLint     iDataSetSize = 16;
GLfloat   fStepSize = 1.0/iDataSetSize;
GLfloat   fTargetValue = 0;
GLvector  sSourcePoint[3];

//fGetOffset finds the approximate point of intersection of the surface
// between two points with the values fValue1 and fValue2
GLfloat fGetOffset(GLfloat fValue1, GLfloat fValue2, GLfloat fValueDesired)
{
    GLdouble fDelta = fValue2 - fValue1;

    if(fDelta == 0.0)
    {
        return 0.5;
    }
    return (fValueDesired - fValue1)/fDelta;
}


//vGetColor generates a color from a given position and normal of a point
void vGetColor(GLvector &rfColor, GLvector &rfPosition, GLvector &rfNormal)
{
    GLfloat fX = rfNormal.fX;
    GLfloat fY = rfNormal.fY;
    GLfloat fZ = rfNormal.fZ;
    rfColor.fX = (fX > 0.0 ? fX : 0.0) + (fY < 0.0 ? -0.5*fY : 0.0) + (fZ < 0.0 ? -0.5*fZ : 0.0);
    rfColor.fY = (fY > 0.0 ? fY : 0.0) + (fZ < 0.0 ? -0.5*fZ : 0.0) + (fX < 0.0 ? -0.5*fX : 0.0);
    rfColor.fZ = (fZ > 0.0 ? fZ : 0.0) + (fX < 0.0 ? -0.5*fX : 0.0) + (fY < 0.0 ? -0.5*fY : 0.0);
}

void vNormalizeVector(GLvector &rfVectorResult, GLvector &rfVectorSource)
{
    GLfloat fOldLength;
    GLfloat fScale;

    fOldLength = sqrtf( (rfVectorSource.fX * rfVectorSource.fX) +
                        (rfVectorSource.fY * rfVectorSource.fY) +
                        (rfVectorSource.fZ * rfVectorSource.fZ) );

    if(fOldLength == 0.0)
    {
        rfVectorResult.fX = rfVectorSource.fX;
        rfVectorResult.fY = rfVectorSource.fY;
        rfVectorResult.fZ = rfVectorSource.fZ;
    }
    else
    {
        fScale = 1.0/fOldLength;
        rfVectorResult.fX = rfVectorSource.fX*fScale;
        rfVectorResult.fY = rfVectorSource.fY*fScale;
        rfVectorResult.fZ = rfVectorSource.fZ*fScale;
    }
}

//fSample1 finds the distance of (fX, fY, fZ) from three moving points
GLfloat fSample(GLfloat fX, GLfloat fY, GLfloat fZ)
{
    GLdouble fResult = 0.0;
    GLdouble fDx, fDy, fDz;
    fDx = fX - sSourcePoint[0].fX;
    fDy = fY - sSourcePoint[0].fY;
    fDz = fZ - sSourcePoint[0].fZ;
    fResult += 0.5/(fDx*fDx + fDy*fDy + fDz*fDz);

    fDx = fX - sSourcePoint[1].fX;
    fDy = fY - sSourcePoint[1].fY;
    fDz = fZ - sSourcePoint[1].fZ;
    fResult += 1.0/(fDx*fDx + fDy*fDy + fDz*fDz);

    fDx = fX - sSourcePoint[2].fX;
    fDy = fY - sSourcePoint[2].fY;
    fDz = fZ - sSourcePoint[2].fZ;
    fResult += 1.5/(fDx*fDx + fDy*fDy + fDz*fDz);

    return fResult;
}

//vGetNormal() finds the gradient of the scalar field at a point
//This gradient can be used as a very accurate vertx normal for lighting calculations
void vGetNormal(GLvector &rfNormal, GLfloat fX, GLfloat fY, GLfloat fZ)
{
    rfNormal.fX = fSample(fX-0.01, fY, fZ) - fSample(fX+0.01, fY, fZ);
    rfNormal.fY = fSample(fX, fY-0.01, fZ) - fSample(fX, fY+0.01, fZ);
    rfNormal.fZ = fSample(fX, fY, fZ-0.01) - fSample(fX, fY, fZ+0.01);
    vNormalizeVector(rfNormal, rfNormal);
}

//vMarchCube1 performs the Marching Cubes algorithm on a single cube
template<typename T>
void vMarchCube(
    const BoundedVolume<T,Gpu::TargetHost> vol,
    int x, int y, int z,
    std::vector<aiVector3D>& verts,
    std::vector<aiVector3D>& norms,
    std::vector<aiFace>& faces,
    std::vector<aiColor4D>& colors
) {
    const float3 p = vol.VoxelPositionInUnits(x,y,z);
    GLfloat fX = p.x;
    GLfloat fY = p.y;
    GLfloat fZ = p.z;

    // TODO: Allow voxel to have different scales in each direction
    GLfloat fScale = vol.VoxelSizeUnits().x;

//    GLint aiCubeEdgeFlags[256];
//    GLint a2iTriangleConnectionTable[256][16];

    GLint iCorner, iVertex, iVertexTest, iEdge, iTriangle, iFlagIndex, iEdgeFlags;
    GLfloat fOffset;
    GLvector sColor;
    GLfloat afCubeValue[8];
    GLvector asEdgeVertex[12];
    GLvector asEdgeNorm[12];

    //Make a local copy of the values at the cube's corners
    for(iVertex = 0; iVertex < 8; iVertex++)
    {
        afCubeValue[iVertex] = vol.Get(x+a2fVertexOffset[iVertex][0],y+a2fVertexOffset[iVertex][1],z+a2fVertexOffset[iVertex][2]);
//      afCubeValue[iVertex] = fSample(fX + a2fVertexOffset[iVertex][0]*fScale,
//                                       fY + a2fVertexOffset[iVertex][1]*fScale,
//                                       fZ + a2fVertexOffset[iVertex][2]*fScale);
    }

    //Find which vertices are inside of the surface and which are outside
    iFlagIndex = 0;
    for(iVertexTest = 0; iVertexTest < 8; iVertexTest++)
    {
        if(afCubeValue[iVertexTest] <= fTargetValue)
            iFlagIndex |= 1<<iVertexTest;
    }

    //Find which edges are intersected by the surface
    iEdgeFlags = aiCubeEdgeFlags[iFlagIndex];

    //If the cube is entirely inside or outside of the surface, then there will be no intersections
    if(iEdgeFlags == 0)
    {
        return;
    }

    //Find the point of intersection of the surface with each edge
    //Then find the normal to the surface at those points
    for(iEdge = 0; iEdge < 12; iEdge++)
    {
        //if there is an intersection on this edge
        if(iEdgeFlags & (1<<iEdge))
        {
            fOffset = fGetOffset(afCubeValue[ a2iEdgeConnection[iEdge][0] ],
                                 afCubeValue[ a2iEdgeConnection[iEdge][1] ], fTargetValue);

            asEdgeVertex[iEdge].fX = fX + (a2fVertexOffset[ a2iEdgeConnection[iEdge][0] ][0]  +  fOffset * a2fEdgeDirection[iEdge][0]) * fScale;
            asEdgeVertex[iEdge].fY = fY + (a2fVertexOffset[ a2iEdgeConnection[iEdge][0] ][1]  +  fOffset * a2fEdgeDirection[iEdge][1]) * fScale;
            asEdgeVertex[iEdge].fZ = fZ + (a2fVertexOffset[ a2iEdgeConnection[iEdge][0] ][2]  +  fOffset * a2fEdgeDirection[iEdge][2]) * fScale;

            const float3 N = vol.GetUnitsBackwardDiffDxDyDz( make_float3(asEdgeVertex[iEdge].fX, asEdgeVertex[iEdge].fY, asEdgeVertex[iEdge].fZ) );

            // TODO: Compute normal
            asEdgeNorm[iEdge].fX = N.x;
            asEdgeNorm[iEdge].fX = N.y;
            asEdgeNorm[iEdge].fX = N.z;
//            vGetNormal(asEdgeNorm[iEdge], asEdgeVertex[iEdge].fX, asEdgeVertex[iEdge].fY, asEdgeVertex[iEdge].fZ);
        }
    }


    //Draw the triangles that were found.  There can be up to five per cube
    for(iTriangle = 0; iTriangle < 5; iTriangle++)
    {
        if(a2iTriangleConnectionTable[iFlagIndex][3*iTriangle] < 0)
            break;

        aiFace face;
        face.mNumIndices = 3;
        face.mIndices = new unsigned int[face.mNumIndices];

        for(iCorner = 0; iCorner < 3; iCorner++)
        {
            iVertex = a2iTriangleConnectionTable[iFlagIndex][3*iTriangle+iCorner];
            // TODO: Get colour
            vGetColor(sColor, asEdgeVertex[iVertex], asEdgeNorm[iVertex]);

            face.mIndices[iCorner] = verts.size();
            verts.push_back(aiVector3D(asEdgeVertex[iVertex].fX, asEdgeVertex[iVertex].fY, asEdgeVertex[iVertex].fZ) );
            norms.push_back(aiVector3D(asEdgeNorm[iVertex].fX,   asEdgeNorm[iVertex].fY,   asEdgeNorm[iVertex].fZ) );
            colors.push_back(aiColor4D(sColor.fX, sColor.fY, sColor.fZ, 1.0f));
        }

        faces.push_back(face);
    }
}

aiMesh* MeshFromLists(
    const std::vector<aiVector3D>& verts,
    const std::vector<aiVector3D>& norms,
    const std::vector<aiFace>& faces,
    const std::vector<aiColor4D>& colors
) {
    aiMesh* mesh = new aiMesh();
    mesh->mPrimitiveTypes = aiPrimitiveType_TRIANGLE;

    mesh->mNumVertices = verts.size();
    mesh->mVertices = new aiVector3D[verts.size()];
    for(int i=0; i < verts.size(); ++i) {
        mesh->mVertices[i] = verts[i];
    }

    if(norms.size() == verts.size()) {
        mesh->mNormals = new aiVector3D[norms.size()];
        for(int i=0; i < norms.size(); ++i) {
            mesh->mNormals[i] = norms[i];
        }
    }else{
        mesh->mNormals = 0;
    }

    mesh->mNumFaces = faces.size();
    mesh->mFaces = new aiFace[faces.size()];
    for(int i=0; i < faces.size(); ++i) {
        mesh->mFaces[i] = faces[i];
    }

//    if( colors.size() == verts.size()) {
//        mesh->mColors[0] = new aiColor4D[colors.size()];
//        for(int i=0; i < colors.size(); ++i) {
//            mesh->mColors[0][i] = colors[i];
//        }
//    }

    return mesh;
}

void SaveMesh(std::string filename, aiMesh* mesh)
{
    // Create root node which indexes first mesh
    aiNode* root = new aiNode();
    root->mNumMeshes = 1;
    root->mMeshes = new unsigned int[root->mNumMeshes];
    root->mMeshes[0] = 0;
    root->mName = "root";

    aiMaterial* material = new aiMaterial();

    // Create scene to contain root node and mesh
    aiScene scene;
    scene.mRootNode = root;
    scene.mNumMeshes = 1;
    scene.mMeshes = new aiMesh*[scene.mNumMeshes];
    scene.mMeshes[0] = mesh;
    scene.mNumMaterials = 1;
    scene.mMaterials = new aiMaterial*[scene.mNumMaterials];
    scene.mMaterials[0] = material;

    aiReturn res = aiExportScene(&scene, "ply", (filename + ".ply").c_str(), 0);
    std::cout << "Mesh export result: " << res << std::endl;
}

template<typename T>
void SaveMesh(std::string filename, const BoundedVolume<T,TargetHost> vol )
{
    std::vector<aiVector3D> verts;
    std::vector<aiVector3D> norms;
    std::vector<aiFace> faces;
    std::vector<aiColor4D> colors;

    for(GLint iX = 0; iX < vol.Voxels().x-1; iX++) {
        for(GLint iY = 0; iY < vol.Voxels().y-1; iY++) {
            for(GLint iZ = 0; iZ < vol.Voxels().z-1; iZ++) {
                vMarchCube(vol, iX,iY,iZ, verts, norms, faces, colors);
            }
        }
    }

    aiMesh* mesh = MeshFromLists(verts,norms,faces,colors);
    SaveMesh(filename, mesh);
}

// Instantiate templates
template void SaveMesh<SDF_t>(std::string, const BoundedVolume<SDF_t,TargetHost,DontManage> vol);



}
