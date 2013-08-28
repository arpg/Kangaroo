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

#include <kangaroo/config.h>

#include "stdio.h"
#include "math.h"

#include "Sdf.h"
#include "MarchingCubes.h"
#include "MarchingCubesTables.h"

#ifdef HAVE_ASSIMP
#include <assimp/cexport.h>
#include <assimp/scene.h>
#include "extra/AssimpMissing.h"
#endif // HAVE_ASSIMP

namespace roo
{

#ifdef HAVE_ASSIMP

//fGetOffset finds the approximate point of intersection of the surface
// between two points with the values fValue1 and fValue2
inline float fGetOffset(float fValue1, float fValue2, float fValueDesired)
{
    const double fDelta = fValue2 - fValue1;
    if(fDelta == 0.0) {
        return 0.5;
    }
    return (fValueDesired - fValue1)/fDelta;
}

//vGetColor generates a color from a given position and normal of a point
inline void vGetColor(float3 &rfColor, const float3 &rfPosition, const float3 &rfNormal)
{
    rfColor.x = (rfNormal.x > 0.0 ? rfNormal.x : 0.0) + (rfNormal.y < 0.0 ? -0.5*rfNormal.y : 0.0) + (rfNormal.z < 0.0 ? -0.5*rfNormal.z : 0.0);
    rfColor.y = (rfNormal.y > 0.0 ? rfNormal.y : 0.0) + (rfNormal.z < 0.0 ? -0.5*rfNormal.z : 0.0) + (rfNormal.x < 0.0 ? -0.5*rfNormal.x : 0.0);
    rfColor.z = (rfNormal.z > 0.0 ? rfNormal.z : 0.0) + (rfNormal.x < 0.0 ? -0.5*rfNormal.x : 0.0) + (rfNormal.y < 0.0 ? -0.5*rfNormal.y : 0.0);
}

//vMarchCube performs the Marching Cubes algorithm on a single cube
template<typename T, typename TColor>
void vMarchCube(
    const BoundedVolume<T,roo::TargetHost> vol,
    const BoundedVolume<TColor,roo::TargetHost> volColor,
    int x, int y, int z,
    std::vector<aiVector3D>& verts,
    std::vector<aiVector3D>& norms,
    std::vector<aiFace>& faces,
    std::vector<aiColor4D>& colors,
    float fTargetValue = 0.0f
) {
    const float3 p = vol.VoxelPositionInUnits(x,y,z);
    const float3 fScale = vol.VoxelSizeUnits();

    //Make a local copy of the values at the cube's corners
    float afCubeValue[8];
    for(int iVertex = 0; iVertex < 8; iVertex++) {
        afCubeValue[iVertex] = vol.Get(x+a2fVertexOffset[iVertex][0],y+a2fVertexOffset[iVertex][1],z+a2fVertexOffset[iVertex][2]);
        if(!std::isfinite(afCubeValue[iVertex])) return;
    }

    //Find which vertices are inside of the surface and which are outside
    int iFlagIndex = 0;
    for(int iVertexTest = 0; iVertexTest < 8; iVertexTest++) {
        if(afCubeValue[iVertexTest] <= fTargetValue)
            iFlagIndex |= 1<<iVertexTest;
    }

    //Find which edges are intersected by the surface
    int iEdgeFlags = aiCubeEdgeFlags[iFlagIndex];

    //If the cube is entirely inside or outside of the surface, then there will be no intersections
    if(iEdgeFlags == 0) {
        return;
    }

    //Find the point of intersection of the surface with each edge
    //Then find the normal to the surface at those points
    float3 asEdgeVertex[12];
    float3 asEdgeNorm[12];

    for(int iEdge = 0; iEdge < 12; iEdge++)
    {
        //if there is an intersection on this edge
        if(iEdgeFlags & (1<<iEdge))
        {
            float fOffset = fGetOffset(afCubeValue[ a2iEdgeConnection[iEdge][0] ],
                                 afCubeValue[ a2iEdgeConnection[iEdge][1] ], fTargetValue);

            asEdgeVertex[iEdge] = make_float3(
                p.x + (a2fVertexOffset[ a2iEdgeConnection[iEdge][0] ][0]  +  fOffset * a2fEdgeDirection[iEdge][0]) * fScale.x,
                p.y + (a2fVertexOffset[ a2iEdgeConnection[iEdge][0] ][1]  +  fOffset * a2fEdgeDirection[iEdge][1]) * fScale.y,
                p.z + (a2fVertexOffset[ a2iEdgeConnection[iEdge][0] ][2]  +  fOffset * a2fEdgeDirection[iEdge][2]) * fScale.z
            );

            const float3 deriv = vol.GetUnitsBackwardDiffDxDyDz( asEdgeVertex[iEdge] );
            asEdgeNorm[iEdge] = deriv / length(deriv);

            if( !std::isfinite(asEdgeNorm[iEdge].x) || !std::isfinite(asEdgeNorm[iEdge].y) || !std::isfinite(asEdgeNorm[iEdge].z) ) {
                asEdgeNorm[iEdge] = make_float3(0,0,0);
            }
        }
    }


    //Draw the triangles that were found.  There can be up to five per cube
    for(int iTriangle = 0; iTriangle < 5; iTriangle++)
    {
        if(a2iTriangleConnectionTable[iFlagIndex][3*iTriangle] < 0)
            break;

        aiFace face;
        face.mNumIndices = 3;
        face.mIndices = new unsigned int[face.mNumIndices];

        for(int iCorner = 0; iCorner < 3; iCorner++)
        {
            int iVertex = a2iTriangleConnectionTable[iFlagIndex][3*iTriangle+iCorner];

            face.mIndices[iCorner] = verts.size();
            verts.push_back(aiVector3D(asEdgeVertex[iVertex].x, asEdgeVertex[iVertex].y, asEdgeVertex[iVertex].z) );
            norms.push_back(aiVector3D(asEdgeNorm[iVertex].x,   asEdgeNorm[iVertex].y,   asEdgeNorm[iVertex].z) );

            if(volColor.IsValid()) {
                const TColor c = volColor.GetUnitsTrilinearClamped(asEdgeVertex[iVertex]);
                float3 sColor = roo::ConvertPixel<float3,TColor>(c);
                colors.push_back(aiColor4D(sColor.x, sColor.y, sColor.z, 1.0f));
            }

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

    if( colors.size() == verts.size()) {
        mesh->mColors[0] = new aiColor4D[colors.size()];
        for(int i=0; i < colors.size(); ++i) {
            mesh->mColors[0][i] = colors[i];
        }
    }

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

#endif // HAVE_ASSIMP

template<typename T, typename TColor>
void SaveMesh(std::string filename, const BoundedVolume<T,TargetHost> vol, const BoundedVolume<TColor,TargetHost> volColor )
{
#ifdef HAVE_ASSIMP
    std::vector<aiVector3D> verts;
    std::vector<aiVector3D> norms;
    std::vector<aiFace> faces;
    std::vector<aiColor4D> colors;

    for(GLint iX = 0; iX < vol.Voxels().x-1; iX++) {
        for(GLint iY = 0; iY < vol.Voxels().y-1; iY++) {
            for(GLint iZ = 0; iZ < vol.Voxels().z-1; iZ++) {
                vMarchCube(vol, volColor, iX,iY,iZ, verts, norms, faces, colors);
            }
        }
    }

    aiMesh* mesh = MeshFromLists(verts,norms,faces,colors);
    SaveMesh(filename, mesh);
#else
    std::cerr << "Mesh cannot be saved. Please configure Kangaroo with Assimp." << std::endl;
#endif // HAVE_ASSIMP
}

// Instantiate templates
template void SaveMesh<SDF_t,float>(std::string, const BoundedVolume<SDF_t,TargetHost,DontManage> vol, const BoundedVolume<float,TargetHost> volColor);

}
