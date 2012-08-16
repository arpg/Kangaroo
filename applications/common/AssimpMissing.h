// For some reason I need these to link.

// ------------------------------------------------------------------------------------------------
aiScene::aiScene()
	: mFlags()
	, mRootNode()
	, mNumMeshes()
	, mMeshes()
	, mNumMaterials()
	, mMaterials()
	, mNumAnimations()
	, mAnimations()
	, mNumTextures()
	, mTextures()
	, mNumLights()
	, mLights()
	, mNumCameras()
	, mCameras()
//    , mPrivate(new Assimp::ScenePrivateData())
    , mPrivate(0)
    {
	}

// ------------------------------------------------------------------------------------------------
aiScene::~aiScene()
{
	// delete all sub-objects recursively
	delete mRootNode;

	// To make sure we won't crash if the data is invalid it's
	// much better to check whether both mNumXXX and mXXX are
	// valid instead of relying on just one of them.
	if (mNumMeshes && mMeshes) 
		for( unsigned int a = 0; a < mNumMeshes; a++)
			delete mMeshes[a];
	delete [] mMeshes;

	if (mNumMaterials && mMaterials) 
		for( unsigned int a = 0; a < mNumMaterials; a++)
			delete mMaterials[a];
	delete [] mMaterials;

	if (mNumAnimations && mAnimations) 
		for( unsigned int a = 0; a < mNumAnimations; a++)
			delete mAnimations[a];
	delete [] mAnimations;

	if (mNumTextures && mTextures) 
		for( unsigned int a = 0; a < mNumTextures; a++)
			delete mTextures[a];
	delete [] mTextures;

	if (mNumLights && mLights) 
		for( unsigned int a = 0; a < mNumLights; a++)
			delete mLights[a];
	delete [] mLights;

	if (mNumCameras && mCameras) 
		for( unsigned int a = 0; a < mNumCameras; a++)
			delete mCameras[a];
	delete [] mCameras;

//	delete static_cast<Assimp::ScenePrivateData*>( mPrivate );
}

