#include <string>
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <vector>
#include <algorithm>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "cudaRenderer.h"
#include "image.h"
#include "noise.h"
#include "sceneLoader.h"
#include "util.h"
#include "CycleTimer.h"
#include "circleBoxTest.cu_inl"
////////////////////////////////////////////////////////////////////////////////////////
// Putting all the cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////

// #define PRINT_DEBUG

#define DEBUG

#ifdef DEBUG
#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr, "CUDA Error: %s at %s:%d\n",
        cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#else
#define cudaCheckError(ans) ans
#endif

struct GlobalConstants {

    SceneName sceneName;

    int numCircles;
    float* position;
    float* velocity;
    float* color;
    float* radius;

    int imageWidth;
    int imageHeight;
    float* imageData;
};

// Global variable that is in scope, but read-only, for all cuda
// kernels.  The __constant__ modifier designates this variable will
// be stored in special "constant" memory on the GPU. (we didn't talk
// about this type of memory in class, but constant memory is a fast
// place to put read-only variables).
__constant__ GlobalConstants cuConstRendererParams;

// read-only lookup tables used to quickly compute noise (needed by
// advanceAnimation for the snowflake scene)
__constant__ int    cuConstNoiseYPermutationTable[256];
__constant__ int    cuConstNoiseXPermutationTable[256];
__constant__ float  cuConstNoise1DValueTable[256];

// color ramp table needed for the color ramp lookup shader
#define COLOR_MAP_SIZE 5
__constant__ float  cuConstColorRamp[COLOR_MAP_SIZE][3];


// including parts of the CUDA code from external files to keep this
// file simpler and to seperate code that should not be modified
#include "noiseCuda.cu_inl"
#include "lookupColor.cu_inl"


// kernelClearImageSnowflake -- (CUDA device code)
//
// Clear the image, setting the image to the white-gray gradation that
// is used in the snowflake image
__global__ void kernelClearImageSnowflake() {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float shade = .4f + .45f * static_cast<float>(height-imageY) / height;
    float4 value = make_float4(shade, shade, shade, 1.f);

    // write to global memory: As an optimization, I use a float4
    // store, that results in more efficient code than if I coded this
    // up as four seperate fp32 stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelClearImage --  (CUDA device code)
//
// Clear the image, setting all pixels to the specified color rgba
__global__ void kernelClearImage(float r, float g, float b, float a) {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float4 value = make_float4(r, g, b, a);

    // write to global memory: As an optimization, I use a float4
    // store, that results in more efficient code than if I coded this
    // up as four seperate fp32 stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelAdvanceFireWorks
// 
// Update the position of the fireworks (if circle is firework)
__global__ void kernelAdvanceFireWorks() {
    const float dt = 1.f / 60.f;
    const float pi = 3.14159;
    const float maxDist = 0.25f;

    float* velocity = cuConstRendererParams.velocity;
    float* position = cuConstRendererParams.position;
    float* radius = cuConstRendererParams.radius;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles)
        return;

    if (0 <= index && index < NUM_FIREWORKS) { // firework center; no update 
        return;
    }

    // determine the fire-work center/spark indices
    int fIdx = (index - NUM_FIREWORKS) / NUM_SPARKS;
    int sfIdx = (index - NUM_FIREWORKS) % NUM_SPARKS;

    int index3i = 3 * fIdx;
    int sIdx = NUM_FIREWORKS + fIdx * NUM_SPARKS + sfIdx;
    int index3j = 3 * sIdx;

    float cx = position[index3i];
    float cy = position[index3i+1];

    // update position
    position[index3j] += velocity[index3j] * dt;
    position[index3j+1] += velocity[index3j+1] * dt;

    // fire-work sparks
    float sx = position[index3j];
    float sy = position[index3j+1];

    // compute vector from firework-spark
    float cxsx = sx - cx;
    float cysy = sy - cy;

    // compute distance from fire-work 
    float dist = sqrt(cxsx * cxsx + cysy * cysy);
    if (dist > maxDist) { // restore to starting position 
        // random starting position on fire-work's rim
        float angle = (sfIdx * 2 * pi)/NUM_SPARKS;
        float sinA = sin(angle);
        float cosA = cos(angle);
        float x = cosA * radius[fIdx];
        float y = sinA * radius[fIdx];

        position[index3j] = position[index3i] + x;
        position[index3j+1] = position[index3i+1] + y;
        position[index3j+2] = 0.0f;

        // travel scaled unit length 
        velocity[index3j] = cosA/5.0;
        velocity[index3j+1] = sinA/5.0;
        velocity[index3j+2] = 0.0f;
    }
}

// kernelAdvanceHypnosis   
//
// Update the radius/color of the circles
__global__ void kernelAdvanceHypnosis() { 
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles) 
        return; 

    float* radius = cuConstRendererParams.radius; 

    float cutOff = 0.5f;
    // place circle back in center after reaching threshold radisus 
    if (radius[index] > cutOff) { 
        radius[index] = 0.02f; 
    } else { 
        radius[index] += 0.01f; 
    }   
}   


// kernelAdvanceBouncingBalls
// 
// Update the positino of the balls
__global__ void kernelAdvanceBouncingBalls() { 
    const float dt = 1.f / 60.f;
    const float kGravity = -2.8f; // sorry Newton
    const float kDragCoeff = -0.8f;
    const float epsilon = 0.001f;

    int index = blockIdx.x * blockDim.x + threadIdx.x; 
   
    if (index >= cuConstRendererParams.numCircles) 
        return; 

    float* velocity = cuConstRendererParams.velocity; 
    float* position = cuConstRendererParams.position; 

    int index3 = 3 * index;
    // reverse velocity if center position < 0
    float oldVelocity = velocity[index3+1];
    float oldPosition = position[index3+1];

    if (oldVelocity == 0.f && oldPosition == 0.f) { // stop-condition 
        return;
    }

    if (position[index3+1] < 0 && oldVelocity < 0.f) { // bounce ball 
        velocity[index3+1] *= kDragCoeff;
    }

    // update velocity: v = u + at (only along y-axis)
    velocity[index3+1] += kGravity * dt;

    // update positions (only along y-axis)
    position[index3+1] += velocity[index3+1] * dt;

    if (fabsf(velocity[index3+1] - oldVelocity) < epsilon
        && oldPosition < 0.0f
        && fabsf(position[index3+1]-oldPosition) < epsilon) { // stop ball 
        velocity[index3+1] = 0.f;
        position[index3+1] = 0.f;
    }
}

// kernelAdvanceSnowflake -- (CUDA device code)
//
// move the snowflake animation forward one time step.  Updates circle
// positions and velocities.  Note how the position of the snowflake
// is reset if it moves off the left, right, or bottom of the screen.
__global__ void kernelAdvanceSnowflake() {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numCircles)
        return;

    const float dt = 1.f / 60.f;
    const float kGravity = -1.8f; // sorry Newton
    const float kDragCoeff = 2.f;

    int index3 = 3 * index;

    float* positionPtr = &cuConstRendererParams.position[index3];
    float* velocityPtr = &cuConstRendererParams.velocity[index3];

    // loads from global memory
    float3 position = *((float3*)positionPtr);
    float3 velocity = *((float3*)velocityPtr);

    // hack to make farther circles move more slowly, giving the
    // illusion of parallax
    float forceScaling = fmin(fmax(1.f - position.z, .1f), 1.f); // clamp

    // add some noise to the motion to make the snow flutter
    float3 noiseInput;
    noiseInput.x = 10.f * position.x;
    noiseInput.y = 10.f * position.y;
    noiseInput.z = 255.f * position.z;
    float2 noiseForce = cudaVec2CellNoise(noiseInput, index);
    noiseForce.x *= 7.5f;
    noiseForce.y *= 5.f;

    // drag
    float2 dragForce;
    dragForce.x = -1.f * kDragCoeff * velocity.x;
    dragForce.y = -1.f * kDragCoeff * velocity.y;

    // update positions
    position.x += velocity.x * dt;
    position.y += velocity.y * dt;

    // update velocities
    velocity.x += forceScaling * (noiseForce.x + dragForce.y) * dt;
    velocity.y += forceScaling * (kGravity + noiseForce.y + dragForce.y) * dt;

    float radius = cuConstRendererParams.radius[index];

    // if the snowflake has moved off the left, right or bottom of
    // the screen, place it back at the top and give it a
    // pseudorandom x position and velocity.
    if ( (position.y + radius < 0.f) ||
         (position.x + radius) < -0.f ||
         (position.x - radius) > 1.f)
    {
        noiseInput.x = 255.f * position.x;
        noiseInput.y = 255.f * position.y;
        noiseInput.z = 255.f * position.z;
        noiseForce = cudaVec2CellNoise(noiseInput, index);

        position.x = .5f + .5f * noiseForce.x;
        position.y = 1.35f + radius;

        // restart from 0 vertical velocity.  Choose a
        // pseudo-random horizontal velocity.
        velocity.x = 2.f * noiseForce.y;
        velocity.y = 0.f;
    }

    // store updated positions and velocities to global memory
    *((float3*)positionPtr) = position;
    *((float3*)velocityPtr) = velocity;
}

// shadePixel -- (CUDA device code)
//
// given a pixel and a circle, determines the contribution to the
// pixel from the circle.  Update of the image is done in this
// function.  Called by kernelRenderCircles()
__device__ __inline__ void
shadePixel(int circleIndex, float2 pixelCenter, float3 p, float4* imagePtr) {

    float diffX = p.x - pixelCenter.x;
    float diffY = p.y - pixelCenter.y;
    float pixelDist = diffX * diffX + diffY * diffY;

    float rad = cuConstRendererParams.radius[circleIndex];;
    float maxDist = rad * rad;
    // printf("update circle at pixel x:%f, y: %f - pixelDist: %f, maxDist: %f\n", pixelCenter.x, pixelCenter.y, pixelDist, maxDist);

    // circle does not contribute to the image
    if (pixelDist > maxDist)
        return;

    float3 rgb;
    float alpha;

    // there is a non-zero contribution.  Now compute the shading value

    // suggestion: This conditional is in the inner loop.  Although it
    // will evaluate the same for all threads, there is overhead in
    // setting up the lane masks etc to implement the conditional.  It
    // would be wise to perform this logic outside of the loop next in
    // kernelRenderCircles.  (If feeling good about yourself, you
    // could use some specialized template magic).
    if (cuConstRendererParams.sceneName == SNOWFLAKES || cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME) {

        const float kCircleMaxAlpha = .5f;
        const float falloffScale = 4.f;

        float normPixelDist = sqrt(pixelDist) / rad;
        rgb = lookupColor(normPixelDist);

        float maxAlpha = .6f + .4f * (1.f-p.z);
        maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f); // kCircleMaxAlpha * clamped value
        alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);

    } else {
        // simple: each circle has an assigned color
        int index3 = 3 * circleIndex;
        rgb = *(float3*)&(cuConstRendererParams.color[index3]);
        alpha = .5f;
    }

    float oneMinusAlpha = 1.f - alpha;

    // BEGIN SHOULD-BE-ATOMIC REGION
    // global memory read
    // printf("update circle at pixel x:%f, y: %f\n", pixelCenter.x, pixelCenter.y);

    float4 existingColor = *imagePtr;
    float4 newColor;
    newColor.x = alpha * rgb.x + oneMinusAlpha * existingColor.x;
    newColor.y = alpha * rgb.y + oneMinusAlpha * existingColor.y;
    newColor.z = alpha * rgb.z + oneMinusAlpha * existingColor.z;
    newColor.w = alpha + existingColor.w;

    // global memory write
    *imagePtr = newColor;

    // END SHOULD-BE-ATOMIC REGION
}

// kernelRenderCircles -- (CUDA device code)
//
// Each thread renders a circle.  Since there is no protection to
// ensure order of update or mutual exclusion on the output image, the
// resulting image will be incorrect.
__global__ void kernelRenderCircles() {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numCircles)
        return;

    int index3 = 3 * index;

    // read position and radius
    float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
    float  rad = cuConstRendererParams.radius[index];

    // compute the bounding box of the circle. The bound is in integer
    // screen coordinates, so it's clamped to the edges of the screen.
    short imageWidth = cuConstRendererParams.imageWidth;
    short imageHeight = cuConstRendererParams.imageHeight;
    short minX = static_cast<short>(imageWidth * (p.x - rad));
    short maxX = static_cast<short>(imageWidth * (p.x + rad)) + 1;
    short minY = static_cast<short>(imageHeight * (p.y - rad));
    short maxY = static_cast<short>(imageHeight * (p.y + rad)) + 1;

    // a bunch of clamps.  Is there a CUDA built-in for this?
    short screenMinX = (minX > 0) ? ((minX < imageWidth) ? minX : imageWidth) : 0;
    short screenMaxX = (maxX > 0) ? ((maxX < imageWidth) ? maxX : imageWidth) : 0;
    short screenMinY = (minY > 0) ? ((minY < imageHeight) ? minY : imageHeight) : 0;
    short screenMaxY = (maxY > 0) ? ((maxY < imageHeight) ? maxY : imageHeight) : 0;

    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;

    // for all pixels in the bonding box
    for (int pixelY=screenMinY; pixelY<screenMaxY; pixelY++) {
        float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixelY * imageWidth + screenMinX)]);
        for (int pixelX=screenMinX; pixelX<screenMaxX; pixelX++) {
            float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
                                                 invHeight * (static_cast<float>(pixelY) + 0.5f));
            shadePixel(index, pixelCenterNorm, p, imgPtr);
            imgPtr++;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////


CudaRenderer::CudaRenderer() {
    image = NULL;

    numCircles = 0;
    position = NULL;
    velocity = NULL;
    color = NULL;
    radius = NULL;

    cudaDevicePosition = NULL;
    cudaDeviceVelocity = NULL;
    cudaDeviceColor = NULL;
    cudaDeviceRadius = NULL;
    cudaDeviceImageData = NULL;
}

CudaRenderer::~CudaRenderer() {

    if (image) {
        delete image;
    }

    if (position) {
        delete [] position;
        delete [] velocity;
        delete [] color;
        delete [] radius;
    }

    if (cudaDevicePosition) {
        cudaFree(cudaDevicePosition);
        cudaFree(cudaDeviceVelocity);
        cudaFree(cudaDeviceColor);
        cudaFree(cudaDeviceRadius);
        cudaFree(cudaDeviceImageData);
    }
}

const Image*
CudaRenderer::getImage() {

    // need to copy contents of the rendered image from device memory
    // before we expose the Image object to the caller

    printf("Copying image data from device\n");

    cudaMemcpy(image->data,
               cudaDeviceImageData,
               sizeof(float) * 4 * image->width * image->height,
               cudaMemcpyDeviceToHost);

    return image;
}

void
CudaRenderer::loadScene(SceneName scene, int seed) {
    sceneName = scene;
    loadCircleScene(sceneName, numCircles, position, velocity, color, radius, seed);
}

void
CudaRenderer::setup() {

    int deviceCount = 0;
    std::string name;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Initializing CUDA for CudaRenderer\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        name = deviceProps.name;

        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
    
    // By this time the scene should be loaded.  Now copy all the key
    // data structures into device memory so they are accessible to
    // CUDA kernels
    //
    // See the CUDA Programmer's Guide for descriptions of
    // cudaMalloc and cudaMemcpy

    cudaMalloc(&cudaDevicePosition, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceVelocity, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceColor, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceRadius, sizeof(float) * numCircles);
    cudaMalloc(&cudaDeviceImageData, sizeof(float) * 4 * image->width * image->height);

    cudaMemcpy(cudaDevicePosition, position, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceVelocity, velocity, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceColor, color, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceRadius, radius, sizeof(float) * numCircles, cudaMemcpyHostToDevice);

    // Initialize parameters in constant memory.  We didn't talk about
    // constant memory in class, but the use of read-only constant
    // memory here is an optimization over just sticking these values
    // in device global memory.  NVIDIA GPUs have a few special tricks
    // for optimizing access to constant memory.  Using global memory
    // here would have worked just as well.  See the Programmer's
    // Guide for more information about constant memory.

    GlobalConstants params;
    params.sceneName = sceneName;
    params.numCircles = numCircles;
    params.imageWidth = image->width;
    params.imageHeight = image->height;
    params.position = cudaDevicePosition;
    params.velocity = cudaDeviceVelocity;
    params.color = cudaDeviceColor;
    params.radius = cudaDeviceRadius;
    params.imageData = cudaDeviceImageData;

    cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));

    // also need to copy over the noise lookup tables, so we can
    // implement noise on the GPU
    int* permX;
    int* permY;
    float* value1D;
    getNoiseTables(&permX, &permY, &value1D);
    cudaMemcpyToSymbol(cuConstNoiseXPermutationTable, permX, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoiseYPermutationTable, permY, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoise1DValueTable, value1D, sizeof(float) * 256);

    // last, copy over the color table that's used by the shading
    // function for circles in the snowflake demo

    float lookupTable[COLOR_MAP_SIZE][3] = {
        {1.f, 1.f, 1.f},
        {1.f, 1.f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, 0.8f, 1.f},
    };

    cudaMemcpyToSymbol(cuConstColorRamp, lookupTable, sizeof(float) * 3 * COLOR_MAP_SIZE);

}

// allocOutputImage --
//
// Allocate buffer the renderer will render into.  Check status of
// image first to avoid memory leak.
void
CudaRenderer::allocOutputImage(int width, int height) {

    if (image)
        delete image;
    image = new Image(width, height);
}

// clearImage --
//
// Clear's the renderer's target image.  The state of the image after
// the clear depends on the scene being rendered.
void
CudaRenderer::clearImage() {

    // 256 threads per block is a healthy number
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(
        (image->width + blockDim.x - 1) / blockDim.x,
        (image->height + blockDim.y - 1) / blockDim.y);

    if (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME) {
        kernelClearImageSnowflake<<<gridDim, blockDim>>>();
    } else {
        kernelClearImage<<<gridDim, blockDim>>>(1.f, 1.f, 1.f, 1.f);
    }
    cudaDeviceSynchronize();
}

// advanceAnimation --
//
// Advance the simulation one time step.  Updates all circle positions
// and velocities
void
CudaRenderer::advanceAnimation() {
     // 256 threads per block is a healthy number
    dim3 blockDim(256, 1);
    dim3 gridDim((numCircles + blockDim.x - 1) / blockDim.x);

    // only the snowflake scene has animation
    if (sceneName == SNOWFLAKES) {
        kernelAdvanceSnowflake<<<gridDim, blockDim>>>();
    } else if (sceneName == BOUNCING_BALLS) {
        kernelAdvanceBouncingBalls<<<gridDim, blockDim>>>();
    } else if (sceneName == HYPNOSIS) {
        kernelAdvanceHypnosis<<<gridDim, blockDim>>>();
    } else if (sceneName == FIREWORKS) { 
        kernelAdvanceFireWorks<<<gridDim, blockDim>>>(); 
    }
    cudaDeviceSynchronize();
}

void
CudaRenderer::renderBase() {

    // 256 threads per block is a healthy number
    dim3 blockDim(256, 1);
    dim3 gridDim((numCircles + blockDim.x - 1) / blockDim.x);

    kernelRenderCircles<<<gridDim, blockDim>>>();
    cudaDeviceSynchronize();
}

/***
 * circleIndex: the index of the circle
 * pixelCenter: current pixel center
 * p: position of the circle
 * imagePtr: image
 */
__device__ __inline__ void
shadePixelForCircle(int circleIndex, float2 pixelCenter, float3 p, float rad, float4 &existingColor) {

    float diffX = p.x - pixelCenter.x;
    float diffY = p.y - pixelCenter.y;
    float pixelDist = diffX * diffX + diffY * diffY;

    float maxDist = rad * rad;

    // circle does not contribute to the image
    if (pixelDist > maxDist)
        return;

    float3 rgb;
    float alpha;

    // there is a non-zero contribution.  Now compute the shading value

    // suggestion: This conditional is in the inner loop.  Although it
    // will evaluate the same for all threads, there is overhead in
    // setting up the lane masks etc to implement the conditional.  It
    // would be wise to perform this logic outside of the loop next in
    // kernelRenderCircles.  (If feeling good about yourself, you
    // could use some specialized template magic).
    if (cuConstRendererParams.sceneName == SNOWFLAKES || cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME) {

        const float kCircleMaxAlpha = .5f;
        const float falloffScale = 4.f;

        float normPixelDist = sqrt(pixelDist) / rad;
        rgb = lookupColor(normPixelDist);

        float maxAlpha = .6f + .4f * (1.f-p.z);
        maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f); // kCircleMaxAlpha * clamped value
        alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);

    } else {
        // simple: each circle has an assigned color
        int index3 = 3 * circleIndex;
        rgb = *(float3*)&(cuConstRendererParams.color[index3]);
        alpha = .5f;
    }

    float oneMinusAlpha = 1.f - alpha;

    // BEGIN SHOULD-BE-ATOMIC REGION
    // global memory read

    existingColor.x = alpha * rgb.x + oneMinusAlpha * existingColor.x;
    existingColor.y = alpha * rgb.y + oneMinusAlpha * existingColor.y;
    existingColor.z = alpha * rgb.z + oneMinusAlpha * existingColor.z;
    existingColor.w = alpha + existingColor.w;

    // END SHOULD-BE-ATOMIC REGION
}

__global__ void kernelRenderPixels(int width, int height) {

    int pixelX = blockIdx.x * blockDim.x + threadIdx.x; //col
    int pixelY = blockIdx.y * blockDim.y + threadIdx.y; // row

    if (pixelX > height  || pixelY > width )
        return;

    short imageWidth = cuConstRendererParams.imageWidth;
    short imageHeight = cuConstRendererParams.imageHeight;
    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;

    // printf("update  pixel x:%d, y: %d \n", pixelX, pixelY);
    float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
                                        invHeight * (static_cast<float>(pixelY) + 0.5f));
    float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixelY * imageWidth + pixelX)]);
    float4 existingColor = *imgPtr;

    // Go through each circle and apply effect
    for (int index = 0; index < cuConstRendererParams.numCircles; index++){
        int index3 = 3 * index;

        // read position and radius
        float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
        float rad = cuConstRendererParams.radius[index];;

        // printf("update circle at pixel circle %d, x:%d, y: %d \n", index, pixelX, pixelY);        
        shadePixelForCircle(index, pixelCenterNorm, p, rad, existingColor);
    }
    *imgPtr = existingColor;

}

void CudaRenderer::renderPixels() {
    // 256 threads per block is a healthy number
    dim3 blockSize(32, 32);
    dim3 gridSize( (image->width + blockSize.x - 1) / blockSize.x, (image->height + blockSize.y - 1) / blockSize.y);

    kernelRenderPixels<<<gridSize, blockSize>>>(image->width, image->height);
    cudaCheckError(cudaDeviceSynchronize())
}


// __device__ __inline__ int
// pixelInCircle(float2* circleCenter, float  maxDist, float2 pixelCenter) {
//     float diffX = circleCenter.x - pixelCenter.x;
//     float diffY = circleCenter.y - pixelCenter.y;
//     float pixelDist = diffX * diffX + diffY * diffY;

//     return pixelDist > maxDist ? 0: 1;
// }

/**
 * this kernel computes the ordering of BLOCK Size consecutive circles in the circle list
 * where each block corresponds to a non overlapping 2d box in the image.
 *
 * shared memory contains the max on whether the BLOCKSIZE number of circle are contained inside the 
 * it
 * 
 * One it is constructed, each pixel in the block will only try to apply the effect from those circles
 * 
//  */
// __global__ void kernelRenderPixelsBlocks(int circle_start, int BLOCKSIZE, uint* output, int num_circles){
//     // each thread is a block of size (1024)
//     __shared__ float3 __circles[BLOCKSIZE]; 
//     __shared__ uint __prefixSumInput[BLOCKSIZE];
//     __shared__ uint __prefixSumOutput[BLOCKSIZE];
//     __shared__ uint __prefixSumScratch[2 * BLOCKSIZE];

//     int block_id = blockIdx.x;

//     // get box dimensions, top/bottom/left/right. Each thread of thre block identify a circle
//     float boxL, boxR, boxT, boxB;

//     short imageWidth = cuConstRendererParams.imageWidth;
//     short imageHeight = cuConstRendererParams.imageHeight;
//     float invWidth = 1.f / imageWidth;
//     float invHeight = 1.f / imageHeight;


//     // linear thread is used to access BLOCK SIZE consecutive amount of shared memory within a thread Block
//     int linearThreadIndex =  threadIdx.y * blockDim.x + threadIdx.x;

//     // for each blockID (which is a box, check for circle that interset)
//     // the next BLOCK SIZE amount of circles
//     int circle_index = circle_start + threadIdx.x;
//     if (circle_index > &cuConstRendererParams.numCircles)
//         return;

//     int index3 = 3 * circle_index;
//     __circles[circle_index] = *(float3*)(&cuConstRendererParams.position[index3]);

//     float circleX = __circles[threadIdx.x].x;
//     float circleY = __circles[threadIdx.x].y;
//     float circleRadius = cuConstRendererParams.radius[circleIndex];;

//     // set whether the circle is in the current Box 
//     __prefixSumInput[linearThreadIndex % BLOCKSIZE] = circleInBoxConservative(circleX, circleY, circleRadius, boxL,  boxR,  boxT,  boxB)

//     // prefixSum input is ready to be scanned
//     __syncthreads__();
    
//     // sharedMemExclusiveScan(linearThreadIndex, __prefixSumInput, __prefixSumOutput, __prefixSumScratch, BLOCKSIZE);

//     // __syncthreads__();

//     // At this point, all threads in this block have a mask (__prefixSumOutput) indicated which of the circles 
//     // has an impact on the pixel.

//     // float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * )];


//     // if(__prefixSumInput[__circles[circle_index]] == 1){
//     //     // 
//     //     shadePixelForCircle(circleIndex, float2 pixelCenter, __circles[circle_index], float4 &existingColor) {

//     // }

//     // float4 existingColor = *imgPtr;

//     // // Go through each circle and apply effect
//     // for (int index = 0; index < cuConstRendererParams.numCircles; index++){
//     //     int index3 = 3 * index;

//     //     // read position and radius
//     //     float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
//     //     // printf("update circle at pixel circle %d, x:%d, y: %d \n", index, pixelX, pixelY);        
//     //     shadePixelForCircle(index, pixelCenterNorm, p, existingColor);
//     // }
//     // *imgPtr = existingColor;


//     // 1024 circles have been order for this box
    


//     // // printf("update  pixel x:%d, y: %d \n", pixelX, pixelY);
//     // float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
//     //                                     invHeight * (static_cast<float>(pixelY) + 0.5f));

//     // // each pixel works on the same first BLOCKSIZE circles. 
//     // // for 
//     // // each thread updates infromation about the circle
//     // if (linearThreadIndex < BLOCKSIZE)
//     //     circles[linearThreadIndex] = *(float3*)(&cuConstRendererParams.position[index3]);

//     // __syncthreads__();

//     // //
//     // // is this pixel in one of the circles?
//     // __shared__ uint prefixSumInput[BLOCKSIZE];
//     // __shared__ uint prefixSumOutput[BLOCKSIZE];
//     // __shared__ uint prefixSumScratch[2 * BLOCKSIZE];

//     // for (int i = start; i < std::min(cuConstRendererParams.numCircles, start + BLOCKSIZE); i++){
//     //     float rad = cuConstRendererParams.radius[circleIndex];;
//     //     prefixSumInput[i] = pixelInCircle(circles[linearThreadIndex], pixelCenter, rad * rad);
//     //     prefixSumOutput[i] = prefixSumInput[i];
//     // }

//     // __syncthreads__();

//     // // Exclusive scan must run within the same thread block data. One thread block corresponds to 
//     // // C circles for 1 pixel. We want 
//     // sharedMemExclusiveScan(linearThreadIndex, prefixSumInput, prefixSumOutput, prefixSumScratch, BLOCKSIZE);

//     // __syncthreads__();




//     // // printf("update circle at pixel circle %d, x:%d, y: %d \n", index, pixelX, pixelY);        
//     // shadePixelForCircle(index, pixelCenterNorm, p, existingColor);

// }


// void CudaRenderer::render3() {

//     int N = image->width * image->height;
//     int PixelPerBox = 10;
//     int numberOfBoxes = (N + 1) / PixelPerBox;

//     int CIRCLES_PER_BOX = std::min(1024, num_circles);
//     dim3 blockDim(CIRCLES_PER_BOX, 1);
//     dim3 gridDim( (numberOfBoxes * CIRCLES_PER_BOX  + blockDim.x - 1) / blockDim.x, 1);

//     // take up the first 1024 circles, update all pixels

//     for (int i=0; i < (num_circles + 1) / blockDim.x; i++){
//         // apply the effect of first N circles in increment of 1024 across all pixels
//         uint* output;
//         cudaMalloc(&output, sizeof(uint)* numberOfBoxes * CIRCLES_PER_BOX);

//         kernelRenderPixelsBlocks<<<gridSize, blockSize>>>(i * CIRCLES_PER_BOX,
//                                                          CIRCLES_PER_BOX, 
//                                                          output);

//         // for each blcok                                                  
//     }
    
//     cudaCheckError(cudaDeviceSynchronize())
// }

__global__ void kernelRenderPixelsWithCache(int imageWidth, int imageHeight, int circleStart, int circleEnd, int num_circles) {

    int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
    int pixelY = blockIdx.y * blockDim.y + threadIdx.y;

    if (pixelX > imageHeight  || pixelY > imageWidth )
        return;


    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;

    // printf("update  pixel x:%d, y: %d \n", pixelX, pixelY);

    __shared__ float3 circles_positions[1024];
    __shared__ float circles_radius[1024];

    int circle_index = circleStart + threadIdx.x;
    int index3 = 3 * circle_index;


    if (circle_index < num_circles){
        circles_positions[threadIdx.x] = *(float3*)(&cuConstRendererParams.position[index3]);
        circles_radius[threadIdx.x] = cuConstRendererParams.radius[circle_index];;     
    }

    __syncthreads();

    // if (pixelX == 16 && pixelY ==30){
    // // if (pixelX == 16){
    //     for (int i =0; i < (circleEnd - circleStart); i++){
    //         printf("-- circle_index %d - circles_positions: %f - radius %f\n", circle_index, circles_positions[i].x, circles_radius[i]);
    //     }
    // }

    
    // now each pixel should iterate over all circle in the shared data list and update the imagePtr values

    // printf("Apply effect of circles from %d to %d for pixel (%d,%d) \n", circleStart, circleEnd, pixelX, pixelY);

    // Each pixel can now apply shaders using 
    // Go through each circle and apply effect  
    float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
                                         invHeight * (static_cast<float>(pixelY) + 0.5f));

    float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixelY * imageWidth + pixelX)]);
    float4 existingColor = *imgPtr;

    // Go through each circle and apply effect
    for (int i = 0; i < (circleEnd - circleStart); i++){
        // printf("update circle at shared pixel circle %d, x:%d, y: %d \n", circle_id - circleStart, pixelX, pixelY);        
        shadePixelForCircle(circleStart + i, 
                            pixelCenterNorm, 
                            circles_positions[i], 
                            circles_radius[i], 
                            existingColor);
    }

    *imgPtr = existingColor;
}


__device__ __inline__ int current_box(int imageWidth, int imageHeight, int pixelX, int pixelY, int NUM_BOXES_PER_DIM){

    int pixel_per_box_x = imageWidth / NUM_BOXES_PER_DIM;
    int pixel_per_box_y = imageHeight / NUM_BOXES_PER_DIM;

    return (pixelY / pixel_per_box_y) * NUM_BOXES_PER_DIM + pixelX / pixel_per_box_x;
}

void CudaRenderer::renderPixelsIncrement() {
    // int max_circle_per_iterations = 1024;
    // int N = image->width * image->height;
    // dim3 blockDim(max_circle_per_iterations, 1);
    // dim3 gridDim( (N/max_circle_per_iterations  + blockDim.x - 1) / blockDim.x, 1);

    int max_circle_per_iterations = 32;
    dim3 blockSize(max_circle_per_iterations, max_circle_per_iterations);
    dim3 gridSize( (image->width + blockSize.x - 1) / blockSize.x, (image->height + blockSize.y - 1) / blockSize.y);

    printf("== Scene %d - numCircles %d\n", sceneName, numCircles );

    // take up the first 1024 circles, update all pixels
    int circleEnd= 0;
    for (int circleStart=0; circleStart < numCircles; circleStart+= max_circle_per_iterations){
        circleEnd = std::min(numCircles, circleStart + max_circle_per_iterations);

        // apply the effect of first N circles in increment of 1024 across all pixels
        // printf("Launch kernel for circles from %d to %d\n", circleStart, circleEnd);
        kernelRenderPixelsWithCache<<<gridSize, blockSize>>>(image->width, image->height, circleStart, circleEnd, numCircles);
        // for each blcok                                                  
    }
    
    cudaCheckError(cudaDeviceSynchronize())
}

__global__ void kernelRenderPixelsWithBox(int **circles_for_box, int *num_circles_for_box, int imageWidth, int imageHeight, int NUM_BOXES_PER_DIM) {

    int pixelX = blockIdx.x * blockDim.x + threadIdx.x; //col
    int pixelY = blockIdx.y * blockDim.y + threadIdx.y; // row

    if (pixelX > imageWidth || pixelY > imageHeight )
        return;

    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;

    // printf("update  pixel x:%d, y: %d \n", pixelX, pixelY);
    float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
                                        invHeight * (static_cast<float>(pixelY) + 0.5f));
    float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixelY * imageWidth + pixelX)]);
    float4 existingColor = *imgPtr;

    // Go through each circle and apply effect
    int curr_box = current_box(imageWidth, imageHeight, pixelX, pixelY, NUM_BOXES_PER_DIM);

    for (int i = 0; i < num_circles_for_box[curr_box]; i++){
        // Get the next circle in the list of circles for the current box
        int index = circles_for_box[curr_box][i];
        int index3 = 3 * index;

#ifdef PRINT_DEBUG            
        if (curr_box == -1){
            printf("box %d pixelX: %d pixelY: %d apply circle %d effect\n", curr_box, pixelX, pixelY, index);
        }
#endif        
        // read position and radius
        float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
        float rad = cuConstRendererParams.radius[index];;

        // printf("update circle at pixel circle %d, x:%d, y: %d \n", index, pixelX, pixelY);        
        shadePixelForCircle(index, pixelCenterNorm, p, rad, existingColor);
    }
    *imgPtr = existingColor;

}

/**
 * This kernel launches each circle, checks which box this circle intersects with and marks
 * that in the output `boxes` 
 */
__global__ void assignToBlock(int* boxes, int NUM_BOXES_PER_DIM, int PIXELS_IN_BOX_DIM, int numCircles){
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numCircles)
        return;

    int index3 = 3 * index;

    // read position and radius
    float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
    float radius = cuConstRendererParams.radius[index];
    short imageHeight = cuConstRendererParams.imageHeight;
    short imageWidth = cuConstRendererParams.imageWidth;

    // iterator over the boxes
    for (int i = 0; i< NUM_BOXES_PER_DIM * NUM_BOXES_PER_DIM; i++){
        // box i has what    
        int boxL = (i % NUM_BOXES_PER_DIM) * PIXELS_IN_BOX_DIM;
        int boxR = boxL + PIXELS_IN_BOX_DIM;
        int boxB = (i / NUM_BOXES_PER_DIM) * PIXELS_IN_BOX_DIM;
        int boxT = boxB + PIXELS_IN_BOX_DIM;
                                    
        int inside = circleInBoxConservative( p.x * imageWidth, p.y * imageHeight , radius* imageWidth,
                                 1.0 * boxL, 1.0 * boxR,  1.0 *boxT,  1.0 * boxB);
        // printf("circle %d - p.x %f, p.y: %f, rad %f - boxL:%d, boxR:%d, boxB: %d, boxT: %d - inside? %d\n", index,p.x * imageWidth, p.y* imageHeight, radius*imageWidth, boxL, boxR, boxB, boxT, inside);

        boxes[i * numCircles + index] = inside;
    }
}

#define BLOCKSIZE 32
#define SCAN_BLOCK_DIM   BLOCKSIZE  // needed by sharedMemExclusiveScan implementation
#include "exclusiveScan.cu_inl"


/**
 * SegmentScan the masks for each of the boxes. We can do this BLOCKSIZE
 * each block does 1 Box
 *  
 */
__global__ void prefixScanBox(int* boxes, int* prefixScan, int num_boxes, int numCircles){

    // int linearThreadIndex =  blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ uint prefixSumInput[BLOCKSIZE];
    __shared__ uint prefixSumOutput[BLOCKSIZE];
    __shared__ uint prefixSumScratch[2 * BLOCKSIZE];

    if (blockIdx.x >= num_boxes )
        return;

    // int* current_box = &boxes[blockIdx.x * numCircles];    
    int last_value = 0;
    // iterate in block size just and copy the values
    for (int i= 0; i< numCircles; i+= BLOCKSIZE ){
        __syncthreads();
        // pick the last value written at the previous iteration which is then used to add up the values from prefix scan
        // with this value in the nexst iteration
        if (i > 0)
            last_value = prefixScan[blockIdx.x * numCircles + i - 1];

#ifdef PRINT_DEBUG     
        int boxid = 32;       
        if (blockIdx.x == boxid && i + threadIdx.x < numCircles)
            printf("box:%d - circle %d - input %d - iter %d\n", blockIdx.x, threadIdx.x, boxes[blockIdx.x * numCircles + (i + threadIdx.x)], i);
#endif

        if (i + threadIdx.x < numCircles)
            prefixSumInput[threadIdx.x] = boxes[blockIdx.x * numCircles + (i + threadIdx.x)];
        else
            prefixSumInput[threadIdx.x] = 0;

        __syncthreads();

#ifdef PRINT_DEBUG            
        if (blockIdx.x == boxid && i + threadIdx.x < numCircles)
            printf("prefixSumInput for box:%d - circle %d - input %d iter %d\n", blockIdx.x, threadIdx.x, prefixSumInput[threadIdx.x], i);
#endif
        // printf("Box: %d - iteration %d - threadid: %d\n", blockIdx.x, i, threadIdx.x);

        sharedMemExclusiveScan(threadIdx.x, prefixSumInput, prefixSumOutput, prefixSumScratch, BLOCKSIZE);

        __syncthreads();
        // copy prefixSumOutput out
        if (i + threadIdx.x < numCircles){

            if (threadIdx.x == BLOCKSIZE -1)
                prefixScan[blockIdx.x * numCircles + i + threadIdx.x] = last_value + prefixSumOutput[threadIdx.x] + boxes[blockIdx.x * numCircles + (i + threadIdx.x)];
            else
                prefixScan[blockIdx.x * numCircles + i + threadIdx.x] = last_value + prefixSumOutput[threadIdx.x + 1];
        }
#ifdef PRINT_DEBUG            
        if (blockIdx.x == boxid && i + threadIdx.x < numCircles)
            printf("box:%d - circle %d - output %d iter %d \n", blockIdx.x, threadIdx.x, prefixSumOutput[threadIdx.x], i);
#endif

    }

}

__global__ void val(int* input, int index, int *out){
    int i = blockIdx.x  * blockDim.x  + threadIdx.x;

    if (i == 0){
        out[0] = input[index];
    }
}

__global__ void getMax(int* prefix_scan_boxes_device, int* num_circles_for_box_device, int numCircles, int num_boxes){
    // 1 thread for each Box 
    int i = blockIdx.x  * blockDim.x  + threadIdx.x;

    if (i > num_boxes)
        return;

    num_circles_for_box_device[i] = prefix_scan_boxes_device[(i + 1) * numCircles - 1];

}

__global__ void getCircleListForEachBox(int* boxes_device_mask, int* prefix_scan_boxes_device, int** circles_for_box_out, int* num_circles_for_box, int num_boxes, int nunCircles, int N){
    // this kernel gather all circles to apply for each box in sequential order

    int index = blockIdx.x  * blockDim.x  + threadIdx.x;

    if (index > N)
        return;    

    int box = index % num_boxes;
    int circle = index / num_boxes;

#ifdef PRINT_DEBUG    
    // printf("getCircleListForEachBox() - box %d, circle %d, - is set: %d store at: %d\n", box, circle, boxes_device_mask[box * nunCircles + circle], prefix_scan_boxes_device[box * nunCircles + circle] - 1);
#endif

    if (boxes_device_mask[box * nunCircles + circle] == 1){
        // printf("%d\n", prefix_scan_boxes_device[box * nunCircles + circle] - 1);
        circles_for_box_out[box][prefix_scan_boxes_device[box * nunCircles + circle] - 1] = circle;
        
        // printf("box %d: circle %d at %d\n", box, circles_for_box_out[box][prefix_scan_boxes_device[box * nunCircles + circle] - 1], prefix_scan_boxes_device[box * nunCircles + circle] - 1);

    }

}


void CudaRenderer::render() {
    /**
     * divide the image into small boxes
     *  num_box division in each dimension
     *  -> there num_box ^2 boxes
     * 2D kernel gives us a natual box division with max of 32 pixel per block
     *  
     */ 
    int THREADS_PER_BLOCK = 1024;
    
    // assume same width and heigh dimenseion
    int NUM_BOXES_PER_DIM = 8;
    int NUM_BOXES = NUM_BOXES_PER_DIM * NUM_BOXES_PER_DIM;
    int PIXELS_IN_BOX = image->width * image->height / NUM_BOXES;
    int PIXELS_IN_BOX_DIM = std::sqrt(PIXELS_IN_BOX);

    dim3 blockSize(THREADS_PER_BLOCK, 1);
    dim3 gridSize((numCircles + blockSize.x - 1) / blockSize.x, 1);

    int* boxes_device_mask;
    int* prefix_scan_boxes_device;
    cudaCheckError(cudaMalloc(&boxes_device_mask, sizeof(int) * NUM_BOXES * numCircles))
    cudaCheckError(cudaMalloc(&prefix_scan_boxes_device, sizeof(int) * NUM_BOXES * numCircles))

    printf("NumCircles %d - Num Boxes %d, Pixels in Box: %d\n", numCircles, NUM_BOXES, PIXELS_IN_BOX);

    assignToBlock<<<gridSize, blockSize>>>(boxes_device_mask, NUM_BOXES_PER_DIM, PIXELS_IN_BOX_DIM, numCircles);
    cudaCheckError(cudaDeviceSynchronize())

    dim3 blockSizeScan(SCAN_BLOCK_DIM, 1);
    dim3 gridSizeScan(NUM_BOXES, 1);

    prefixScanBox<<<gridSizeScan,blockSizeScan>>>(boxes_device_mask, prefix_scan_boxes_device, NUM_BOXES, numCircles);
    cudaCheckError(cudaDeviceSynchronize())

    // Get max number of circles for each box, allocate memory and copy index of circles
    int *num_circles_for_box = new int[NUM_BOXES];
    int *num_circles_for_box_device;
    cudaCheckError(cudaMalloc(&num_circles_for_box_device, sizeof(int) * NUM_BOXES))

    dim3 blockSizeMax(32, 1);
    dim3 gridSizeMax((NUM_BOXES + blockSizeMax.x - 1) / blockSizeMax.x, 1);

    getMax<<<gridSizeMax, blockSizeMax>>>(prefix_scan_boxes_device, num_circles_for_box_device, numCircles, NUM_BOXES);
    cudaCheckError(cudaDeviceSynchronize())

    cudaCheckError(cudaMemcpy(num_circles_for_box, num_circles_for_box_device, NUM_BOXES * sizeof(int), cudaMemcpyDeviceToHost))

#ifdef PRINT_DEBUG
    for(int i =0; i< NUM_BOXES; i++){
        printf("Box %d has %d circles intersecting\n", i, num_circles_for_box[i]);
    }
#endif

    // Create the actual list of circles for each box
    int total_circles = 0;
    for(int i =0; i< NUM_BOXES; i++){
       total_circles += num_circles_for_box[i];
    }

    // int** circles_for_box = new int*[NUM_BOXES];
    int** circles_for_box;
    cudaCheckError(cudaMalloc((void**)&circles_for_box, NUM_BOXES * sizeof(int*)))

    int** h_ptr_array = new int*[NUM_BOXES]; // create array of points on the host side

    for (int i = 0; i < NUM_BOXES; ++i) {
        cudaCheckError(cudaMalloc((void**)&(h_ptr_array[i]), num_circles_for_box[i] * sizeof(int)))
    }

    // Copy the host-side array of device pointers to the device
    cudaCheckError(cudaMemcpy(circles_for_box, h_ptr_array, NUM_BOXES * sizeof(int*), cudaMemcpyHostToDevice))
    
    // for(int i =0; i< NUM_BOXES; i++){
    //     cudaCheckError(cudaMalloc((int**)&temp_ptr, num_circles_for_box[i] * sizeof(int)))
    //     cudaCheckError(cudaMemcpy(&circles_for_box[i], &temp_ptr, sizeof(int*), cudaMemcpyHostToDevice))
    //     // cudaCheckError(cudaMalloc(&circles_for_box[i], sizeof(int) * num_circles_for_box[i]))
    //     total_circles += num_circles_for_box[i];
    // }

    // cudaCheckError(cudaMalloc(&circles_for_box, sizeof(int*) * total_circles))


    // pb: to index into a specific box, we need to know indexing of all previous box

    dim3 blockSizeGetCircles(32, 1);
    dim3 gridSizeGetCircles((NUM_BOXES * numCircles + blockSizeGetCircles.x - 1) / blockSizeGetCircles.x, 1);    
    getCircleListForEachBox<<<gridSizeGetCircles, blockSizeGetCircles>>>(boxes_device_mask, 
                                                                         prefix_scan_boxes_device,
                                                                         circles_for_box, 
                                                                         num_circles_for_box_device,
                                                                         NUM_BOXES,
                                                                         numCircles,
                                                                         NUM_BOXES * numCircles);
    cudaCheckError(cudaDeviceSynchronize())
    
    printf("Done with circle list!\n");
    
#ifdef PRINT_DEBUG
    int boxid = 42;
    int *box1Circles = new int[num_circles_for_box[boxid]];
    cudaCheckError(cudaMemcpy(box1Circles, h_ptr_array[boxid], num_circles_for_box[boxid] * sizeof(int), cudaMemcpyDeviceToHost))
    printf("==Box %d circle list: ", boxid);
    for (int i=0 ; i< num_circles_for_box[boxid]; i++){
        printf("%d,", box1Circles[i]);
    }
    printf("\n");
    delete [] box1Circles;
#endif

    // Now that we have the list, launch the kernel for each pixel together with the list of circles for each box
    dim3 blockSizeFinal(32, 32);
    dim3 gridSizeFinal( (image->width + blockSizeFinal.x - 1) / blockSizeFinal.x, (image->height + blockSizeFinal.y - 1) / blockSizeFinal.y);

    kernelRenderPixelsWithBox<<<gridSizeFinal, blockSizeFinal>>>(circles_for_box, num_circles_for_box_device, image->width, image->height, NUM_BOXES_PER_DIM);
    cudaCheckError(cudaDeviceSynchronize())

#ifdef PRINT_DEBUG
    // check boxes results
    int *boxes = new int[NUM_BOXES * numCircles];
    int *prefix_scan_boxes = new int[NUM_BOXES * numCircles];
    cudaCheckError(cudaMemcpy(boxes, boxes_device_mask, NUM_BOXES * numCircles * sizeof(int), cudaMemcpyDeviceToHost))
    cudaCheckError(cudaMemcpy(prefix_scan_boxes, prefix_scan_boxes_device, NUM_BOXES * numCircles * sizeof(int), cudaMemcpyDeviceToHost))

    int num_circles_ = 3;
    if (true){
        for (int i = 0; i< NUM_BOXES; i++){
            // box i has what
            printf("==Box  %d: ", i);
            for (int c = 0; c < num_circles_; c++){
                printf("%d,", boxes[i * numCircles + c]);            
                // printf("%d,", c);

        }        
        printf("\n");
        printf("->Scan %d: ", i);
            for (int c = 0; c < num_circles_; c++){
                printf("%d,", prefix_scan_boxes[i * numCircles + c]);            
                // printf("%d,", c);

        }        
        printf("\n");
        }
    }
    delete [] boxes;
    delete [] prefix_scan_boxes;
#endif
    
    printf("Done! free ressources\n");

    cudaFree(boxes_device_mask);
    cudaFree(prefix_scan_boxes_device);

    cudaFree(boxes_device_mask);
    cudaFree(num_circles_for_box_device);


    for(int i =0; i< NUM_BOXES; i++){
        cudaFree(h_ptr_array[i]);
    }

    cudaFree(circles_for_box);
    delete [] h_ptr_array;
  
}

/**
 * 
 * 1. assign circles to box
    -> each box has a list of circles preserved in order
   2. run per pixel kernel, idenityf which box it's in and which circles to apply
      .circle order has to be preserved
 */