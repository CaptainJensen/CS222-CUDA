
int main(){

    // Define variables
    int *U, *V;
    int *dev_U, *dev_U;
    int *partialSum;

    // print info about the system
    int count;
    cudaGetDeviceCount( &count );
    printf("there are %d device(s)\n", count);
    for (int i=0; i<count; ++i) {
        cudaGetDeviceProperties( &prop, i );
        printf("name is %s\n", prop.name);
        printf("warp size is %d\n", prop.warpSize);
        printf("maxThreadsPerBlock is %d\n", prop.maxThreadsPerBlock);
        printf("maxThreadsDim is (%d, %d, %d)\n", prop.maxThreadsDim[0],
               prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("maxGridSize is (%d, %d, %d)\n", prop.maxGridSize[0],
               prop.maxGridSize[1], prop.maxGridSize[2]);
    }


    U = (float *) malloc(N * sizeof(float));
    V = (float *) malloc(N * sizeof(float));

    int threadsPerBlock = 256;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // allocate memory on the GPU
    HANDLE_ERROR( cudaMalloc( (void **) &dev_U, N*sizeof(int) ) );
    HANDLE_ERROR( cudaMalloc( (void **) &dev_V, N*sizeof(int) ) );

    partialSum = (float *) malloc(numBlocks* sizeof(float));

    for (int i=0; i<N; ++i) {
        U[i] = (float) (i+1);
        V[i] = 1.0 / U[i];
    }

    HandleError( cudaMemcpy( dev_U, U, N*sizeof(float), cudaMemcpyHostToDevice) );
    HandleError( cudaMemcpy( dev_V, V, N*sizeof(float), cudaMemcpyHostToDevice) );
    dotp<<<numBlocks, threadsPerBlock, blockSize* sizeof(float)>>>( dev_U, dev_V, dev_partialSum, N );
    cudaDeviceSynchronize(); // wait for GPU threads to complete; again, not necessary but good pratice
    HandleError( cudaMemcpy( partialSum, dev_partialSum, numBlocks*sizeof(float), cudaMemcpyDeviceToHost) );

    // finish up on the CPU side
    float gpuResult= 0.0;
    for (int i=0; i<numBlocks; ++i) gpuResult= gpuResult+ partialSum[i];


    cudaFree( dev_U );
    cudaFree( dev_V );
    free(U);
    free(V);

    printf("%s\n", gpuResult);
    printf("%s\n", partialSum);
    printf("%s\n", partialSumarray);
    print("DONE");
    // END OF PROGRAM
}

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}

__global__ void dotp( float *U, float *V, float *partialSum, int N ) {
    //__shared__ float localCache[BLOCK_SIZE];
    extern __shared__ float localCache[];
    int tidx= threadIdx.x; // my position in my threadblock
    localCache[tidx] = U[tidx] * V[tidx];
    __syncthreads();

    // now, we need to add up the values in localCache[]
    if (threadIdx.x== 0) {
        float temp = 0.0;
        for (int i=0; i<blockDim.x; ++i) temp = temp + localCache[i];localCache[0] = temp;
    }
    // now put the result (this thread block's partial sum) in the partialSumarray
    cacheIndex= threadIdx.x;
    int i= blockDim.x/2;
    while (i> 0) {
        if (cacheIndex< i) localCache[cacheIndex] = localCache[cacheIndex] + localCache[cacheIndex+ i];
        __syncthreads();
        i= i/ 2;
    }

    if (cacheIndex== 0) partialSum[blockIdx.x] = localCache[cacheIdx];
}
