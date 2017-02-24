#include <limits>

using namespace std;

// Constantes
const double MENOS_INFINITO = -numeric_limits<double>::max(); // 负无穷
const size_t BLOCK_SIZE = 128;

__device__ unsigned int contadorBloques = 0; // counterBlocks

__device__ double logaritmoDeterminante(double *g_L, const size_t k, const size_t numDimensiones) // logarithmDeterminant 协方差矩阵Sigma行列式对数
{
    double suma = 0.0;

    for (size_t j = 0; j < numDimensiones; j++) {
        suma += log(g_L[k * numDimensiones * numDimensiones + j * numDimensiones + j]); // 对角线上元素的对数求和
    }

    return 2.0 * suma;
}

template <size_t blockSize>
__device__ void reducirBloque(volatile double *sharedData, double suma, const size_t tid) // 使用循环展开(loop unrolling)对一个BLOCK中的元素进行规约
{
    sharedData[tid] = suma;

    __syncthreads();

    if (blockSize >= 512) {
        if (tid < 256) {
            sharedData[tid] = suma = suma + sharedData[tid + 256];
        }

        __syncthreads();
    }

    if (blockSize >= 256) {
        if (tid < 128) {
            sharedData[tid] = suma = suma + sharedData[tid + 128];
        }

        __syncthreads();
    }

    if (blockSize >= 128) {
        if (tid < 64) {
            sharedData[tid] = suma = suma + sharedData[tid + 64];
        }

        __syncthreads();
    }

    if (tid < 32) {
        if (blockSize >= 64) {
            sharedData[tid] = suma = suma + sharedData[tid + 32];
        }

        if (blockSize >= 32) {
            sharedData[tid] = suma = suma + sharedData[tid + 16];
        }

        if (blockSize >= 16) {
            sharedData[tid] = suma = suma + sharedData[tid + 8];
        }

        if (blockSize >= 8) {
            sharedData[tid] = suma = suma + sharedData[tid + 4];
        }

        if (blockSize >= 4) {
            sharedData[tid] = suma = suma + sharedData[tid + 2];
        }

        if (blockSize >= 2) {
            sharedData[tid] = suma = suma + sharedData[tid + 1];
        }
    }
}

template <size_t blockSize, typename Predicate, typename Predicate2>
__device__ void reducirFinal(Predicate valor, Predicate2 direccionResultado, volatile double *sharedData, size_t numTrozos) // 使用一个BLOCK进行最终的规约
{
    const size_t tid = threadIdx.x;
    double suma = 0.0;
    int i = tid;

    while (i < numTrozos)
    {
        suma += *(valor(i));
        i += blockSize;
    }

    reducirBloque<blockSize>(sharedData, suma, tid);

    if (tid == 0) {
        *(direccionResultado()) = sharedData[0];
    }
}

template <size_t blockSize, typename Predicate, typename Predicate2, typename Predicate3>
__device__ void reducir(Predicate valor, Predicate2 direccionResultado, Predicate3 reduccionFinal, const size_t n, volatile double *sharedData, const size_t numBloques) // valor=value, direccionResultado=directionResult, reduccionFinal=finalReduction
{
    __shared__ bool esUltimoBloque; // is the last block

    const size_t tid = threadIdx.x; // 当前thread在BLOCK中的索引
    const size_t gridSize = (blockSize * 2) * gridDim.x; // 所有BLOCK一次最多能处理的元素的个数

    size_t i = blockIdx.x * (blockSize * 2) + threadIdx.x; // 当前thread在整体中的索引
    double suma = 0.0;

    while (i < n) { // 执行multiple adds/thread n是待处理的元素的总数
        suma += valor(i);

        if (i + blockSize < n) {
            suma += valor(i+blockSize);
        }

        i += gridSize;
    }

    reducirBloque<blockSize>(sharedData, suma, tid); // 使用循环展开(loop unrolling)对一个BLOCK中的元素进行规约

    if (tid == 0) {
    	*(direccionResultado()) = sharedData[0]; // 将共享内存中存储的部分规约值写入全局内存

        __threadfence(); // 暂停该thread直到其写入可以被grid中其他所有thread看到

        unsigned int ticket = atomicInc(&contadorBloques, numBloques);
        esUltimoBloque = (ticket == numBloques - 1);
    }

    __syncthreads();

    if (esUltimoBloque) {
        reduccionFinal(); // 使用一个BLOCK进行最终的规约

        if (tid == 0) {
            contadorBloques = 0; // 将计数重新置为0
        }
    }
}
