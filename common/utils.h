#include <helper_cuda.h>

// GPU Time Calculation
#define START_GPU {\
cudaEvent_t     start, stop;\
float   elapsedTime;\
checkCudaErrors(cudaEventCreate(&start)); \
checkCudaErrors(cudaEventCreate(&stop));\
checkCudaErrors(cudaEventRecord(start, 0));\

#define END_GPU(taskName) \
checkCudaErrors(cudaEventRecord(stop, 0));\
checkCudaErrors(cudaEventSynchronize(stop));\
checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop)); \
printf("%s takes:  %3.1f ms\n",taskName, elapsedTime);\
checkCudaErrors(cudaEventDestroy(start));\
checkCudaErrors(cudaEventDestroy(stop));}