#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include "CycleTimer.h"

void saxpyCuda(int N, float alpha, float* x, float* y, float* result);
void pinnedMem(float **ptr, int N);
void freePinned(float* ptr);


void printCudaInfo();


void usage(const char* progname) {
    printf("Usage: %s [options]\n", progname);
    printf("Program Options:\n");
    printf("  -n  --arraysize <INT>  Number of elements in arrays\n");
    printf("  -?  --help             This message\n");
}

static float toBW(int bytes, float sec) {
    return static_cast<float>(bytes) / (1024. * 1024. * 1024.) / sec;
}

static float toGFLOPS(int ops, float sec) {
    return static_cast<float>(ops) / 1e9 / sec;
}

void saxpySerial(int N,
                       float scale,
                       float X[],
                       float Y[],
                       float result[])
{

    for (int i=0; i<N; i++) {
        result[i] = scale * X[i] + Y[i];
    }
}

void runPinned(int N, const float alpha){
  printf("--- Pinned Test \n");

  float *xarray = NULL;
  float *yarray = NULL;
  float *resultarray = NULL;

  pinnedMem(&xarray, N);
  pinnedMem(&yarray, N);
  pinnedMem(&resultarray, N);

  for (int i=0; i<N; i++) {
        xarray[i] = yarray[i] = i % 10;
        resultarray[i] = 0.f;
  }


  printf("Running 3 timing tests:\n");
  for (int i=0; i<3; i++) {
    saxpyCuda(N, alpha, xarray, yarray, resultarray);
  }

  double minSerial = 1e30;
  for (int i=0; i<3; i++) {
    double startTime =CycleTimer::currentSeconds();
    saxpySerial(N, alpha, xarray, yarray, resultarray);
    double endTime = CycleTimer::currentSeconds();
    minSerial = std::min(minSerial, endTime - startTime);
  }

  const unsigned int TOTAL_BYTES = 4 * N * sizeof(float); // cj: 2 loads and 1 write + cache eviction
  const unsigned int TOTAL_FLOPS = 2 * N; // cj: 1 addition + 1 multiplication for N elements

  printf("[saxpy serial]:\t\t[%.3f] ms\t[%.3f] GB/s\t[%.3f] GFLOPS\n",
        minSerial * 1000,
        toBW(TOTAL_BYTES, minSerial),
        toGFLOPS(TOTAL_FLOPS, minSerial));

  freePinned(xarray);
  freePinned(yarray);
  freePinned(resultarray);
}

void runNotPinned(int N, const float alpha){
  printf("--- Not Pinned Test \n");

  float* xarray = new float[N];
  float* yarray = new float[N];
  float* resultarray = new float[N];

  for (int i=0; i<N; i++) {
        xarray[i] = yarray[i] = i % 10;
        resultarray[i] = 0.f;
  }


  printf("Running 3 timing tests:\n");
  for (int i=0; i<3; i++) {
    saxpyCuda(N, alpha, xarray, yarray, resultarray);
  }

  double minSerial = 1e30;
  for (int i=0; i<3; i++) {
    double startTime =CycleTimer::currentSeconds();
    saxpySerial(N, alpha, xarray, yarray, resultarray);
    double endTime = CycleTimer::currentSeconds();
    minSerial = std::min(minSerial, endTime - startTime);
  }

  const unsigned int TOTAL_BYTES = 4 * N * sizeof(float); // cj: 2 loads and 1 write + cache eviction
  const unsigned int TOTAL_FLOPS = 2 * N; // cj: 1 addition + 1 multiplication for N elements

  printf("[saxpy serial]:\t\t[%.3f] ms\t[%.3f] GB/s\t[%.3f] GFLOPS\n",
        minSerial * 1000,
        toBW(TOTAL_BYTES, minSerial),
        toGFLOPS(TOTAL_FLOPS, minSerial));


  delete [] xarray;
  delete [] yarray;
  delete [] resultarray;
}

int main(int argc, char** argv)
{

    // default: arrays of 100M numbers
    int N = 100 * 1000 * 1000;

    // parse commandline options ////////////////////////////////////////////
    int opt;
    static struct option long_options[] = {
        {"arraysize",  1, 0, 'n'},
        {"help",       0, 0, '?'},
        {0 ,0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "?n:", long_options, NULL)) != EOF) {

        switch (opt) {
        case 'n':
            N = atoi(optarg);
            break;
        case '?':
        default:
            usage(argv[0]);
            return 1;
        }
    }
    // end parsing of commandline options //////////////////////////////////////

    const float alpha = 2.0f;

    printCudaInfo();

    runNotPinned(N, alpha);

    runPinned(N, alpha);

    return 0;
}
