/*
Maximal independent set code

Copyright 2025 Martin Burtscher
Copyright 2025 Jasmine Gay

Redistribution in source or binary form, with or without modification, is not
permitted. Use in source or binary form, with or without modification, is only
permitted for academic use in CS 4380 and CS 5351 at Texas State University.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#include <cstdlib>
#include <cstdio>
#include <sys/time.h>
#include "ECLgraph.h"
#include <cuda.h>


static const unsigned char undecided = 0;
static const unsigned char incl = 1;
static const unsigned char excl = 2;
static const int ThreadsPerBlock = 512;


// https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key

__device__ unsigned int hash(unsigned int val)
{
  val = ((val >> 16) ^ val) * 0x45D9F3B;
  val = ((val >> 16) ^ val) * 0x45D9F3B;
  return (val >> 16) ^ val;
}


static __global__ void init(const int nodes, unsigned int* const prio, unsigned char* const stat)
{
  // initialize arrays
  const int v = threadIdx.x + blockIdx.x * blockDim.x;
  if(v < nodes){
    stat[v] = undecided;
    prio[v] = hash(v + 712818237);
  }
}


__device__ bool lowerprio(const int v, const int n, const unsigned int* const prio)
{
  const int pv = prio[v];
  const int pn = prio[n];
  return (pv < pn) || ((pv == pn) && (v < n));
}


static __global__ void mis(const ECLgraph g, const unsigned int* const prio, volatile unsigned char* const stat, volatile bool* const goagain)
{
  // go over all nodes
  const int v = threadIdx.x + blockIdx.x * blockDim.x;
  if(v < g.nodes) {
    if (stat[v] == undecided) {
      const int beg = g.nindex[v];  // beginning of adjacency list
      const int end = g.nindex[v + 1];  // end of adjacency list
      int i = beg;
      // try to find a neighbor whose priority is higher
      while (i < end) {
        const int n = g.nlist[i];  // neighbor
        if ((stat[n] != excl) && lowerprio(v, n, prio)) break;
        i++;
      }
      if (i < end) {
        // found such a neighbor -> status of v still unknown
        *goagain = true;
      } else {
        // no such neighbor -> exclude all neighbors and include v
        stat[v] = incl;
        for (int i = beg; i < end; i++) {
          const int n = g.nlist[i];  // neighbor
          stat[n] = excl;
        }
      }
    }
  }
}

static void CheckCuda(const int line)
{
  cudaError_t e;
  cudaDeviceSynchronize();
  if (cudaSuccess != (e = cudaGetLastError())) {
    fprintf(stderr, "CUDA error %d on line %d: %s\n", e, line, cudaGetErrorString(e));
    exit(-1);
  }
}

int main(int argc, char* argv [])
{
  printf("Maximal independent set v2.2\n");

  // check command line
  if (argc != 2) {fprintf(stderr, "USAGE: %s input_graph\n", argv[0]); exit(-1);}

  // read input
  ECLgraph g = readECLgraph(argv[1]);
  printf("input: %s\n", argv[1]);
  printf("nodes: %d\n", g.nodes);
  printf("edges: %d\n", g.edges);

  // allocate result vector on CPU
  unsigned char* const stat = new unsigned char [g.nodes];

  //allocate data on GPU
  unsigned char* d_stat;
  unsigned int* d_prio;
  bool* d_goagain;
  ECLgraph d_g = g;
  
  cudaMalloc((void **)&d_g.nindex, sizeof(int) * (g.nodes+1));
  cudaMalloc((void **)&d_g.nlist, sizeof(int) * g.edges);
  cudaMalloc((void **)&d_goagain, sizeof(bool));
  cudaMalloc((void **)&d_stat, sizeof(unsigned char) * g.nodes);
  cudaMalloc((void **)&d_prio, sizeof(unsigned int) * g.nodes);
  CheckCuda(__LINE__);

  cudaMemcpy(d_g.nindex, g.nindex, sizeof(int) * (g.nodes+1), cudaMemcpyHostToDevice);
  cudaMemcpy(d_g.nlist, g.nlist, sizeof(int) * g.edges, cudaMemcpyHostToDevice);
  CheckCuda(__LINE__);

  // start time
  timeval beg, end;
  gettimeofday(&beg, NULL);

  // execute timed code
  // initialize d_g, prio, stat on GPU
  init<<<(g.nodes + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(d_g.nodes, d_prio, d_stat);
  bool goagain;
  do {  // repeat until all nodes' statuses have been decided
    goagain = false;
    cudaMemcpy(d_goagain, &goagain, sizeof(bool), cudaMemcpyHostToDevice);
    mis<<<(g.nodes + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(d_g, d_prio, d_stat, d_goagain);
    cudaMemcpy(&goagain, d_goagain, sizeof(bool), cudaMemcpyDeviceToHost);
  } while (goagain);

  // stop time
  gettimeofday(&end, NULL);
  CheckCuda(__LINE__);
  const double runtime = end.tv_sec - beg.tv_sec + (end.tv_usec - beg.tv_usec) / 1000000.0;
  printf("compute time: %.6f s\n", runtime);

  // copy stat results from GPU
  cudaMemcpy(stat, d_stat, sizeof(unsigned char) * g.nodes, cudaMemcpyDeviceToHost);
  CheckCuda(__LINE__);
  // determine and print set size
  int cnt = 0;
  for (int v = 0; v < g.nodes; v++) {
    if (stat[v] == incl) cnt++;
  }
  printf("MIS size: %d (%.1f%%)\n", cnt, 100.0 * cnt / g.nodes);

  // verify result
  for (int v = 0; v < g.nodes; v++) {
    if ((stat[v] != incl) && (stat[v] != excl)) {fprintf(stderr, "ERROR: found undecided node\n"); exit(-1);}
    if (stat[v] == incl) {
      for (int i = g.nindex[v]; i < g.nindex[v + 1]; i++) {
        if (stat[g.nlist[i]] == incl) {fprintf(stderr, "ERROR: found adjacent nodes in MIS\n"); exit(-1);}
      }
    } else {
      bool flag = true;
      for (int i = g.nindex[v]; i < g.nindex[v + 1]; i++) {
        if (stat[g.nlist[i]] == incl) {
          flag = false;
          break;
        }
      }
      if (flag) {fprintf(stderr, "ERROR: set is not maximal\n"); exit(-1);}
    }
  }
  printf("verification passed\n");

  // clean up
  cudaFree(d_g.nindex);
  cudaFree(d_g.nlist);
  cudaFree(d_prio);
  cudaFree(d_stat);
  cudaFree(d_goagain);
  freeECLgraph(g);
  delete [] stat;
  return 0;
}
