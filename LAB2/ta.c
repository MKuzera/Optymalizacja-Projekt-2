#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
int main()
{
  int P=4;
  int N=9;
  int M=2;
  int i;
  int j;
omp_set_num_threads(P);
  #pragma omp parallel for schedule(static,2) private(j)
  for (i=0; i<N; i++)
  {
    for (j=0; j<M; j++)
    {
      printf("i=%d, j=%d, thread=%d\n", i, j, omp_get_thread_num());
    }
  }
  return 0;
}

