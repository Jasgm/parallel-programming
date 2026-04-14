/*
Collatz code

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
#include <algorithm>
#include <sys/time.h>



static int length(long long val)
{
  if (val <= 0) return 0;

  // compute Collatz sequence length
  int len = 1;
  while (val != 1) {
    len++;
    if ((val % 2) != 0) {
      val = 3 * val + 1;  // odd
    } else {
      val = val / 2;  // even
    }
  }

  return len;
}


static int collatz1(const long long stop)
{
  int maxlen = 0;
  for (long long i = 0; i < stop; i++) {
    const int len = length(i);
    maxlen = std::max(maxlen, len);
  }
  return maxlen;
}


static int collatz2(const long long start, const long long stop)
{
  int maxlen = 0;
  for (long long i = start; i < stop; i++) {
    const int len = length(i);
    maxlen = std::max(maxlen, len);
  }
  return maxlen;
}


static int collatz3(const long long stop, const long long incr)
{
  int maxlen = 0;
  for (long long i = 0; i < stop; i += incr) {
    const int len = length(i);
    maxlen = std::max(maxlen, len);
  }
  return maxlen;
}


static int collatz4(const long long start, const long long stop, const long long incr)
{

  int maxlen = 0;
#pragma omp parallel for default(none) shared(start, stop, incr) \
reduction(max:maxlen) schedule(runtime)
  for (long long i = start; i < stop; i += incr) {
    const int len = length(i);
    maxlen = std::max(maxlen, len);
  }
  return maxlen;
}


int main(int argc, char* argv [])
{
  printf("Collatz v2.1\n");

  // check command line
  if (argc != 5) {fprintf(stderr, "USAGE: %s version start_value stop_value increment\n", argv[0]); exit(-1);}
  const int version = atoi(argv[1]);
  const long long start = atoll(argv[2]);
  const long long stop = atoll(argv[3]);
  const long long incr = atoll(argv[4]);
  if ((version < 1) || (version > 4)) {fprintf(stderr, "ERROR: version must be between 1 and 4\n"); exit(-1);}
  printf("version: %d\n", version);
  printf("start value: %lld\n", start);
  printf("stop value: %lld\n", stop);
  printf("increment: %lld\n", incr);

  // start time
  timeval beg, end;
  gettimeofday(&beg, NULL);

  // execute timed code
  int maxlen;
  switch (version) {
    case 1: maxlen = collatz1(stop); break;
    case 2: maxlen = collatz2(start, stop); break;
    case 3: maxlen = collatz3(stop, incr); break;
    case 4: maxlen = collatz4(start, stop, incr); break;
  }

  // stop time
  gettimeofday(&end, NULL);
  const double runtime = end.tv_sec - beg.tv_sec + (end.tv_usec - beg.tv_usec) / 1000000.0;
  printf("compute time: %.6f s\n", runtime);

  // print result
  printf("maximum sequence length: %d\n", maxlen);
  return 0;
}
