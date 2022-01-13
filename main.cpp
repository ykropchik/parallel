#include <thread>
#include <mutex>
#include <vector>
#include <iostream>
#include <omp.h>
#include <cstring>
#include "reduction.cpp"

#define STEPS 1000000000
#define CACHE_LINE 64u
#define A -1
#define B 1

typedef double (*f_t)(double);
typedef double (*E_t)(double, double, f_t);
typedef double (*r_t) (unsigned, unsigned*, size_t, unsigned, unsigned);

struct ExperimentResult {
    double result;
    double time;
};
struct partialSumT {
    alignas(64) double val;
};

typedef double (*I_t)(double, double, f_t);

using namespace std;

double f(double x) {
    return x * x;
}

double integrateDefault(double a, double b, f_t f) {
    double result = 0, dx = (b - a) / STEPS;

    for (unsigned i = 0; i < STEPS; i++) {
        result += f(i * dx + a);
    }

    return result * dx;
}

double integrateCrit(double a, double b, f_t f) {
    double result = 0, dx = (b - a) / STEPS;

#pragma omp parallel
    {
        double R = 0;
        unsigned t = (unsigned) omp_get_thread_num();
        unsigned T = (unsigned) get_num_threads();

        for (unsigned i = t; i < STEPS; i += T) {
            R += f(i * dx + a);
        }
#pragma omp critical
        result += R;
    }
    return result * dx;
}

double integrateMutex(double a, double b, f_t f) {
    unsigned T = get_num_threads();
    mutex mtx;
    vector<thread> threads;
    double result = 0, dx = (b - a) / STEPS;

    for (unsigned t = 0; t < T; t++) {
        threads.emplace_back([=, &result, &mtx]() {
            double R = 0;
            for (unsigned i = t; i < STEPS; i += T) {
                R += f(i * dx + a);
            }

            {
                scoped_lock lck{mtx};
                result += R;
            }
        });

    }
    for (auto &thr: threads) {
        thr.join();
    }

    return result * dx;
}

double integrateArr(double a, double b, f_t f) {
    unsigned T;
    double result = 0, dx = (b - a) / STEPS;
    double *accum;

#pragma omp parallel shared(accum, T)
    {
        unsigned t = (unsigned) omp_get_thread_num();
#pragma omp single
        {
            T = (unsigned) get_num_threads();
            accum = (double *) calloc(T, sizeof(double));
            //accum1.reserve(T);
        }

        for (unsigned i = t; i < STEPS; i += T) {
            accum[t] += f(dx * i + a);
        }
    }

    for (unsigned i = 0; i < T; ++i) {
        result += accum[i];
    }

    free(accum);

    return result * dx;
}

double integrateArrAlign(double a, double b, f_t f) {
    unsigned T;
    double result = 0, dx = (b - a) / STEPS;
    partialSumT *accum = 0;

#pragma omp parallel shared(accum, T)
    {
        unsigned t = (unsigned) omp_get_thread_num();
#pragma omp single
        {
            T = (unsigned) omp_get_num_threads();
            accum = (partialSumT *) aligned_alloc(CACHE_LINE, T * sizeof(partialSumT));
            memset(accum, 0, T*sizeof(*accum));
        }

        for (unsigned i = t; i < STEPS; i += T) {
            accum[t].val += f(dx * i + a);
        }
    }

    for (unsigned i = 0; i < T; ++i) {
        result += accum[i].val;
    }

    free(accum);

    return result * dx;
}

double integrateReduction(double a, double b, f_t f) {
    double result = 0, dx = (b - a) / STEPS;

#pragma omp parallel for reduction(+: result)
    for (unsigned int i = 0; i < STEPS; ++i) {
        result += f(dx * i + a);
    }

    return result * dx;
}

double integratePS(double a, double b, f_t f) {
    double dx = (b - a) / STEPS;
    double result = 0;
    unsigned T = get_num_threads();
    auto vec = vector(T, partialSumT{0.0});
    vector<thread> threadVec;

    auto threadProc = [=, &vec](auto t) {
        for (auto i = t; i < STEPS; i += T) {
            vec[t].val += f(dx * i + a);
        }
    };

    for (auto t = 1; t < T; t++) {
        threadVec.emplace_back(threadProc, t);
    }

    threadProc(0);

    for (auto &thread: threadVec) {
        thread.join();
    }

    for (auto elem: vec) {
        result += elem.val;
    }

    return result * dx;
}

double integrateAtomic(double a, double b, f_t f) {
    vector<thread> threads;
    std::atomic<double> result = {0};
    double dx = (b - a) / STEPS;
    unsigned int T = get_num_threads();

    auto fun = [dx, &result, f, a, T](auto t) {
        double R = 0;
        for (unsigned i = t; i < STEPS; i += T) {
            R += f(i * dx + a);
        }

        result += R;
    };

    for (unsigned int t = 1; t < T; ++t) {
        threads.emplace_back(fun, t);
    }

    fun(0);

    for (auto &thr: threads) {
        thr.join();
    }

    return result * dx;
}

uint64_t pow(uint64_t x, unsigned n) {
    if (n==0)
        return 1;
    else if (n==1)
        return x;
    else if (n % 2 == 0 )
        return pow( x * x, n/2);
    else
        return pow( x * x, n/2)*x;
}

uint64_t getB(unsigned i) {
    uint64_t a = 6364136223846793005;
    uint64_t sum = 0;

    for (unsigned j = 0; j < i; ++j) {
        sum += pow(a, j + 1);
    }

    return sum;
}

template<class type>
void printArray(type* array, unsigned n) {
    for (int i = 0; i < n; ++i) {
        cout << array[i] << " ";
    }

    cout << endl;
}

double RandomizeArraySingle(unsigned seed, unsigned* V, size_t n, unsigned min, unsigned max) {
    uint64_t a = 6364136223846793005;
    unsigned b = 1;

    uint64_t prev = seed;
    uint64_t Sum = 0;
    for(unsigned i = 0; i < n; i++)
    {
        uint64_t next = a*prev + b;
        V[i] = (next % (max - min + 1)) + min;
        prev = next;
        Sum += V[i];
    }

    // printArray(V, n);
    return (double)Sum/(double)n;
}

// without lookuptable, first in block calculation from seed

double RandomizeArray(unsigned seed, unsigned* V, size_t n, unsigned min, unsigned max) {
    uint64_t a = 6364136223846793005;
    unsigned b = 1;
    unsigned T;
    uint64_t lookUpA;
    uint64_t lookUpB;
    uint64_t sum = 0;
    unsigned blockSize;

#pragma omp parallel shared(T, V, lookUpA, lookUpB, blockSize)
    {
    #pragma omp single
        {
            T = (unsigned) omp_get_num_threads();
            blockSize = CACHE_LINE / sizeof(unsigned);

            lookUpA = pow(a, (T - 1)*blockSize + 1);
            lookUpB = getB((T - 1)*blockSize);
        }

        unsigned t = (unsigned) omp_get_thread_num();

        uint64_t prev;
        uint64_t elem;
        for (unsigned i = t * blockSize; i < n; i += T * blockSize) {
            if (i == t * blockSize) {
                uint64_t newA = pow(a, i + 1);
                uint64_t newB = getB(i);
                elem = newA * seed + newB + b;
                V[i] = (elem % (max - min + 1)) + min;
                prev = elem;
            } else {
                elem = lookUpA * prev + lookUpB + b;
                V[i] = (elem % (max - min + 1)) + min;
                prev = elem;
            }

            for (unsigned j = i + 1; j < i + blockSize && j < n; ++j) {
                elem = a*prev + b;
                V[j] = (elem % (max - min + 1)) + min;
                prev = elem;
            }
        }
    }

    for (unsigned i = 0; i < n; ++i) {
        sum += V[i];
    }

//     printArray(V, n);

    return (double)sum/(double)n;
}

double RandomizeArrayFalseSharing(unsigned seed, unsigned* V, size_t n, unsigned min, unsigned max) {
    uint64_t a = 6364136223846793005;
    unsigned b = 1;
    unsigned T;
    uint64_t lookUpA;
    uint64_t lookUpB;
    uint64_t sum = 0;

#pragma omp parallel shared(T, V, lookUpA, lookUpB)
    {
        unsigned t = (unsigned) omp_get_thread_num();
    #pragma omp single
        {
            T = (unsigned) omp_get_num_threads();

            lookUpA = pow(a, T);
            lookUpB = getB(T - 1);
        }
        uint64_t prev = seed;
        uint64_t elem;
        for (unsigned i = t; i < n; i += T) {
            if (i == t) {
                elem = pow(a, i + 1) * prev + getB(i) + b;
                V[i] = (elem % (max - min + 1)) + min;
                prev = elem;
            } else {
                elem = lookUpA * prev + lookUpB + b;
                V[i] = (elem % (max - min + 1)) + min;
                prev = elem;
            }
        }
    }

    for (unsigned i = 0; i < n; ++i) {
        sum += V[i];
    }

     // printArray(V, n);

    return (double)sum/(double)n;
}

double integrate_omp_dynamic(double a, double b, f_t f) {
    double result = 0, dx = (b - a) / STEPS;

#pragma omp parallel for reduction(+: result) schedule(dynamic)
    for (unsigned int i = 0; i < STEPS; ++i) {
        result += f(dx * i + a);
    }

    return result * dx;
}

ExperimentResult runExperiment(I_t I) {
    double t0, t1, result;

    t0 = omp_get_wtime();
    result = I(A, B, f);
    t1 = omp_get_wtime();

    return {result, t1 - t0};
}

void showExperimentResults(I_t I) {
    set_num_threads(1);
    ExperimentResult R = runExperiment(I);
    double T1 = R.time;

    printf("%10s\t %10s\t %10s\n", "Result", "Time", "Acceleration");

    printf("%10g\t %10g\t% 10g\n", R.result, R.time, T1/R.time);
    // printf("%d,%g,%g\n", 1, R.time, T1 / R.time);

    for (int T = 2; T <= omp_get_num_procs(); ++T) {
        set_num_threads(T);
        ExperimentResult result = runExperiment(I);
        printf("%10g\t %10g\t %10g\n", result.result, result.time, T1/result.time);
        // printf("%d,%g,%g\n", T, result.time, T1 / result.time);
    }

    cout << endl;
}

double integrate_reduction(double a, double b, f_t F)
{
    double dx = (b-a)/STEPS;
    return reduce_range(a, b, STEPS, F, [](auto x, auto y){return x + y;}, 0.0)*dx;
}

unsigned Fibonacci(unsigned n) {
    if (n <= 2)
        return 1;
    return Fibonacci(n - 1) + Fibonacci(n - 2);
}

unsigned Fibonacci_omp(unsigned n) {
    if (n <= 2)
        return 1;
    unsigned x1, x2;
    unsigned result;
#pragma omp parallel
    {
    #pragma omp task
        {
            x1 = Fibonacci_omp(n - 1);
        }
    #pragma omp task
        {
            x2 = Fibonacci_omp(n - 2);
        }
    #pragma omp taskwait
        result = x1 + x2;
    }

    return result;
}

ExperimentResult runRandomizeExperiment(r_t f) {
    size_t ArrayLength = 100000;
    unsigned Array[ArrayLength];
    unsigned Seed = 100;

    double t0, t1, result;

    t0 = omp_get_wtime();
    result = f(Seed, (unsigned *)&Array, ArrayLength, 1, 255);
    t1 = omp_get_wtime();

    return {result, t1 - t0};
}

void randomizeExperiment(r_t f) {
    set_num_threads(1);
    ExperimentResult R = runRandomizeExperiment(f);
    double T1 = R.time;


    printf("%10s\t %10s\t %10s\n", "Result", "Time", "Acceleration");

    printf("%10g\t %10g\t% 10g\n", R.result, R.time, T1/R.time);
    // printf("%d,%g,%g\n", 1, R.time, T1 / R.time);

    for (int T = 2; T <= omp_get_num_procs(); ++T) {
        set_num_threads(T);
        ExperimentResult result = runRandomizeExperiment(f);
        printf("%10g\t %10g\t %10g\n", result.result, result.time, T1/result.time);
        // printf("%d,%g,%g\n", T, result.time, T1 / result.time);
    }

    cout << endl;
}

ExperimentResult runExperimentFib() {
    double t0, t1, result;

    t0 = omp_get_wtime();
    result = Fibonacci_omp(10);
    t1 = omp_get_wtime();

    return {result, t1 - t0};
}

void experimentFib() {
    set_num_threads(1);
    ExperimentResult R = runExperimentFib();
    double T1 = R.time;


    printf("%10s\t %10s\t %10s\n", "Result", "Time", "Acceleration");

    printf("%10g\t %10g\t% 10g\n", R.result, R.time, T1/R.time);
    // printf("%d,%g,%g\n", 1, R.time, T1 / R.time);

    for (int T = 2; T <= omp_get_num_procs(); ++T) {
        set_num_threads(T);
        ExperimentResult result = runExperimentFib();
        printf("%10g\t %10g\t %10g\n", result.result, result.time, T1/result.time);
        // printf("%d,%g,%g\n", T, result.time, T1 / result.time);
    }

    cout << endl;
}

int main() {
    std::cout << "integrateDefault" << std::endl;
    showExperimentResults(integrateDefault);
    std::cout << "integrateCrit" << std::endl;
    showExperimentResults(integrateCrit);
    std::cout << "integrateMutex" << std::endl;
    showExperimentResults(integrateMutex);
    std::cout << "integrateArr" << std::endl;
    showExperimentResults(integrateArr);
    std::cout << "integrateArrAlign" << std::endl;
    showExperimentResults(integrateArrAlign);
    std::cout << "integrateReduction" << std::endl;
    showExperimentResults(integrateReduction);
    std::cout << "integratePS" << std::endl;
    showExperimentResults(integratePS);
    std::cout << "integrateAtomic" << std::endl;
    showExperimentResults(integrateAtomic);
    std::cout << "randomize single" << std::endl;
    randomizeExperiment(RandomizeArraySingle);
    std::cout << "randomize with false sharing" << std::endl;
    randomizeExperiment(RandomizeArrayFalseSharing);
    std::cout << "randomize without false sharing" << std::endl;
    randomizeExperiment(RandomizeArray);
    std::cout << "fib" << std::endl;
    experimentFib();
    std::cout << "integrate reduction c++" << std::endl;
    showExperimentResults(integrateReduction);

//    std::cout << "integrate_omp_dynamic" << std::endl;
//    showExperimentResults(integrate_omp_dynamic);

//    size_t ArrayLength = 16;
//    unsigned Array[ArrayLength];
//    unsigned Seed = 100;
//
//    set_num_threads(2);
//    cout << RandomizeArraySingle(Seed, (unsigned *)&Array, ArrayLength, 1, 255) << endl;
//    cout << RandomizeArrayFallSharing(Seed, (unsigned *)&Array, ArrayLength, 1, 255) << endl;

    return 0;
}
