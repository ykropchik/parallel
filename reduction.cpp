#include <thread>
#include <vector>
#include <iostream>
#include <barrier>

#ifdef __cpp_lib_hardware_interference_size
    using std::hardware_constructive_interference_size;
    using std::hardware_destructive_interference_size;
#else
    constexpr std::size_t hardware_constructive_interference_size = 64;
    constexpr std::size_t hardware_destructive_interference_size = 64;
#endif

unsigned get_num_threads();

void set_num_threads(unsigned T);

auto ceil_div(auto x, auto y) {
    return (x + y - 1) / y;
}

template <class ElementType, class BinaryFn>
ElementType reduce_vector(const ElementType* V, std::size_t n, BinaryFn f, ElementType zero) {

    struct reduction_partial_result_t {
        alignas(hardware_destructive_interference_size) ElementType value;
    };
    unsigned T = get_num_threads();
    static auto reduction_partial_results = std::vector<reduction_partial_result_t>(T, reduction_partial_result_t{zero});

    constexpr std::size_t k = 2;
    auto thread_proc = [=] (unsigned t) {
        auto K = ceil_div(n, k);
        std::size_t Mt = K/T, it1 = k % T;
        if (t < it1) {
           it1 = ++Mt * t;
        } else {
           it1 = Mt * it1 + t;
        }
        it1 *= k;
        std::size_t mt = Mt * k;
        auto it2 = it1 * mt;
        ElementType accum = zero;

        for (std::size_t i = it1; i < it2; ++i) {
            accum = f(accum, V[i]);
        }

        reduction_partial_results[t].value = accum;
    };

//    auto thread_proc_2 = [=] (unsigned t) {
//        constexpr std::size_t k = 2;
//        std::size_t s = 1;
//
//        while((t % (s * k)) == 0 && s + t < T) {
//            reduction_partial_results[T].value = f(reduction_partial_results[t].value, reduction_partial_results[t + s].value);
//            s *= k;
//        }
//    };

    auto thread_proc_2_ = [=] (unsigned t, std::size_t s) {
        if(((t % (s * k)) == 0) && (t + s < T))
            reduction_partial_results[t].value = f(reduction_partial_results[t].value,
                                                   reduction_partial_results[t + s].value);
    };

    std::vector<std::thread> threads;
    for(unsigned t = 1; t < T; t++) {
        threads.emplace_back(thread_proc, t);
    }

    thread_proc(0);

    for(auto& thread:threads) {
        thread.join();
    }

    std::size_t s = 1;
    while(s < T) {
        for(unsigned t = 1; t < T; ++t) {
            threads[t - 1] = std::thread(thread_proc_2_, t, s);
        }
        thread_proc_2_(0, s);
        s *= k;

        for(auto& thread:threads) {
            thread.join();
        }
    }

//    for(unsigned t = 1; t < T; t++) {
//        threads[t-1] = std::thread(thread_proc_2, t);
//    }
//
//    thread_proc_2(0);
//
//    for(auto& thread:threads) {
//        thread.join();
//    }

    return reduction_partial_results[0].value;
}

template <class ElementType, class UnaryFn, class BinaryFn>
requires (
        std::is_invocable_r_v<ElementType, UnaryFn, ElementType> &&
        std::is_invocable_r_v<ElementType, BinaryFn, ElementType, ElementType>
)
ElementType reduce_range(ElementType a, ElementType b, std::size_t n, UnaryFn get, BinaryFn reduce_2, ElementType zero) {
    unsigned T = get_num_threads();
    struct reduction_partial_result_t
    {
        alignas(hardware_destructive_interference_size) ElementType value;
    };
    static auto reduction_partial_results =
            std::vector<reduction_partial_result_t>(std::thread::hardware_concurrency(), reduction_partial_result_t{zero});

    std::barrier<> bar{T};
    constexpr std::size_t k = 2;
    auto thread_proc = [=, &bar](unsigned t)
    {

        auto K = ceil_div(n, k);
        double dx = (b - a) / n;
        std::size_t Mt = K / T;
        std::size_t it1 = K % T;

        if(t < it1)
        {
            it1 = ++Mt * t;
        }
        else
        {
            it1 = Mt * t + it1;
        }
        it1 *= k;
        std::size_t mt = Mt * k;
        std::size_t it2 = it1 + mt;

        ElementType accum = zero;
        for(std::size_t i = it1; i < it2; i++)
            accum = reduce_2(accum, get(a + i*dx));

        reduction_partial_results[t].value = accum;

        for(std::size_t s = 1, s_next = 2; s < T; s = s_next, s_next += s_next)
        {
            bar.arrive_and_wait();
            if(((t % s_next) == 0) && (t + s < T))
                reduction_partial_results[t].value = reduce_2(reduction_partial_results[t].value,
                                                              reduction_partial_results[t + s].value);
        }
    };

    std::vector<std::thread> threads;
    for(unsigned t = 1; t < T; t++)
        threads.emplace_back(thread_proc, t);
    thread_proc(0);
    for(auto& thread : threads)
        thread.join();
    return reduction_partial_results[0].value;
}

int reduction() {
    unsigned V[15];
    for(unsigned i = 0; i < std::size(V); ++i) {
        V[i] = i + 1;
    }

    std::cout << "Average: " << reduce_vector(V, 16, [](auto x, auto y) { return x + y;}, 0u) / std::size(V) << "\n";
    return 0;
}