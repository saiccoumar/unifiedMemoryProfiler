// Jetson benchmark — unified-memory SoC (integrated GPU, CUDA 12).
// CPU and GPU share DRAM, so "transfers" are address remappings not data movement.
// Adds producer-consumer (D1) and concurrent CPU/GPU access (D2) vs discrete_benchmark.
// Build: nvcc -O3 -std=c++14 jetson_benchmark.cu -o jetson_benchmark

#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <cmath>
#include <cstring>
#include <functional>
#include <string>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

// ---- Enums ---------------------------------------------------------------

enum class MemoryModel {
    PAGEABLE,             // A1: malloc + cudaMemcpy
    PINNED,               // A2: cudaMallocHost + cudaMemcpy
    ZERO_COPY,            // A3: cudaHostAllocMapped — key test on shared-DRAM SoC
    MANAGED_NO_PREFETCH,  // B1: demand paging, no hints
    MANAGED_PREFETCH,     // B2: explicit prefetch
    MANAGED_THRASHING,    // B3: interleaved CPU/GPU ownership
    PRODUCER_CONSUMER,    // D1: CPU→GPU→CPU pipeline, no explicit copies
    CONCURRENT_ACCESS,    // D2: CPU and GPU on separate halves simultaneously
};

enum class AccessPattern {
    SEQUENTIAL,  // C1: data[i]
    SPARSE,      // C2: data[lcg(i) % n]
    STRIDED,     // C3: data[i * stride]
};

// ---- Config / Result -----------------------------------------------------

struct BenchmarkConfig {
    MemoryModel   model;
    AccessPattern pattern;
    size_t        bytes;
    int           n_runs;
    int           stride          = 8;
    int           thrash_iters    = 10;
    int           pipeline_stages = 5;   // D1 only
};

struct ProfileResult {
    std::string label;
    float alloc_ms, h2d_ms, kernel_ms, d2h_ms, total_ms;
};

// ---- Kernels -------------------------------------------------------------

__global__ void k_sequential(float* d, size_t n, float f) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] *= f;
}

__global__ void k_sparse(float* d, size_t n, float f) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        size_t idx = (i * 6364136223846793005ULL + 1442695040888963407ULL) % n;
        d[idx] *= f;
    }
}

__global__ void k_strided(float* d, size_t n, int stride, float f) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i * (size_t)stride < n) d[i * (size_t)stride] *= f;
}

// Reads src, writes scaled result to dst — used by producer-consumer pipeline.
__global__ void k_transform(const float* src, float* dst, size_t n, float f) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i] * f;
}

// ---- Timing helpers ------------------------------------------------------

using Clock = std::chrono::high_resolution_clock;

template<typename Fn>
static float timeCpu(Fn&& fn) {
    auto t0 = Clock::now();
    fn();
    return std::chrono::duration<float, std::milli>(Clock::now() - t0).count();
}

static float timeGpu(cudaEvent_t s, cudaEvent_t e, std::function<void()> fn) {
    cudaEventRecord(s);
    fn();
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms = 0;
    cudaEventElapsedTime(&ms, s, e);
    return ms;
}

static void launchKernel(float* data, size_t N, AccessPattern pat, int stride) {
    const int T = 256;
    size_t work = (pat == AccessPattern::STRIDED) ? N / stride : N;
    int B = (int)((work + T - 1) / T);
    switch (pat) {
        case AccessPattern::SEQUENTIAL: k_sequential<<<B, T>>>(data, N, 2.0f);        break;
        case AccessPattern::SPARSE:     k_sparse<<<B, T>>>(data, N, 2.0f);            break;
        case AccessPattern::STRIDED:    k_strided<<<B, T>>>(data, N, stride, 2.0f);   break;
    }
}

// ---- Managed memory API --------------------------------------------------

static void advisePreferGpu(void* p, size_t n, int dev) {
    cudaMemLocation loc{cudaMemLocationTypeDevice, dev};
    CHECK_CUDA(cudaMemAdvise(p, n, cudaMemAdviseSetPreferredLocation, loc));
}

static void prefetchToGpu(void* p, size_t n, int dev) {
    cudaMemLocation loc{cudaMemLocationTypeDevice, dev};
    CHECK_CUDA(cudaMemPrefetchAsync(p, n, loc, 0));
}

static void prefetchToCpu(void* p, size_t n) {
    cudaMemLocation loc{cudaMemLocationTypeHost, 0};
    CHECK_CUDA(cudaMemPrefetchAsync(p, n, loc, 0));
}

// ---- Runners -------------------------------------------------------------

static ProfileResult runPageable(size_t N, size_t bytes, AccessPattern pat, int stride,
                                  cudaEvent_t s, cudaEvent_t e) {
    ProfileResult r{"", 0, 0, 0, 0, 0};
    float *h, *d;
    r.alloc_ms  = timeCpu([&]{ h = (float*)malloc(bytes); CHECK_CUDA(cudaMalloc(&d, bytes)); });
    for (size_t i = 0; i < N; ++i) h[i] = 1.0f;
    r.h2d_ms    = timeGpu(s, e, [&]{ CHECK_CUDA(cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice)); });
    r.kernel_ms = timeGpu(s, e, [&]{ launchKernel(d, N, pat, stride); });
    r.d2h_ms    = timeGpu(s, e, [&]{ CHECK_CUDA(cudaMemcpy(h, d, bytes, cudaMemcpyDeviceToHost)); });
    r.total_ms  = r.alloc_ms + r.h2d_ms + r.kernel_ms + r.d2h_ms;
    free(h); cudaFree(d);
    return r;
}

static ProfileResult runPinned(size_t N, size_t bytes, AccessPattern pat, int stride,
                                cudaEvent_t s, cudaEvent_t e) {
    ProfileResult r{"", 0, 0, 0, 0, 0};
    float *h, *d;
    r.alloc_ms  = timeCpu([&]{ CHECK_CUDA(cudaMallocHost(&h, bytes)); CHECK_CUDA(cudaMalloc(&d, bytes)); });
    for (size_t i = 0; i < N; ++i) h[i] = 1.0f;
    r.h2d_ms    = timeGpu(s, e, [&]{ CHECK_CUDA(cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice)); });
    r.kernel_ms = timeGpu(s, e, [&]{ launchKernel(d, N, pat, stride); });
    r.d2h_ms    = timeGpu(s, e, [&]{ CHECK_CUDA(cudaMemcpy(h, d, bytes, cudaMemcpyDeviceToHost)); });
    r.total_ms  = r.alloc_ms + r.h2d_ms + r.kernel_ms + r.d2h_ms;
    cudaFreeHost(h); cudaFree(d);
    return r;
}

// On Jetson the GPU reads from the same DRAM as CPU — no PCIe hop.
static ProfileResult runZeroCopy(size_t N, size_t bytes, AccessPattern pat, int stride,
                                  cudaEvent_t s, cudaEvent_t e) {
    ProfileResult r{"", 0, 0, 0, 0, 0};
    float *h, *dptr;
    r.alloc_ms  = timeCpu([&]{
        CHECK_CUDA(cudaHostAlloc(&h, bytes, cudaHostAllocMapped));
        CHECK_CUDA(cudaHostGetDevicePointer(&dptr, h, 0));
    });
    for (size_t i = 0; i < N; ++i) h[i] = 1.0f;
    r.kernel_ms = timeGpu(s, e, [&]{ launchKernel(dptr, N, pat, stride); });
    r.total_ms  = r.alloc_ms + r.kernel_ms;
    cudaFreeHost(h);
    return r;
}

static ProfileResult runManagedNoPrefetch(size_t N, size_t bytes, AccessPattern pat, int stride,
                                           cudaEvent_t s, cudaEvent_t e) {
    ProfileResult r{"", 0, 0, 0, 0, 0};
    float* u;
    r.alloc_ms  = timeCpu([&]{ CHECK_CUDA(cudaMallocManaged(&u, bytes)); });
    memset(u, 0, bytes);
    cudaDeviceSynchronize();
    r.kernel_ms = timeGpu(s, e, [&]{ launchKernel(u, N, pat, stride); });
    r.total_ms  = r.alloc_ms + r.kernel_ms;
    cudaFree(u);
    return r;
}

static ProfileResult runManagedPrefetch(size_t N, size_t bytes, AccessPattern pat, int stride,
                                         int dev, cudaEvent_t s, cudaEvent_t e) {
    ProfileResult r{"", 0, 0, 0, 0, 0};
    float* u;
    r.alloc_ms  = timeCpu([&]{ CHECK_CUDA(cudaMallocManaged(&u, bytes)); advisePreferGpu(u, bytes, dev); });
    memset(u, 0, bytes);
    cudaDeviceSynchronize();
    r.h2d_ms    = timeGpu(s, e, [&]{ prefetchToGpu(u, bytes, dev); });
    r.kernel_ms = timeGpu(s, e, [&]{ launchKernel(u, N, pat, stride); });
    r.d2h_ms    = timeGpu(s, e, [&]{ prefetchToCpu(u, bytes); });
    r.total_ms  = r.alloc_ms + r.h2d_ms + r.kernel_ms + r.d2h_ms;
    cudaFree(u);
    return r;
}

// kernel_ms holds wall time of the full thrash loop.
static ProfileResult runManagedThrashing(size_t N, size_t bytes, int iters,
                                          cudaEvent_t /*s*/, cudaEvent_t /*e*/) {
    ProfileResult r{"", 0, 0, 0, 0, 0};
    float* u;
    r.alloc_ms = timeCpu([&]{ CHECK_CUDA(cudaMallocManaged(&u, bytes)); });
    memset(u, 0, bytes);
    cudaDeviceSynchronize();

    size_t cpu_region = N / 4;
    int B = (int)((N + 255) / 256);
    r.kernel_ms = timeCpu([&]{
        for (int i = 0; i < iters; ++i) {
            CHECK_CUDA(cudaDeviceSynchronize());
            for (size_t j = 0; j < cpu_region; ++j) u[j] = (float)i;
            k_sequential<<<B, 256>>>(u, N, 2.0f);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
    });
    r.total_ms = r.alloc_ms + r.kernel_ms;
    cudaFree(u);
    return r;
}

// D1: CPU writes src, GPU transforms src→dst, CPU reads dst — no explicit copies.
// Models a sensor-preprocessing → inference → postprocessing pipeline.
// kernel_ms holds wall time of pipeline_stages complete CPU→GPU→CPU rounds.
static ProfileResult runProducerConsumer(size_t N, size_t bytes, int stages,
                                          cudaEvent_t /*s*/, cudaEvent_t /*e*/) {
    ProfileResult r{"", 0, 0, 0, 0, 0};
    float *src, *dst;
    r.alloc_ms = timeCpu([&]{
        CHECK_CUDA(cudaMallocManaged(&src, bytes));
        CHECK_CUDA(cudaMallocManaged(&dst, bytes));
    });
    memset(src, 0, bytes);
    memset(dst, 0, bytes);
    cudaDeviceSynchronize();

    int B = (int)((N + 255) / 256);
    r.kernel_ms = timeCpu([&]{
        for (int s = 0; s < stages; ++s) {
            for (size_t i = 0; i < N; ++i) src[i] = (float)s;  // CPU produce
            CHECK_CUDA(cudaDeviceSynchronize());
            k_transform<<<B, 256>>>(src, dst, N, 2.0f);          // GPU consume/transform
            CHECK_CUDA(cudaDeviceSynchronize());
            volatile float check = dst[0]; (void)check;          // CPU consume (peek)
        }
    });
    r.total_ms = r.alloc_ms + r.kernel_ms;
    cudaFree(src); cudaFree(dst);
    return r;
}

// D2: GPU processes upper half while CPU writes lower half concurrently.
// SetAccessedBy lets both processors hold live TLB mappings without faulting
// on each other's non-overlapping region — measures coherence traffic cost.
// kernel_ms holds wall time of the concurrent phase (including final sync).
static ProfileResult runConcurrentAccess(size_t N, size_t bytes, int dev,
                                          cudaEvent_t /*s*/, cudaEvent_t /*e*/) {
    ProfileResult r{"", 0, 0, 0, 0, 0};
    float* u;
    r.alloc_ms = timeCpu([&]{
        CHECK_CUDA(cudaMallocManaged(&u, bytes));
        cudaMemLocation gpuLoc{cudaMemLocationTypeDevice, dev};
        cudaMemLocation cpuLoc{cudaMemLocationTypeHost, 0};
        CHECK_CUDA(cudaMemAdvise(u, bytes, cudaMemAdviseSetAccessedBy, gpuLoc));
        CHECK_CUDA(cudaMemAdvise(u, bytes, cudaMemAdviseSetAccessedBy, cpuLoc));
    });
    memset(u, 0, bytes);
    cudaDeviceSynchronize();

    size_t half = N / 2;
    int B = (int)((half + 255) / 256);
    r.kernel_ms = timeCpu([&]{
        k_sequential<<<B, 256>>>(u + half, half, 2.0f);          // GPU: upper half (async)
        for (size_t i = 0; i < half; ++i) u[i] = (float)i;      // CPU: lower half (concurrent)
        CHECK_CUDA(cudaDeviceSynchronize());
    });
    r.total_ms = r.alloc_ms + r.kernel_ms;
    cudaFree(u);
    return r;
}

// ---- Output --------------------------------------------------------------

static std::pair<double, double> colStats(const std::vector<ProfileResult>& res,
                                           float ProfileResult::*field) {
    double sum = 0, sq = 0;
    int n = (int)res.size();
    for (const auto& r : res) { double v = r.*field; sum += v; sq += v * v; }
    double m = sum / n;
    return {m, std::sqrt(sq / n - m * m)};
}

static void reportResults(std::ofstream& csv, const std::vector<ProfileResult>& res) {
    for (size_t i = 0; i < res.size(); ++i) {
        const auto& r = res[i];
        printf("%zu, %s, %.2f, %.2f, %.2f, %.2f, %.2f\n",
               i + 1, r.label.c_str(), r.alloc_ms, r.h2d_ms, r.kernel_ms, r.d2h_ms, r.total_ms);
        csv << (i+1) << "," << r.label << "," << r.alloc_ms << "," << r.h2d_ms << ","
            << r.kernel_ms << "," << r.d2h_ms << "," << r.total_ms << "\n";
    }
    auto a = colStats(res, &ProfileResult::alloc_ms);
    auto h = colStats(res, &ProfileResult::h2d_ms);
    auto k = colStats(res, &ProfileResult::kernel_ms);
    auto d = colStats(res, &ProfileResult::d2h_ms);
    auto t = colStats(res, &ProfileResult::total_ms);
    const std::string& lbl = res[0].label;
    printf("Mean±Std, %s, %.2f±%.2f, %.2f±%.2f, %.2f±%.2f, %.2f±%.2f, %.2f±%.2f\n\n",
           lbl.c_str(), a.first, a.second, h.first, h.second,
           k.first, k.second, d.first, d.second, t.first, t.second);
    csv << "Mean,"   << lbl << "," << a.first  << "," << h.first  << "," << k.first  << "," << d.first  << "," << t.first  << "\n";
    csv << "StdDev," << lbl << "," << a.second << "," << h.second << "," << k.second << "," << d.second << "," << t.second << "\n";
}

static const char* modelStr(MemoryModel m) {
    switch (m) {
        case MemoryModel::PAGEABLE:            return "Pageable";
        case MemoryModel::PINNED:              return "Pinned";
        case MemoryModel::ZERO_COPY:           return "ZeroCopy";
        case MemoryModel::MANAGED_NO_PREFETCH: return "ManagedNoPrefetch";
        case MemoryModel::MANAGED_PREFETCH:    return "ManagedPrefetch";
        case MemoryModel::MANAGED_THRASHING:   return "ManagedThrashing";
        case MemoryModel::PRODUCER_CONSUMER:   return "ProducerConsumer";
        case MemoryModel::CONCURRENT_ACCESS:   return "ConcurrentAccess";
    }
    return "Unknown";
}

static const char* patternStr(AccessPattern p) {
    switch (p) {
        case AccessPattern::SEQUENTIAL: return "Sequential";
        case AccessPattern::SPARSE:     return "Sparse";
        case AccessPattern::STRIDED:    return "Strided";
    }
    return "Unknown";
}

// ---- Main ----------------------------------------------------------------

int main() {
    CHECK_CUDA(cudaSetDeviceFlags(cudaDeviceMapHost));

    int dev;
    CHECK_CUDA(cudaGetDevice(&dev));
    cudaDeviceProp props;
    CHECK_CUDA(cudaGetDeviceProperties(&props, dev));
    printf("Device: %s  Total mem: %.1f GB  Integrated: %s\n\n",
           props.name, (double)props.totalGlobalMem / (1ULL << 30),
           props.integrated ? "yes" : "no");

    const size_t DEFAULT_BYTES = 12ULL << 30;
    // On Jetson totalGlobalMem == system RAM — cap oversubscription well below physical limit.
    const size_t OVERSUB_BYTES = (size_t)(props.totalGlobalMem * 0.85);
    const int N_RUNS = 30;

    const std::vector<BenchmarkConfig> configs = {
        // A: Baselines
        {MemoryModel::PAGEABLE,            AccessPattern::SEQUENTIAL, DEFAULT_BYTES, N_RUNS},
        {MemoryModel::PINNED,              AccessPattern::SEQUENTIAL, DEFAULT_BYTES, N_RUNS},
        {MemoryModel::ZERO_COPY,           AccessPattern::SEQUENTIAL, DEFAULT_BYTES, N_RUNS},
        // B: Managed memory variants
        {MemoryModel::MANAGED_NO_PREFETCH, AccessPattern::SEQUENTIAL, DEFAULT_BYTES, N_RUNS},
        {MemoryModel::MANAGED_PREFETCH,    AccessPattern::SEQUENTIAL, DEFAULT_BYTES, N_RUNS},
        {MemoryModel::MANAGED_THRASHING,   AccessPattern::SEQUENTIAL, DEFAULT_BYTES, N_RUNS},
        // C: Access pattern variants
        {MemoryModel::PINNED,              AccessPattern::STRIDED,    DEFAULT_BYTES, N_RUNS},
        {MemoryModel::PINNED,              AccessPattern::SPARSE,     DEFAULT_BYTES, N_RUNS},
        {MemoryModel::MANAGED_PREFETCH,    AccessPattern::STRIDED,    DEFAULT_BYTES, N_RUNS},
        {MemoryModel::MANAGED_PREFETCH,    AccessPattern::SPARSE,     DEFAULT_BYTES, N_RUNS},
        {MemoryModel::MANAGED_NO_PREFETCH, AccessPattern::STRIDED,    DEFAULT_BYTES, N_RUNS},
        {MemoryModel::MANAGED_NO_PREFETCH, AccessPattern::SPARSE,     DEFAULT_BYTES, N_RUNS},
        // C4: oversubscription within safe fraction of system RAM
        {MemoryModel::MANAGED_NO_PREFETCH, AccessPattern::SEQUENTIAL, OVERSUB_BYTES, N_RUNS},
        {MemoryModel::MANAGED_PREFETCH,    AccessPattern::SEQUENTIAL, OVERSUB_BYTES, N_RUNS},
        // D: Jetson-specific pipeline and coherence tests
        {MemoryModel::PRODUCER_CONSUMER,   AccessPattern::SEQUENTIAL, DEFAULT_BYTES, N_RUNS},
        {MemoryModel::CONCURRENT_ACCESS,   AccessPattern::SEQUENTIAL, DEFAULT_BYTES, N_RUNS},
    };

    std::ofstream csv("jetson_results.csv");
    csv << "Run,Label,Alloc_ms,H2D_ms,Kernel_ms,D2H_ms,Total_ms\n";

    cudaEvent_t evS, evE;
    CHECK_CUDA(cudaEventCreate(&evS));
    CHECK_CUDA(cudaEventCreate(&evE));

    for (const auto& cfg : configs) {
        size_t N = cfg.bytes / sizeof(float);
        std::string lbl = std::string(modelStr(cfg.model)) + "_" + patternStr(cfg.pattern);
        if (cfg.bytes != DEFAULT_BYTES) lbl += "_Oversub";

        printf("=== %s  [%.1f GB, %d runs] ===\n",
               lbl.c_str(), (double)cfg.bytes / (1ULL << 30), cfg.n_runs);

        std::vector<ProfileResult> results;
        results.reserve(cfg.n_runs);

        for (int i = 0; i < cfg.n_runs; ++i) {
            printf("  run %d/%d\r", i + 1, cfg.n_runs);
            fflush(stdout);

            ProfileResult r;
            switch (cfg.model) {
                case MemoryModel::PAGEABLE:
                    r = runPageable(N, cfg.bytes, cfg.pattern, cfg.stride, evS, evE); break;
                case MemoryModel::PINNED:
                    r = runPinned(N, cfg.bytes, cfg.pattern, cfg.stride, evS, evE); break;
                case MemoryModel::ZERO_COPY:
                    r = runZeroCopy(N, cfg.bytes, cfg.pattern, cfg.stride, evS, evE); break;
                case MemoryModel::MANAGED_NO_PREFETCH:
                    r = runManagedNoPrefetch(N, cfg.bytes, cfg.pattern, cfg.stride, evS, evE); break;
                case MemoryModel::MANAGED_PREFETCH:
                    r = runManagedPrefetch(N, cfg.bytes, cfg.pattern, cfg.stride, dev, evS, evE); break;
                case MemoryModel::MANAGED_THRASHING:
                    r = runManagedThrashing(N, cfg.bytes, cfg.thrash_iters, evS, evE); break;
                case MemoryModel::PRODUCER_CONSUMER:
                    r = runProducerConsumer(N, cfg.bytes, cfg.pipeline_stages, evS, evE); break;
                case MemoryModel::CONCURRENT_ACCESS:
                    r = runConcurrentAccess(N, cfg.bytes, dev, evS, evE); break;
            }
            r.label = lbl;
            results.push_back(r);
        }
        printf("\nRun, Label, Alloc(ms), H2D(ms), Kernel(ms), D2H(ms), Total(ms)\n");
        reportResults(csv, results);
    }

    csv.close();
    CHECK_CUDA(cudaEventDestroy(evS));
    CHECK_CUDA(cudaEventDestroy(evE));
    printf("Results written to jetson_results.csv\n");
    return 0;
}
