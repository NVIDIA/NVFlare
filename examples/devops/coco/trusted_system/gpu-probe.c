/* CUDA driver smoke test; no CUDA toolkit headers or compiler required. */
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define LOAD(var, symbol) do { *(void **)(&var) = dlsym(lib, symbol); \
    if (!(var)) { fprintf(stderr, "Missing CUDA symbol: %s\n", symbol); return 1; } } while (0)
#define CHECK(call) do { int rc = (call); if (rc) { \
    fprintf(stderr, "%s failed: CUDA error %d\n", #call, rc); return 1; } } while (0)

int main(void) {
    void *lib = dlopen("libcuda.so.1", RTLD_NOW);
    if (!lib) { fprintf(stderr, "Cannot load libcuda.so.1: %s\n", dlerror()); return 1; }
    int (*init)(unsigned), (*count)(int *), (*get)(int *, int);
    int (*name)(char *, int, int), (*create)(void **, unsigned, int);
    int (*alloc)(unsigned long long *, size_t), (*set)(unsigned long long, unsigned char, size_t);
    int (*copy)(void *, unsigned long long, size_t), (*release)(unsigned long long), (*destroy)(void *);
    LOAD(init, "cuInit"); LOAD(count, "cuDeviceGetCount"); LOAD(get, "cuDeviceGet");
    LOAD(name, "cuDeviceGetName"); LOAD(create, "cuCtxCreate_v2"); LOAD(alloc, "cuMemAlloc_v2");
    LOAD(set, "cuMemsetD8_v2"); LOAD(copy, "cuMemcpyDtoH_v2");
    LOAD(release, "cuMemFree_v2"); LOAD(destroy, "cuCtxDestroy_v2");
    CHECK(init(0)); int n = 0; CHECK(count(&n));
    printf("CUDA device count: %d\n", n);
    if (n != 1) { fprintf(stderr, "Expected exactly one assigned GPU\n"); return 1; }
    int dev; char label[256]; void *context = NULL; unsigned long long memory = 0;
    CHECK(get(&dev, 0)); CHECK(name(label, sizeof(label), dev));
    printf("CUDA device: %s\n", label); CHECK(create(&context, 0, dev));
    unsigned char output[4096]; CHECK(alloc(&memory, sizeof(output)));
    CHECK(set(memory, 0x5a, sizeof(output))); CHECK(copy(output, memory, sizeof(output)));
    int valid = 1;
    for (size_t i = 0; i < sizeof(output); i++) if (output[i] != 0x5a) valid = 0;
    CHECK(release(memory)); CHECK(destroy(context)); dlclose(lib);
    if (!valid) { fprintf(stderr, "GPU memory round-trip mismatch\n"); return 1; }
    puts("PASS: CUDA context, GPU memory allocation, memset and device-to-host verification");
    return 0;
}
