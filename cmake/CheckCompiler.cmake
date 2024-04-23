if(CMAKE_SIZEOF_VOID_P EQUAL 4)
    set(AC_COMPILER_32BIT 1)
else()
    set(AC_COMPILER_32BIT 0)
endif()

include(CheckCXXCompilerFlag)

if (NOT MSVC)
    set(CMAKE_REQUIRED_FLAGS "-msse3")
endif()
check_cxx_source_compiles("#include <pmmintrin.h>\nint main() { __m128 u, v; u = _mm_set1_ps(0.0f); v = _mm_moveldup_ps(u); return 0; }" AC_COMPILER_SUPPORT_SSE)

if (NOT MSVC)
    set(CMAKE_REQUIRED_FLAGS "-mfma")
endif()
check_cxx_source_compiles("#include <immintrin.h>\nint main() {  __m256 s0, s1, s2; s0 = _mm256_fmadd_ps(s1, s2, s0); return 0; }" AC_COMPILER_SUPPORT_FMA)

if (NOT MSVC)
    set(CMAKE_REQUIRED_FLAGS "-mavx")
endif()
check_cxx_source_compiles("#include <immintrin.h>\nint main() {  __m256 a = _mm256_set1_ps(0.0f); return 0; }" AC_COMPILER_SUPPORT_AVX)

if (NOT MSVC AND AC_COMPILER_32BIT)
    set(CMAKE_REQUIRED_FLAGS "-mfpu=neon")
endif()
check_cxx_source_compiles("#include <arm_neon.h>\nint main() { float32x4_t a = vdupq_n_f32(0.0f); return 0; }" AC_COMPILER_SUPPORT_NEON)

if (NOT MSVC)
    unset(CMAKE_REQUIRED_FLAGS)
endif()
