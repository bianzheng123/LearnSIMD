#include <iostream>
#include <vector>
#include <x86intrin.h>
#include <cassert>
#include <random>

std::vector<float> generate_array(const uint32_t size) {
    // Declare an array of floating-point numbers
    std::vector<float> array(size);

    // std::random_device rd; // Seed generator
    // std::mt19937 gen(rd()); // Mersenne Twister engine
    std::mt19937 gen(0); // Mersenne Twister engine
    std::uniform_real_distribution<> dis(0.0, 1.0); // Uniform distribution between 0 and 1

    for (int i = 0; i < size; ++i) {
        array[i] = dis(gen);
    }
    return array;
}

std::vector<float> add_vector_simd(std::vector<float> arr1, std::vector<float> arr2, const uint32_t n_pack_num) {
    assert(arr1.size() == arr2.size());
    const uint32_t size = arr1.size();

    std::vector<float> result(size);

    for (uint32_t i = 0; i + (n_pack_num - 1) < size; i += n_pack_num) {\
        __m256 reg_arra = _mm256_loadu_ps(arr1.data() + i);
        __m256 reg_arrb = _mm256_loadu_ps(arr2.data() + i);

        __m256 reg_arr = _mm256_add_ps(reg_arra, reg_arrb);

        _mm256_storeu_ps(result.data() + i, reg_arr);
    }

    uint32_t offset = size - size / n_pack_num;
    for (uint32_t i = offset; i < size; i++) {
        result[i] = arr1[i] + arr2[i];
    }

    return result;
}

std::vector<float> add_vector_simd_align(std::vector<float> arr1, std::vector<float> arr2, const uint32_t n_pack_num) {
    assert(arr1.size() == arr2.size());
    const uint32_t size = arr1.size();

    const uint32_t pad_size = size + (size % n_pack_num == 0 ? 0 : n_pack_num - size % n_pack_num);
    assert(pad_size > size && pad_size % n_pack_num == 0);
    float *arr1_align = (float *) std::aligned_alloc(32, sizeof(float) * pad_size);
    for (uint32_t i = 0; i < pad_size; i++) {
        if (i < size) {
            arr1_align[i] = arr1[i];
        } else {
            arr1_align[i] = 0;
        }
    }
    float *arr2_align = (float *) std::aligned_alloc(32, sizeof(float) * pad_size);
    for (uint32_t i = 0; i < pad_size; i++) {
        if (i < size) {
            arr2_align[i] = arr2[i];
        } else {
            arr2_align[i] = 0;
        }
    }
    float *result_align = (float *) std::aligned_alloc(32, sizeof(float) * pad_size);

    for (uint32_t i = 0; i < size; i += n_pack_num) {
        __m256 reg_arra = _mm256_load_ps(arr1_align + i);
        __m256 reg_arrb = _mm256_load_ps(arr2_align + i);

        __m256 reg_arr = _mm256_add_ps(reg_arra, reg_arrb);

        _mm256_store_ps(result_align + i, reg_arr);
    }

    std::vector<float> result(size);
    result.assign(result_align, result_align + pad_size);

    return result;
}

std::vector<float> add_vector_simple(std::vector<float> arr1, std::vector<float> arr2) {
    assert(arr1.size() == arr2.size());
    const uint32_t size = arr1.size();

    std::vector<float> result(size);
    std::transform(arr1.begin(), arr1.end(), arr2.begin(), result.begin(),
                   [](const float a, const float b) { return a + b; });
    return result;
}


int main(int argc, char *argv[]) {
    uint32_t size = 100;
    const uint32_t n_pack_num = 256 / 32;
    std::vector<float> arra = generate_array(size);
    std::vector<float> arrb = generate_array(size);

    std::vector<float> arr_simd = add_vector_simd(arra, arrb, n_pack_num);
    std::vector<float> arr_simd_align = add_vector_simd_align(arra, arrb, n_pack_num);
    std::vector<float> arr_simple = add_vector_simple(arra, arrb);

    for (uint32_t i = 0; i < size; i++) {
        assert(arr_simd[i] == arr_simple[i]);
        if (arr_simd[i] != arr_simple[i]) {
            printf("arr_simd[%d] != arr_simple[%d], simd value %.3f, simple value %.3f\n",
                   i, i, arr_simd[i], arr_simple[i]);
        }

        assert(arr_simd_align[i] == arr_simple[i]);
        if (arr_simd_align[i] != arr_simple[i]) {
            printf("arr_simd_align[%d] != arr_simple[%d], simd value %.3f, simple value %.3f\n",
                   i, i, arr_simd_align[i], arr_simple[i]);
        }
    }


    return 0;
}
