#include <cstdio>
#include <exception>
#include "tensor.h"

int main() {
    try {
        Tensor s   = Tensor::scalar(2.0f);
        Tensor row = Tensor::fromHost({3},    {10.0f, 20.0f, 30.0f});
        Tensor col = Tensor::fromHost({2, 1}, {100.0f, 200.00f});
        Tensor m   = Tensor::fromHost({2, 3}, {1.0f, 2.0f, 3.0f,
                                               4.0f, 5.0f, 6.0f});

        printTensor("s", s);               //   2.00
        printTensor("row * s", row * s);   // [ 20 40 60 ]
        printTensor("m + row", m + row);   // [ 11 22 33 ] [ 14 25 36 ]
        printTensor("m + col", m + col);   // [ 101 102 103 ] [ 204 205 206 ]
        printTensor("-m", -m);             // [ -1 -2 -3 ] [ -4 -5 -6 ]

        // Should throw: (2, 3) and (2) can't broadcast
        Tensor bad = m + Tensor::fromHost({2}, {1.0f, 2.0f});
    } catch (const std::exception& e) {
        printf("Error: %s\n", e.what());
    }
    return 0;
}
