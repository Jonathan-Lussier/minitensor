#include <iostream>

int add(const int* a, const int* b) { 
    return *a + *b; 
}

int main() {
    int a, b;
    int c = add(&a, &b);
    printf("result: %d", c);
    return(0);
}