#include <iostream>

int add(const int* a, const int* b) { 
    return *a + *b; 
}

int main() {
    int a = 1;
    int b = 4;
    int c = add(&a, &b);
    printf("result: %d", c);
    return(0);
}