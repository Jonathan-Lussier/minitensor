#pragma once
#include <memory>
#include <string>
#include <vector>

struct TensorImpl {
    std::shared_ptr<float> data;    
    std::vector<int> shape;
};

class Tensor {
public:
    explicit Tensor(const std::vector<int>& shape);
    static Tensor fromHost(const std::vector<int>& shape, const std::vector<float>& values);
    static Tensor fromHostData(const std::vector<int>& shape, const float* values);
    static Tensor scalar(float value);
    std::vector<float> toHost() const;
    void copyToHost(float* dest) const;

    float* data() const { return impl_->data.get(); }
    const std::vector<int>& shape() const { return impl_->shape; }
    int numel() const;

private:
    std::shared_ptr<TensorImpl> impl_;
};

Tensor operator+(const Tensor& a, const Tensor& b);
Tensor operator-(const Tensor& a, const Tensor& b);
Tensor operator*(const Tensor& a, const Tensor& b);
Tensor operator/(const Tensor& a, const Tensor& b);
Tensor operator-(const Tensor& a);
Tensor sum(const Tensor& a);
Tensor mean(const Tensor& a);
Tensor max(const Tensor& a);
Tensor min(const Tensor& a);
Tensor dot(const Tensor& a, const Tensor& b);
Tensor naiveMatMul(const Tensor& a, const Tensor& b);
Tensor tiledMatMul(const Tensor& a, const Tensor& b);

std::string shapeToString(const std::vector<int>& shape);
void printTensor(const char* name, const Tensor& t);
void synchronize();