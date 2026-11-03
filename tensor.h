// tensor.h
#pragma once
#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <stdexcept>

bool isGradEnabled();
class NoGradGuard {
public:
    NoGradGuard();
    ~NoGradGuard();
private:
    bool prev_;
};

struct Node;

struct TensorImpl {
    std::shared_ptr<float> data;    
    std::vector<int> shape;
    bool requires_grad = false;
    std::shared_ptr<TensorImpl> grad;
    std::shared_ptr<Node> grad_fn;
};

class Tensor {
public:
    explicit Tensor(const std::vector<int>& shape);
    explicit Tensor(std::shared_ptr<TensorImpl> impl) : impl_(std::move(impl)) {}   // wrap an existing TensorImpl
    static Tensor fromHost(const std::vector<int>& shape, const std::vector<float>& values);
    static Tensor fromHostData(const std::vector<int>& shape, const float* values);
    static Tensor scalar(float value);
    std::vector<float> toHost() const;
    void copyToHost(float* dest) const;

    float* data() const { return impl_->data.get(); }
    const std::vector<int>& shape() const { return impl_->shape; }
    int numel() const;

    // autograd stuff
    TensorImpl* impl() const { return impl_.get(); }
    bool requiresGrad() const { return impl_->requires_grad; }
    void setRequiresGrad(bool r) { impl_->requires_grad = r; }
    bool hasGrad() const { return impl_->grad != nullptr; }
    Tensor grad() const {
    if (!impl_->grad) throw std::runtime_error("grad: this tensor has no gradient (did you call backward()?)");
    return Tensor(impl_->grad);
    }    
    void backward();
    static Tensor full(const std::vector<int>& shape, float value);   
    const std::shared_ptr<TensorImpl>& implPtr() const { return impl_; }
    
private:
    std::shared_ptr<TensorImpl> impl_;
};

struct Node{
    std::vector<Tensor> inputs;
    const char* op_name;
    std::function<std::vector<Tensor>(const Tensor& gradOut)> backward;
};


Tensor operator+(const Tensor& a, const Tensor& b);
Tensor operator-(const Tensor& a, const Tensor& b);
Tensor operator*(const Tensor& a, const Tensor& b);
Tensor operator/(const Tensor& a, const Tensor& b);
Tensor relu(const Tensor& a);
Tensor operator-(const Tensor& a);
Tensor sum(const Tensor& a);
Tensor sumAxis(const Tensor& a, int axis);
Tensor reshape(const Tensor& a, const std::vector<int>& shape);
Tensor transpose(const Tensor& a);
Tensor exp(const Tensor& a);
Tensor log(const Tensor& a);
Tensor mean(const Tensor& a);
Tensor max(const Tensor& a);
Tensor min(const Tensor& a);
Tensor dot(const Tensor& a, const Tensor& b);
Tensor matmul(const Tensor& a, const Tensor& b);
Tensor naiveMatMul(const Tensor& a, const Tensor& b);
Tensor tiledMatMul(const Tensor& a, const Tensor& b);
Tensor softmax(const Tensor& a);
Tensor softmaxCrossEntropy(const Tensor& logits, const Tensor& labels);
Tensor detach(const Tensor& a);
std::string shapeToString(const std::vector<int>& shape);
void printTensor(const char* name, const Tensor& t);
void synchronize();