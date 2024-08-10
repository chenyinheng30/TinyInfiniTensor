#include "core/graph.h"
#include "core/blob.h"
#include "core/common.h"
#include "core/op_type.h"
#include "core/ref.h"
#include "core/runtime.h"
#include "core/tensor.h"
#include "operators/matmul.h"
#include "operators/transpose.h"
#include <algorithm>
#include <any>
#include <cstddef>
#include <memory>
#include <numeric>
#include <queue>

namespace infini
{

    void GraphObj::addOperatorAndConnect(const Operator &op)
    {
        sorted = false;
        ops.push_back(op);
        for (auto &input : op->getInputs())
        {
            if (input)
            {
                input->addTarget(op);
                if (auto pred = input->getSource())
                {
                    pred->addSuccessors(op);
                    op->addPredecessors(pred);
                }
            }
        }
        for (auto &output : op->getOutputs())
        {
            if (output)
            {
                output->setSource(op);
                for (auto &succ : output->getTargets())
                {
                    succ->addPredecessors(op);
                    op->addSuccessors(succ);
                }
            }
        }
    }

    string GraphObj::toString() const
    {
        std::ostringstream oss;
        oss << "Graph Tensors:\n";
        for (const auto &tensor : tensors)
            oss << tensor << "\n";

        oss << "Graph operators:\n";
        for (const auto &op : ops)
        {
            vector<UidBaseType> preds, succs;
            for (auto &o : op->getPredecessors())
                preds.emplace_back(o->getGuid());
            for (auto &o : op->getSuccessors())
                succs.emplace_back(o->getGuid());
            oss << "OP " << op->getGuid();
            oss << ", pred " << vecToString(preds);
            oss << ", succ " << vecToString(succs);
            oss << ", " << op << "\n";
        }
        return oss.str();
    }

    bool GraphObj::topo_sort()
    {
        if (this->sorted)
        {
            return true;
        }
        std::vector<Operator> sorted;
        std::unordered_set<OperatorObj *> flags;
        sorted.reserve(ops.size());
        flags.reserve(ops.size());
        while (sorted.size() < ops.size())
        {
            // Any node is move to sorted in this loop.
            auto modified = false;
            for (auto const &op : ops)
            {
                if (auto const &inputs = op->getInputs();
                    flags.find(op.get()) == flags.end() &&
                    std::all_of(inputs.begin(), inputs.end(),
                                [&flags](auto const &input)
                                {
                                    auto ptr = input->getSource().get();
                                    return !ptr || flags.find(ptr) != flags.end();
                                }))
                {
                    modified = true;
                    sorted.emplace_back(op);
                    flags.insert(op.get());
                }
            }
            if (!modified)
            {
                return false;
            }
        }
        this->ops = std::move(sorted);
        return this->sorted = true;
    }

    Shape permutation(const Shape& x, const Shape& y){
        Shape z;
        for (auto i : y) {
            z.push_back(x[i]);
        }
        return z;
    }

    void GraphObj::optimize()
    {
        // =================================== 作业 ===================================
        // TODO: 设计一个算法来实现指定的图优化规则
        // 图优化规则如下：
        // 1. 去除冗余的算子（例如，两个相邻的算子都是 transpose 算子，且做的是相反的操作，可以将其全部删除）
        // 2. 合并算子（例如，矩阵乘算子中含有属性transA、transB，如果其输入存在transpose，且对最后两个维度做交换，就可以将transpose融入到矩阵乘算子的属性中去）
        // =================================== 作业 ===================================
        if(!sorted)
            topo_sort();
        for (auto p = ops.begin();p < ops.end();p++) {
            auto op = *p;
            if (op->getOpType() == OpType::Transpose) {
                Operator succ = op->getSuccessors()[0];
                if(succ->getOpType() == OpType::Transpose) {
                    Shape p1 = std::dynamic_pointer_cast<TransposeObj>(op)->getPermute();
                    Shape p2 = std::dynamic_pointer_cast<TransposeObj>(succ)->getPermute();
                    if (p1 == p2) {
                        Tensor input = op->getInputs()[0];
                        input->removeTarget(op);
                        Tensor output = succ->getOutput();
                        for (auto n : output->getTargets()) {
                            input->addTarget(n);
                            n->removePredecessors(succ);
                            if (auto source = input->getSource(); source) {
                                n->addPredecessors(source);
                                source->removeSuccessors(op);
                                source->addSuccessors(n);
                            }
                            n->replaceInput(output, input);
                        }
                        removeTensor(op->getOutput());
                        removeTensor(output);
                        removeOperator(succ);
                        removeOperator(op);
                    }
                }
            }
        }
        for (auto p = ops.begin();p < ops.end();p++) {
            auto op = *p;
            if (op->getOpType() == OpType::MatMul) {
                Operator pred = op->getPredecessors()[0];
                if (pred->getOpType() == OpType::Transpose) {
                    Shape p = std::dynamic_pointer_cast<TransposeObj>(pred)->getPermute();
                    Tensor output = pred->getOutput();
                    auto matmul = std::dynamic_pointer_cast<MatmulObj>(op);
                    if (matmul->getInputs()[0] == output && !matmul->getTransA()) {
                        std::swap(p[p.size()-1], p[p.size()-2]);
                        matmul->setTransA(true);
                    }
                    if (matmul->getInputs()[1] == output && !matmul->getTransB()) {
                        std::swap(p[p.size()-1], p[p.size()-2]);
                        matmul->setTransB(true);
                    }
                    int i = 0, permSize = static_cast<int>(p.size());
                    for (;i < permSize;i++) {
                        if (p[i] != i){
                            break;
                        }
                    }
                    if (i == permSize) {
                        int j = matmul->getInputs()[0] == output ? 0 : 1;
                        Tensor input = pred->getInputs()[0];
                        auto source = input->getSource();
                        if (source) {
                            source->removeSuccessors(pred);
                            source->addSuccessors(matmul);
                            op->addPredecessors(source);
                        }
                        input->removeTarget(pred);
                        input->addTarget(op);
                        op->removePredecessors(pred);
                        op->inputs[j] = input;
                        removeOperator(pred);
                        removeTensor(output);
                    }
                }
            }
        }
    }

    Tensor GraphObj::getTensor(int fuid) const
    {
        for (auto tensor : tensors)
        {
            if (tensor->getFuid() == fuid)
            {
                return tensor;
            }
        }
        return nullptr;
    }

    void GraphObj::shape_infer()
    {
        for (auto &op : ops)
        {
            auto ans = op->inferShape();
            IT_ASSERT(ans.has_value());
            auto oldOutputs = op->getOutputs();
            IT_ASSERT(ans.value().size() == oldOutputs.size());
            // replace the old outputshape and size with new one
            for (int i = 0; i < (int)ans.value().size(); ++i)
            {
                auto newShape = ans.value()[i];
                auto oldShape = oldOutputs[i]->getDims();
                auto fuid = oldOutputs[i]->getFuid();
                if (newShape != oldShape)
                {
                    auto tensor = this->getTensor(fuid);
                    tensor->setShape(newShape);
                }
            }
        }
    }

    void GraphObj::dataMalloc()
    {
        // topological sorting first
        IT_ASSERT(topo_sort() == true);

        // =================================== 作业 ===================================
        // TODO：利用 allocator 给计算图分配内存
        // HINT: 获取分配好的内存指针后，可以调用 tensor 的 setDataBlob 函数给 tensor 绑定内存
        // =================================== 作业 ===================================

        vector<size_t> sizes;
        for(auto tensor : tensors){
            sizes.push_back(allocator.alloc(tensor->getBytes()));
        }
        auto p = (std::byte*)allocator.getPtr();
        for(size_t i = 0;i < tensors.size(); i++){
            tensors[i]->setDataBlob(make_ref<BlobObj>(runtime, p+sizes[i]));
        }

        allocator.info();
    }

    Tensor GraphObj::addTensor(Shape dim, DataType dtype)
    {
        return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
    }

    Tensor GraphObj::addTensor(const Tensor &tensor)
    {
        IT_ASSERT(tensor->getRuntime() == runtime,
                  std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                      tensor->getRuntime()->toString() + " to " +
                      runtime->toString());
        tensors.emplace_back(tensor);
        return tensor;
    }

    TensorVec GraphObj::addTensor(const TensorVec &tensors)
    {
        for (auto &t : tensors)
            addTensor(t);
        return tensors;
    }

    // tensor's "source" and "target" must be in "ops".
    // tensor has no "source" and no "target" must not exist.
    // "inputs" or "outputs" of operators must be in "tensors"
    // "predecessors" and "successors" of an operator of "ops" must be in "ops".
    bool GraphObj::checkValid() const
    {
        for (auto tensor : tensors)
        {
            IT_ASSERT(!(tensor->getTargets().size() == 0 &&
                        nullptr == tensor->getSource()));
            for (auto op : tensor->getTargets())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
            }
            auto op = tensor->getSource();
            IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
        }
        for (auto op : ops)
        {
            for (auto tensor : op->getInputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto tensor : op->getOutputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto pre : op->getPredecessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
            }
            for (auto suc : op->getSuccessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
            }
        }
        std::set<UidBaseType> s;
        // check whether two tensors with the same FUID exist
        for (auto tensor : tensors)
        {
            int cnt = s.count(tensor->getFuid());
            IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
            s.insert(tensor->getFuid());
        }
        return true;
    }

} // namespace infini