#include "operators/matmul.h"
#include <algorithm>
#include <cstddef>

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // =================================== 作业 ===================================
        auto A = inputs[0]->getDims();
        auto B = inputs[1]->getDims();
        auto size = A.size();
        m = transA ? A[size-1] : A[size-2];
        n = transB ? B[size-2] : B[size-1];
        k = transA ? A[size-2] : A[size-1];
        Shape output;
        for(size_t i = 0;i < size-2;i++){
            output.push_back(std::max(A[i], B[i]));
        }
        output.push_back(m);
        output.push_back(n);
        return vector({output});
    }

} // namespace infini