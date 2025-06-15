torch.compile рекурсивно проходит все pytorch модули и функции, чтобы сформировать граф вычислений, для кодогенерации

Код для отладки graph breaks. Он хотя бы дает понять где сломалось, откуда можно начать.
```python
import torch

def bar(a, b):
    x = a / (torch.abs(a) + 1)
    if b.sum() < 0:
        b = b * -1
    return x * b

opt_bar = torch.compile(bar, fullgraph=True)
try:
    opt_bar(torch.randn(10), torch.randn(10))
except:
    tb.print_exc()
```
```text
class GraphModule(torch.nn.Module):
    def forward(self, L_a_: "f32[10][1]cpu", L_b_: "f32[10][1]cpu"):
        l_a_ = L_a_
        l_b_ = L_b_
        
         # File: /tmp/ipykernel_342722/966290996.py:2 in bar, code: x = a / (torch.abs(a) + 1)
        abs_1: "f32[10][1]cpu" = torch.abs(l_a_)
        add: "f32[10][1]cpu" = abs_1 + 1;  abs_1 = None
        x: "f32[10][1]cpu" = l_a_ / add;  l_a_ = add = x = None
        
         # File: /tmp/ipykernel_342722/966290996.py:3 in bar, code: if b.sum() < 0:
        sum_1: "f32[][]cpu" = l_b_.sum();  l_b_ = None
        lt: "b8[][]cpu" = sum_1 < 0;  sum_1 = lt = None
        
Traceback (most recent call last):
  File "/tmp/ipykernel_342722/3610564610.py", line 3, in <module>
    opt_bar(torch.randn(10), torch.randn(10))
  File "/opt/conda/lib/python3.11/site-packages/torch/_dynamo/eval_frame.py", line 659, in _fn
    raise e.with_traceback(None) from None
torch._dynamo.exc.Unsupported: Data-dependent branching
  Explanation: Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). Dynamo does not support tracing dynamic control flow.
  Hint: This graph break is fundamental - it is unlikely that Dynamo will ever be able to trace through your code. Consider finding a workaround.
  Hint: Use `torch.cond` to express dynamic control flow.

  Developer debug context: attempted to jump with TensorVariable()


from user code:
   File "/tmp/ipykernel_342722/966290996.py", line 3, in bar
    if b.sum() < 0:

Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"
```

### Numerics divergence from eager
- https://blog.ezyang.com/2024/11/ways-to-use-torch-compile/
Numerics divergence from eager. Unfortunately, the compiler does not guarantee exact bitwise equivalence with eager code; we reserve the right to do things like select different matrix multiply algorithms with different numerics or eliminate unnecessary downcast/upcasts when fusing half precision compute together. The compiler is also complicated and can have bugs that can cause loss not to converge. Expect to also have to evaluate whether or not application of torch.compile affects accuracy. Fortunately, for most uses of compiler for training efficiency, the baseline is the eager model, so you can just run an ablation to figure out who is actually causing the accuracy problem. (This won't be true in a later use case when the compiler is load bearing, see below!)


### Should you even expect your program to compile?
- https://docs.google.com/document/d/1y5CRfMLdwEoF1nTk9q8qEu1mgMUuUtvhklPKJ2emLU8/edit?tab=t.0#heading=h.fljwtnsw69i8
At a high level, you should only expect programs that correspond to a fixed sequence of PyTorch tensor operations to compile in a useful way. Importantly, this sequence of tensor operations must stay the same from run to run. This is different from TorchScript, where if you made your code TorchScriptable, you had some access to nontrivial Python language features such as loops and Python data structures like lists, which you could reasonably expect to be captured by the compiler and execute in the runtime. In PT2, our expectation is that those parts of the program are just run by the normal CPython interpreter, and torch.compile is used on the “tensor compute.” Even simple things like indexing into a Python list by a dynamic index, or taking a list of Tensors which may vary in size, do not work with PT2. If this is a sticking point for your model, you should perhaps consider using something like Mojo.

### Printing things out at compile time
- https://docs.google.com/document/d/1y5CRfMLdwEoF1nTk9q8qEu1mgMUuUtvhklPKJ2emLU8/edit?tab=t.0#heading=h.2pn0ke53ckbb
Normally, if you add a print statement to compiled code, this will cause a graph break.

### CUDA graphs advice
Smaller graphs are more likely to exhibit dynamic behavior (seeing different sizes), and in general dynamic shapes codegen is less performant than static shapes codegen.

By default, if you have any data-dependent computation, e.g., boolean masking, item() call, nonzero(), etc, this will trigger a graph break. If you are feeling brave, you can get past these problems by setting torch._dynamo.config.capture_scalar_outputs = True and torch._dynamo.config.capture_dynamic_output_shape_ops = True.

CUDA graphs don’t support dynamic shapes. We actually do support mode=”reduce-overhead” in conjunction with dynamic=True, but the way this is implemented is by recording a distinct CUDA graph for each size you see. If this is too many CUDA graphs, you will want to pad sizes to multiples to reduce the number of CUDA graphs you need to record. See also notes at Compiled results are slow on first run

### RoPE implementation
- https://pytorch.org/blog/maximizing-training-throughput/
RoPE implementation uses complex numbers, which was not supported in torch.compile at the time of testing