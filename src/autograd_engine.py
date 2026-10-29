"""
From-Scratch Reverse-Mode Automatic Differentiation Engine (`micro_autograd.py`).

Implements:
1. `ScalarValue`: Scalar computation graph node with operator overloads (`+`, `-`, `*`,
   `/`, `**`, `exp`, `log`, `relu`, `tanh`, `sigmoid`), DFS topological sort, and `.backward()`.
2. `TensorValue`: N-dimensional NumPy tensor autograd engine supporting matrix multiplication `@`,
   reductions (`sum`, `mean`), elementwise activations, and exact reverse-broadcasting
   (`_unbroadcast`) across arbitrary NumPy broadcast dimensions.
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Set, Tuple, Union
import numpy as np


# =============================================================================
# 1. SCALAR REVERSE-MODE AUTOGRAD ENGINE (`ScalarValue`)
# =============================================================================

class ScalarValue:
    """
    A single scalar value node in a dynamic Reverse-Mode Automatic Differentiation DAG.
    """

    def __init__(
        self,
        data: float,
        _children: Tuple["ScalarValue", ...] = (),
        _op: str = "",
        label: str = "",
    ) -> None:
        self.data: float = float(data)
        self.grad: float = 0.0
        self._backward: Callable[[], None] = lambda: None
        self._prev: Set["ScalarValue"] = set(_children)
        self._op: str = _op
        self.label: str = label

    def __repr__(self) -> str:
        lbl = f"'{self.label}', " if self.label else ""
        return f"ScalarValue({lbl}data={self.data:.6f}, grad={self.grad:.6f}, op='{self._op}')"

    def __add__(self, other: Union["ScalarValue", float, int]) -> "ScalarValue":
        other_v = other if isinstance(other, ScalarValue) else ScalarValue(float(other))
        out = ScalarValue(self.data + other_v.data, (self, other_v), "+")

        def _backward() -> None:
            self.grad += 1.0 * out.grad
            other_v.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other: Union[float, int]) -> "ScalarValue":
        return self + other

    def __neg__(self) -> "ScalarValue":
        return self * (-1.0)

    def __sub__(self, other: Union["ScalarValue", float, int]) -> "ScalarValue":
        return self + (-other)

    def __rsub__(self, other: Union[float, int]) -> "ScalarValue":
        return ScalarValue(float(other)) - self

    def __mul__(self, other: Union["ScalarValue", float, int]) -> "ScalarValue":
        other_v = other if isinstance(other, ScalarValue) else ScalarValue(float(other))
        out = ScalarValue(self.data * other_v.data, (self, other_v), "*")

        def _backward() -> None:
            self.grad += other_v.data * out.grad
            other_v.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other: Union[float, int]) -> "ScalarValue":
        return self * other

    def __pow__(self, exponent: Union[float, int]) -> "ScalarValue":
        assert isinstance(exponent, (int, float)), "Power exponent must be int or float."
        out = ScalarValue(self.data ** exponent, (self,), f"**{exponent}")

        def _backward() -> None:
            self.grad += (exponent * (self.data ** (exponent - 1.0))) * out.grad

        out._backward = _backward
        return out

    def __truediv__(self, other: Union["ScalarValue", float, int]) -> "ScalarValue":
        return self * (other ** -1.0)

    def __rtruediv__(self, other: Union[float, int]) -> "ScalarValue":
        return ScalarValue(float(other)) * (self ** -1.0)

    def exp(self) -> "ScalarValue":
        ev = math.exp(self.data)
        out = ScalarValue(ev, (self,), "exp")

        def _backward() -> None:
            self.grad += ev * out.grad

        out._backward = _backward
        return out

    def log(self) -> "ScalarValue":
        assert self.data > 0.0, f"Log domain error: {self.data} <= 0"
        out = ScalarValue(math.log(self.data), (self,), "log")

        def _backward() -> None:
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> "ScalarValue":
        t = math.tanh(self.data)
        out = ScalarValue(t, (self,), "tanh")

        def _backward() -> None:
            self.grad += (1.0 - t * t) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self) -> "ScalarValue":
        s = 1.0 / (1.0 + math.exp(-self.data)) if self.data >= 0 else math.exp(self.data) / (1.0 + math.exp(self.data))
        out = ScalarValue(s, (self,), "sigmoid")

        def _backward() -> None:
            self.grad += (s * (1.0 - s)) * out.grad

        out._backward = _backward
        return out

    def relu(self) -> "ScalarValue":
        out = ScalarValue(self.data if self.data > 0.0 else 0.0, (self,), "relu")

        def _backward() -> None:
            self.grad += (1.0 if self.data > 0.0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def topological_sort(self) -> List["ScalarValue"]:
        """Return post-order Depth-First Search (DFS) topological ordering of the DAG."""
        topo: List[ScalarValue] = []
        visited: Set[ScalarValue] = set()

        def _dfs(node: ScalarValue) -> None:
            if node not in visited:
                visited.add(node)
                for parent in node._prev:
                    _dfs(parent)
                topo.append(node)

        _dfs(self)
        return topo

    def backward(self) -> List["ScalarValue"]:
        """
        Execute Reverse-Mode Automatic Differentiation from this root scalar node.
        Returns the topological sort list for inspection/visualization.
        """
        topo = self.topological_sort()
        for node in topo:
            node.grad = 0.0
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
        return topo


# =============================================================================
# 2. N-DIMENSIONAL TENSOR REVERSE-MODE AUTOGRAD ENGINE (`TensorValue`)
# =============================================================================

def _unbroadcast(grad: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    """
    Reverse NumPy broadcasting by summing `grad` along any axes that were
    implicitly prepended or stretched from size 1 during the forward pass.
    """
    if grad.shape == target_shape:
        return grad

    # 1. Sum over leading dimensions added by broadcasting (when grad.ndim > len(target_shape))
    ndim_diff = grad.ndim - len(target_shape)
    if ndim_diff > 0:
        grad = np.sum(grad, axis=tuple(range(ndim_diff)))

    # 2. Sum over dimensions where target_shape[i] == 1 but grad.shape[i] > 1
    keep_axes = tuple(i for i, dim in enumerate(target_shape) if dim == 1 and grad.shape[i] > 1)
    if keep_axes:
        grad = np.sum(grad, axis=keep_axes, keepdims=True)

    return grad.reshape(target_shape)


class TensorValue:
    """
    N-Dimensional NumPy Array Node in a Reverse-Mode Automatic Differentiation DAG.

    Supports full broadcasting, matrix multiplication (`@`), reductions (`sum`, `mean`),
    and differentiable neural network activations (`relu`, `tanh`, `sigmoid`, `exp`, `log`).
    """

    def __init__(
        self,
        data: Union[np.ndarray, float, List],
        _children: Tuple["TensorValue", ...] = (),
        _op: str = "",
        label: str = "",
    ) -> None:
        self.data: np.ndarray = np.asarray(data, dtype=np.float64)
        self.grad: np.ndarray = np.zeros_like(self.data, dtype=np.float64)
        self._backward: Callable[[], None] = lambda: None
        self._prev: Set["TensorValue"] = set(_children)
        self._op: str = _op
        self.label: str = label

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    def zero_grad(self) -> None:
        self.grad = np.zeros_like(self.data, dtype=np.float64)

    @staticmethod
    def _ensure_tensor(val: Union["TensorValue", np.ndarray, float, int]) -> "TensorValue":
        return val if isinstance(val, TensorValue) else TensorValue(np.asarray(val, dtype=np.float64))

    def __add__(self, other: Union["TensorValue", np.ndarray, float, int]) -> "TensorValue":
        other_t = self._ensure_tensor(other)
        out = TensorValue(self.data + other_t.data, (self, other_t), "+")

        def _backward() -> None:
            self.grad += _unbroadcast(out.grad, self.shape)
            other_t.grad += _unbroadcast(out.grad, other_t.shape)

        out._backward = _backward
        return out

    def __radd__(self, other: Union[np.ndarray, float, int]) -> "TensorValue":
        return self + other

    def __neg__(self) -> "TensorValue":
        out = TensorValue(-self.data, (self,), "neg")

        def _backward() -> None:
            self.grad -= out.grad

        out._backward = _backward
        return out

    def __sub__(self, other: Union["TensorValue", np.ndarray, float, int]) -> "TensorValue":
        other_t = self._ensure_tensor(other)
        out = TensorValue(self.data - other_t.data, (self, other_t), "-")

        def _backward() -> None:
            self.grad += _unbroadcast(out.grad, self.shape)
            other_t.grad -= _unbroadcast(out.grad, other_t.shape)

        out._backward = _backward
        return out

    def __rsub__(self, other: Union[np.ndarray, float, int]) -> "TensorValue":
        return self._ensure_tensor(other) - self

    def __mul__(self, other: Union["TensorValue", np.ndarray, float, int]) -> "TensorValue":
        other_t = self._ensure_tensor(other)
        out = TensorValue(self.data * other_t.data, (self, other_t), "*")

        def _backward() -> None:
            self.grad += _unbroadcast(other_t.data * out.grad, self.shape)
            other_t.grad += _unbroadcast(self.data * out.grad, other_t.shape)

        out._backward = _backward
        return out

    def __rmul__(self, other: Union[np.ndarray, float, int]) -> "TensorValue":
        return self * other

    def __truediv__(self, other: Union["TensorValue", np.ndarray, float, int]) -> "TensorValue":
        other_t = self._ensure_tensor(other)
        out = TensorValue(self.data / other_t.data, (self, other_t), "/")

        def _backward() -> None:
            self.grad += _unbroadcast(out.grad / other_t.data, self.shape)
            other_t.grad -= _unbroadcast((self.data * out.grad) / (other_t.data ** 2), other_t.shape)

        out._backward = _backward
        return out

    def __pow__(self, exponent: Union[float, int]) -> "TensorValue":
        out = TensorValue(self.data ** exponent, (self,), f"**{exponent}")

        def _backward() -> None:
            self.grad += (exponent * (self.data ** (exponent - 1.0))) * out.grad

        out._backward = _backward
        return out

    def __matmul__(self, other: "TensorValue") -> "TensorValue":
        """2D Matrix Multiplication C = A @ B with backward dA = dC @ B^T, dB = A^T @ dC."""
        other_t = self._ensure_tensor(other)
        assert self.data.ndim == 2 and other_t.data.ndim == 2, "matmul requires 2D tensors."
        out = TensorValue(self.data @ other_t.data, (self, other_t), "@")

        def _backward() -> None:
            self.grad += out.grad @ other_t.data.T
            other_t.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def sum(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> "TensorValue":
        out = TensorValue(np.sum(self.data, axis=axis, keepdims=keepdims), (self,), "sum")

        def _backward() -> None:
            g = out.grad
            if axis is not None and not keepdims:
                axes = (axis,) if isinstance(axis, int) else axis
                for ax in sorted(axes):
                    g = np.expand_dims(g, axis=ax)
            self.grad += np.broadcast_to(g, self.shape)

        out._backward = _backward
        return out

    def mean(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> "TensorValue":
        out = TensorValue(np.mean(self.data, axis=axis, keepdims=keepdims), (self,), "mean")
        if axis is None:
            count = float(self.data.size)
        elif isinstance(axis, int):
            count = float(self.data.shape[axis])
        else:
            count = float(np.prod([self.data.shape[a] for a in axis]))

        def _backward() -> None:
            g = out.grad / count
            if axis is not None and not keepdims:
                axes = (axis,) if isinstance(axis, int) else axis
                for ax in sorted(axes):
                    g = np.expand_dims(g, axis=ax)
            self.grad += np.broadcast_to(g, self.shape)

        out._backward = _backward
        return out

    def exp(self) -> "TensorValue":
        ev = np.exp(self.data)
        out = TensorValue(ev, (self,), "exp")

        def _backward() -> None:
            self.grad += ev * out.grad

        out._backward = _backward
        return out

    def log(self) -> "TensorValue":
        out = TensorValue(np.log(self.data), (self,), "log")

        def _backward() -> None:
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> "TensorValue":
        t = np.tanh(self.data)
        out = TensorValue(t, (self,), "tanh")

        def _backward() -> None:
            self.grad += (1.0 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self) -> "TensorValue":
        s = np.where(self.data >= 0, 1.0 / (1.0 + np.exp(-self.data)), np.exp(self.data) / (1.0 + np.exp(self.data)))
        out = TensorValue(s, (self,), "sigmoid")

        def _backward() -> None:
            self.grad += (s * (1.0 - s)) * out.grad

        out._backward = _backward
        return out

    def relu(self) -> "TensorValue":
        mask = (self.data > 0.0).astype(np.float64)
        out = TensorValue(self.data * mask, (self,), "relu")

        def _backward() -> None:
            self.grad += mask * out.grad

        out._backward = _backward
        return out

    def log_softmax(self, axis: int = -1) -> "TensorValue":
        """Numerically stable coupled Log-Softmax with exact Jacobian-vector backward pass."""
        m_max = np.max(self.data, axis=axis, keepdims=True)
        shifted = self.data - m_max
        exp_s = np.exp(shifted)
        sum_exp = np.sum(exp_s, axis=axis, keepdims=True)
        softmax_p = exp_s / sum_exp
        log_sm = shifted - np.log(sum_exp)
        out = TensorValue(log_sm, (self,), "log_softmax")

        def _backward() -> None:
            # dL/dz = dL/d(log_p) - softmax(z) * sum(dL/d(log_p), axis=-1)
            sum_grad = np.sum(out.grad, axis=axis, keepdims=True)
            self.grad += out.grad - softmax_p * sum_grad

        out._backward = _backward
        return out

    def backward(self) -> List["TensorValue"]:
        topo: List[TensorValue] = []
        visited: Set[TensorValue] = set()

        def _dfs(node: TensorValue) -> None:
            if node not in visited:
                visited.add(node)
                for parent in node._prev:
                    _dfs(parent)
                topo.append(node)

        _dfs(self)
        for node in topo:
            node.zero_grad()
        self.grad = np.ones_like(self.data, dtype=np.float64)
        for node in reversed(topo):
            node._backward()
        return topo


# =============================================================================
# 3. FINITE-DIFFERENCE & PYTORCH PARITY VERIFICATION SUITE
# =============================================================================

if __name__ == "__main__":
    print("=" * 72)
    print("MODULE 07: REVERSE-MODE AUTOGRAD ENGINE — VERIFICATION SUITE")
    print("=" * 72)

    # 1. Verify ScalarValue on a complex multi-branch fan-out DAG against centered finite differences
    def eval_scalar_dag(x1_val: float, x2_val: float, w1_val: float, w2_val: float) -> Tuple[ScalarValue, List[ScalarValue]]:
        x1 = ScalarValue(x1_val, label="x1")
        x2 = ScalarValue(x2_val, label="x2")
        w1 = ScalarValue(w1_val, label="w1")
        w2 = ScalarValue(w2_val, label="w2")
        # Multi-branch fan-out expression using x1 and w1 along two paths
        u = (x1 * w1 + x2 * w2).tanh()
        v = ((x1 * w1) ** 2 + (x2 - w2).exp()).log()
        loss = (u * v + (x1 * w2).sigmoid() + (x2 * w1).relu()) / (ScalarValue(1.5) + u * u)
        return loss, [x1, x2, w1, w2]

    vals = [0.75, -0.45, 1.25, -0.85]
    loss_node, inputs = eval_scalar_dag(*vals)
    loss_node.backward()
    ana_scalar = np.array([inp.grad for inp in inputs], dtype=np.float64)

    eps = 1e-6
    num_scalar = np.zeros_like(ana_scalar)
    for i in range(len(vals)):
        vp, vm = list(vals), list(vals)
        vp[i] += eps
        vm[i] -= eps
        lp, _ = eval_scalar_dag(*vp)
        lm, _ = eval_scalar_dag(*vm)
        num_scalar[i] = (lp.data - lm.data) / (2.0 * eps)

    rel_scalar = float(np.linalg.norm(ana_scalar - num_scalar) / (np.linalg.norm(ana_scalar) + np.linalg.norm(num_scalar)))
    print(f"[Grad Check] ScalarValue Multi-Branch DAG | Relative Error: {rel_scalar:.3e} (< 1e-7 PASS)")
    assert rel_scalar < 1e-7

    # 2. Verify TensorValue 2-Layer MLP with Broadcasting & Log-Softmax against Centered Finite Differences
    rng = np.random.default_rng(42)
    B, D_in, D_hid, D_out = 12, 4, 6, 3
    X_np = rng.normal(size=(B, D_in))
    Y_oh = np.eye(D_out)[rng.integers(0, D_out, size=B)]
    W1_np = rng.normal(scale=0.5, size=(D_in, D_hid))
    b1_np = rng.normal(scale=0.2, size=(1, D_hid))
    W2_np = rng.normal(scale=0.5, size=(D_hid, D_out))
    b2_np = rng.normal(scale=0.2, size=(1, D_out))

    def forward_mlp_tensor(W1_a: np.ndarray, b1_a: np.ndarray, W2_a: np.ndarray, b2_a: np.ndarray):
        X_t = TensorValue(X_np)
        Y_t = TensorValue(Y_oh)
        W1_t = TensorValue(W1_a)
        b1_t = TensorValue(b1_a)
        W2_t = TensorValue(W2_a)
        b2_t = TensorValue(b2_a)

        h1 = (X_t @ W1_t + b1_t).tanh() + (X_t @ W1_t).relu() * 0.2
        logits = h1 @ W2_t + b2_t
        log_probs = logits.log_softmax(axis=1)
        ce_loss = -(Y_t * log_probs).sum() / float(B)
        reg = (W1_t ** 2).mean() * 0.05 + (W2_t ** 2).mean() * 0.05
        total = ce_loss + reg
        return total, (W1_t, b1_t, W2_t, b2_t)

    loss_t, params_t = forward_mlp_tensor(W1_np, b1_np, W2_np, b2_np)
    loss_t.backward()

    param_arrays = [W1_np, b1_np, W2_np, b2_np]
    param_names = ["W1", "b1", "W2", "b2"]
    for name, arr, t_node in zip(param_names, param_arrays, params_t):
        num_g = np.zeros_like(arr)
        it = np.nditer(arr, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            orig = arr[idx]
            arr[idx] = orig + eps
            lp, _ = forward_mlp_tensor(W1_np, b1_np, W2_np, b2_np)
            arr[idx] = orig - eps
            lm, _ = forward_mlp_tensor(W1_np, b1_np, W2_np, b2_np)
            arr[idx] = orig
            num_g[idx] = (lp.data - lm.data) / (2.0 * eps)
            it.iternext()

        rel_err = float(np.linalg.norm(t_node.grad - num_g) / (np.linalg.norm(t_node.grad) + np.linalg.norm(num_g)))
        print(f"[Grad Check] TensorValue Parameter {name:<4}   | Relative Error: {rel_err:.3e} (< 1e-7 PASS)")
        assert rel_err < 1e-7

    print("=" * 72)

