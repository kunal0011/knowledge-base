# Domain I: NumPy — High-Performance Numerical Computing & Strided Buffers

[![NumPy Version](https://img.shields.io/badge/NumPy-2.x%20Ready-blue.svg)](https://numpy.org/)
[![Foundational Book](https://img.shields.io/badge/Canonical%20Book-Guide%20to%20NumPy%20(Oliphant)-darkblue.svg)](https://archive.org/details/GuideToNumPy)
[![Secondary Book](https://img.shields.io/badge/Companion%20Book-Python%20for%20Data%20Analysis%20(McKinney)-green.svg)](https://wesmckinney.com/book/)

> *"NumPy is the fundamental package for scientific computing in Python. It provides a multidimensional array object, various derived objects (such as masked arrays and matrices), and an assortment of routines for fast operations on arrays, including mathematical, logical, shape manipulation, sorting, selecting, I/O, discrete Fourier transforms, basic linear algebra, basic statistical operations, random simulation and much more."*  
> — **Travis Oliphant**, Creator of NumPy & author of *Guide to NumPy*

---

## 🏛️ NumPy Architectural Philosophy & Memory Blueprint

At the heart of NumPy is the **`ndarray` (N-dimensional array)**. Unlike standard Python lists—which are arrays of pointers to disjointed heap-allocated `PyObject` wrappers—a NumPy array encapsulates a contiguous segment of raw, typed memory paired with rich structural metadata.

```
                    Python List of Integers (Boxed PyObjects)
                    +-----+-----+-----+-----+
List Object:        | ptr | ptr | ptr | ptr |
                    +--+--+--+--+--+--+--+--+
                       |     |     |     |
                       v     v     v     v
                     +---+ +---+ +---+ +---+
                     | 1 | | 2 | | 3 | | 4 |  (Scattered in RAM; cache misses)
                     +---+ +---+ +---+ +---+

                    NumPy ndarray (Contiguous Memory Buffer)
                    +-------------------------------------------------------+
ndarray Object:     | data_ptr | shape: (2, 2) | strides: (16, 8) | dtype   |
                    +----+--------------------------------------------------+
                         |
                         v
                     +-------+-------+-------+-------+
Contiguous Buffer:   |   1   |   2   |   3   |   4   |  (Cache-line friendly, SIMD)
                     +-------+-------+-------+-------+
```

### The C-Structure of `PyArrayObject`
Internally, an `ndarray` is defined in C as:
1. **`char *data`**: Pointer to the first byte of raw memory.
2. **`int nd`**: Number of dimensions ($N$).
3. **`npy_intp *dimensions`** (Shape): An array of integers of length $N$ specifying the extent of each axis.
4. **`npy_intp *strides`**: An array of integers of length $N$ specifying how many bytes must be stepped in memory to advance by one element along each axis.
5. **`PyArray_Descr *descr`** (Dtype): Data type descriptor including size in bytes, alignment, byte order (endianness), and type kind.
6. **`int flags`**: Memory layout metadata (e.g., `C_CONTIGUOUS`, `F_CONTIGUOUS`, `WRITEABLE`, `OWNDATA`, `ALIGNED`).
7. **`PyObject *base`**: Reference to the owner of the memory buffer if this array is a view.

---

## 📚 Master Chapter Index

| Chapter | Title | Core Canonical Concepts & Focus | Canonical Literature Focus |
| :---: | :--- | :--- | :--- |
| **01** | [ndarray Architecture, Strides & Memory Layout](01.%20ndarray%20Architecture%2C%20Strides%20%26%20Memory%20Layout.md) | C vs Fortran contiguity, `as_strided`, stride manipulation, pointer arithmetic | *Guide to NumPy* Ch. 2 (*Array Concepts*) |
| **02** | [Data Types, Casting & Memory Footprint](02.%20Data%20Types%2C%20Casting%20%26%20Memory%20Footprint.md) | Scalar types, safe casting rules, endianness, precision loss, IEEE 754 | *Guide to NumPy* Ch. 3 (*Data-Type Descriptors*) |
| **03** | [Indexing, Slicing, Views vs Copies & Masking](03.%20Indexing%2C%20Slicing%2C%20Views%20vs%20Copies%20%26%20Masking.md) | Basic slicing (views) vs Advanced Indexing (copies), boolean masks, `.base` | *Python for Data Analysis* Ch. 4 |
| **04** | [Vectorization, Ufuncs & Mathematical Computations](04.%20Vectorization%2C%20Ufuncs%20%26%20Mathematical%20Computations.md) | Universal functions, `reduce`, `accumulate`, `outer`, `at`, SIMD vectorization | *Guide to NumPy* Ch. 4 (*Universal Functions*) |
| **05** | [Broadcasting Mechanics & Multidimensional Alignment](05.%20Broadcasting%20Mechanics%20%26%20Multidimensional%20Alignment.md) | 2-Rule formal algorithm, virtual dimension expansion, stride tricks | *Python for Data Analysis* Appendix A |
| **06** | [Linear Algebra, Matrix Decompositions & BLAS-LAPACK](06.%20Linear%20Algebra%2C%20Matrix%20Decompositions%20%26%20BLAS-LAPACK.md) | BLAS/LAPACK linkage, matrix products, Cholesky, QR, SVD, Eigen systems | *Guide to NumPy* Ch. 7 & SciPy Linear Algebra |
| **07** | [Array Manipulation, Reshaping & Structural Transformations](07.%20Array%20Manipulation%2C%20Reshaping%20%26%20Structural%20Transformations.md) | `reshape`, `ravel`, `flatten`, `transpose`, `swapaxes`, concatenation | *Python for Data Analysis* Ch. 4 |
| **08** | [Structured Arrays, Record Arrays & Binary Data](08.%20Structured%20Arrays%2C%20Record%20Arrays%20%26%20Binary%20Data.md) | Compound dtypes, field offsets, byte padding, `np.recarray`, raw I/O | *Guide to NumPy* Ch. 3.3 |
| **09** | [Memory-Mapped Files (memmap) & Out-of-Core Processing](09.%20Memory-Mapped%20Files%20%28memmap%29%20%26%20Out-of-Core%20Processing.md) | `np.memmap`, OS page cache, virtual memory, processing large datasets | *Python for Data Analysis* Ch. 4 & OS System Calls |
| **10** | [Random Number Generation, BitGenerators & Numerical Stability](10.%20Random%20Number%20Generation%2C%20BitGenerators%20%26%20Numerical%20Stability.md) | `Generator` vs Legacy `RandomState`, PCG64, SeedSequence, floating-point error | *Guide to NumPy* & NumPy 1.17+ RNG Architecture |

---

## ⚡ Quick Reference: High-Performance NumPy Idioms

### 1. Zero-Copy View Detection
```python
import numpy as np

arr = np.arange(10, dtype=np.int64)
view = arr[::2]       # Basic slicing -> Returns a VIEW (zero memory allocation)
copy = arr[[0, 2, 4]] # Advanced indexing -> Returns a COPY (allocates new RAM)

assert view.base is arr
assert copy.base is None
assert np.shares_memory(arr, view) is True
assert np.shares_memory(arr, copy) is False
```

### 2. The 2-Rule Broadcasting Algorithm
When operating on two arrays, NumPy compares their shapes element-wise, starting from the **trailing (rightmost) dimensions** and working leftward:
1. Two dimensions are compatible if they are **equal**, or
2. One of the dimensions is **$1$**.
If one dimension is $1$, that dimension is virtually stretched along its axis by setting its stride to $0$, consuming zero additional memory.

```python
A = np.ones((8, 1, 6, 1))
B = np.ones((7, 1, 5))
# Trailing alignment:
# A:  8 x 1 x 6 x 1
# B:  1 x 7 x 1 x 5   (padded with leading 1)
# Result: 8 x 7 x 6 x 5
```

### 3. In-Place Universal Function Accumulation
Avoid allocating intermediate temporary arrays during large reductions:
```python
x = np.ones(10_000_000, dtype=np.float64)
out = np.empty(10_000_000, dtype=np.float64)

# In-place accumulation directly into pre-allocated memory:
np.add.accumulate(x, out=out)
```
