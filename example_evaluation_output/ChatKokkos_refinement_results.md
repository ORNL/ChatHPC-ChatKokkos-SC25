# ChatHPC JSON to Markdown

## Index 0

### Context

OpenMP to Kokkos translation


### Question

Can you translate this OpenACC code to Kokkos?
#pragma acc parallel loop
for (i = 0, i < n, i++) {
  y[i] = a * x[i] + y[i];
}


### Response

Kokkos::parallel_for("saxpy", Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(const int i) {
  y(i) = a * x(i) + y(i);
});

## Index 1

### Context

OpenMP to Kokkos translation


### Question

Can you translate this OpenACC code to Kokkos?
#pragma acc parallel loop reduction(+:sum)
for (i = 0, i < n, i++) {
  sum += x[i] * y[i];
}


### Response

Kokkos::parallel_reduce("sdot", Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(const int i, float &update) {
  update = x(i) * y(i);
}, Kokkos::Sum<float>(sum));

## Index 2

### Context

OpenMP to Kokkos translation


### Question

Can you translate this OpenACC code to Kokkos?
#pragma acc parallel loop private(i, j, sum)
for (i = 0; i < n; ++i) {
  sum = 0.0;
  for (j = 0; j < n; ++j) {
    sum += A[i][j] * x[j];
}


### Response

Kokkos::parallel_for("sgemv", Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(const int i) {
  float sum = 0.0f;
  for (int j = 0; j < n; j++) {
    sum += A(i,j) * x(j);
  }
  y(i) = sum;
});

## Index 3

### Context

OpenMP to Kokkos translation


### Question

Can you translate this OpenACC code to Kokkos?
#pragma acc parallel loop collapse(2) private(i, j, l, sum)
for (i = 0; i < m; ++i) {
  for (j = 0; j < n; ++j) {
    sum = 0.0;
    for (int l = 0; l < k; l++) {
      sum += a[i * lda + l] * b[l * ldb + j];
    }
    c[i * ldc + j] = alpha * sum + beta * c[i * ldc + j];
  }
}


### Response

Kokkos::parallel_for("sgemm", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {n, m}), KOKKOS_LAMBDA (const int i, const int j) {
  float sum = 0.0f;
  for (int l = 0; l < k; l++) {
    sum += a(i,l) * b(l,j);
  }
  c(i, j) = alpha * sum + beta * c(i, j);
 });

## Index 4

### Context

OpenMP to Kokkos translation


### Question

Can you translate this OpenACC code to Kokkos?
#pragma acc parallel loop shared(n, nnz, val, row, col, x, y) private(i, j)
for (i = 0; i < n; i++) {
  y[i] = 0.0;
  for (j = row[i]; j < row[i + 1]; j++) {
    y[i] += val[j] * x[col[j]];
  }
}


### Response

Kokkos::parallel_for("spmv", Kokkos::RangePolicy(0, n), KOKKOS_LAMBDA(const int i) {
  y(i) = 0.0;
  for (int j = row(i); j < row(i + 1); ++j) {
    y(i) += val(j) * x(col(j));
  }
});

## Index 5

### Context

OpenMP to Kokkos translation


### Question

Can you translate this OpenACC code to Kokkos?
#pragma acc parallel loop shared(n, u, u_new) private(i)
for (i = 1; i < n-1; i++) {
  u_new[i] = 0.5 * (u[i-1] + u[i+1]) - u[i];
}


### Response

Kokkos::parallel_for("stencil", Kokkos::RangePolicy(1, n-1), KOKKOS_LAMBDA(const int i) {
  u_new(i) = 0.5 * (u(i-1) + u(i+1)) - u(i);
});

## Index 6

### Context

OpenMP to Kokkos translation


### Question

Can you translate this OpenMP code to Kokkos? 
#pragma omp parallel for shared(n, x, y) private(i)
for (i = 0, i < n, i++) {
  y[i] = a * x[i];
}


### Response

Kokkos::parallel_for("saxpy", Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(const int i) {
  y(i) = a * x(i);
});

## Index 7

### Context

OpenMP to Kokkos translation


### Question

Can you translate this OpenMP code to Kokkos?
#pragma omp parallel for shared(n, x, y) private(i, a)
for (i = 0, i < n, i++) {
  a = x[i];
  x[i] = y[i];
  y[i] = a;
}


### Response

Kokkos::parallel_for("sgemv", Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(const int i) {
  float sum = x(i) + y(i);
  x(i) = y(i);
  y(i) = sum;
});

## Index 8

### Context

OpenMP to Kokkos translation


### Question

Can you translate this OpenMP code to Kokkos?
#pragma omp parallel for shared(n, x) private(i, sum)
for (i = 0, i < n, i++) {
  sum += x[i] * x[i];
}


### Response

Kokkos::parallel_reduce("sdot", Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(const int i, float &update) {
  update = x(i) * x(i);
}, Kokkos::Sum<float>(sum));

## Index 9

### Context

OpenMP to Kokkos translation


### Question

Can you translate this OpenMP code to Kokkos?
#pragma omp parallel for shared(n, x, y) private(i) 
for (i = 0, i < n, i++) {
  y[i] = x[i];
}


### Response

Kokkos::parallel_for("saxpy", Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(const int i) {
  y(i) = x(i);
});

## Index 10

### Context

OpenMP to Kokkos translation


### Question

Can you translate this OpenMP code to Kokkos?
#pragma omp parallel for shared(n, x) private(i, sum)
for (i = 0, i < n, i++) {
  sum += abs(x[i]);
}


### Response

Kokkos::parallel_reduce("sum_absolute_value", Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(const int i, float &update) {
  update = std::abs(x(i));
}, Kokkos::Sum<float>(sum));

## Index 11

### Context

OpenMP to Kokkos translation


### Question

Can you translate this OpenMP code to Kokkos?
#pragma omp parallel for 
for (int col = 0; col < num_cols; ++col) {
  for (int i = col_ptr[col]; i < col_ptr[col + 1]; ++i) {
    y[row_ind[i]] += values[i] * x[col];
  }
}


### Response

Kokkos::parallel_for("spmv_csc", Kokkos::RangePolicy(0, num_cols), KOKKOS_LAMBDA(const int col) {
  for (int i = col_ptr(col); i < col_ptr(col + 1); ++i) {
    y(row_ind(i)) += values(i) * x(col);
  }
});

## Index 12

### Context

OpenMP to Kokkos translation


### Question

Can you translate this OpenMP code to Kokkos?
#pragma omp parallel for
for (int i = 0; i < non_zeros; i++) {
  y[row_ind[i]] += values[i] * x[col_ind[i]];
}


### Response

Kokkos::parallel_for("spmv", Kokkos::RangePolicy(0, non_zeros), KOKKOS_LAMBDA(const int i) {
  y(row_ind(i)) += values(i) * x(col_ind(i));
});

## Index 13

### Context

OpenMP to Kokkos translation


### Question

Can you translate this OpenMP code to Kokkos?
#pragma omp parallel for  
for (i = 1; i < n-1; i++) {
  for (j = 1; j < n-1; j++) {
    u_new[i][j] = (u[i-1][j] + u[i][j+1] + u[i][j-1] + u[i+1][j] + f[i][j]) * 0.25;
  }
}


### Response

Kokkos::parallel_for("stencil_2D", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {n-1, n-1}), KOKKOS_LAMBDA(const int i, const int j) {
  u_new(i,j) = (u(i-1,j) + u(i,j+1) + u(i,j-1 + u(i+1,j) + f(i,j)) * 0.25;
});

## Index 14

### Context

OpenMP to Kokkos translation


### Question

Can you translate this OpenMP code to Kokkos?
#pragma omp parallel for
for (i = 1; i < n-1; i++) {
  for (j = 1; j < n-1; j++) {
    for (k = 1; k < n-1; k++) {
      u_new [i][j][k] = 0.125 * (u[i-1][j][k] + u[i][j+1][k] + u[i][j-1][k] + u[i+1][j][k] + u[i][j][k-1] + u[i][j][k+1] + f[i][j][k] * dx * dy * dz);
    }
  }
}


### Response

Kokkos::parallel_for("stencil_3D", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({1, 1, 1}, {n-1, n-1, n-1}), KOKKOS_LAMBDA(const int i, const int j, cont int k) {
  u_new(i,j,k) = 0.125 * (u(i-1,j,k) + u(i,j+1,k) + u(i,j-1,k) + u(i+1,j,k) + u(i,j,k-1) + u(i,j,k+1) + f(i,j,k) * dx * dy * dz);
});

## Index 15

### Context

OpenMP to Kokkos translation


### Question

Can you translate this OpenMP code to Kokkos?
#pragma omp parallel for 
for (i = 0; i < SIZE; ++i) {
  for (j = 0; j < SIZE; ++j) {
    conv = 0.0;
    for (fi = 0; fi < FILTER_SIZE; ++fi) {
      for (fj = 0; fj < FILTER_SIZE; ++fj) {
        if (i + fi < SIZE && j + fj < SIZE){
          conv += filter[fi][fj] * input[i + fi][j + fj];
        }
      }
    }
    ouput[i, j] = conv;
  }
}


### Response

Kokkos::parallel_for("sgemm", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {n, n}), KOKKOS_LAMBDA (const int i, const int j) {
  float sum = 0.0f;
  for (int l = 0; l < l_dim; l++) {
    sum += a(i,l) * b(l,j);
  }
  c(i, j) = sum;
});

## Index 16

### Context

OpenMP to Kokkos translation


### Question

Can you translate this OpenMP code to Kokkos?
#pragma omp parallel for
for (i = 0; i < SIZE; i+=FILTER_SIZE) {
  for (j = 0; j < SIZE; j+=FILTER_SIZE) {
    for (fi = 0; fi < FILTER_SIZE; ++fi) {
      for (fj = 0; fj < FILTER_SIZE; ++fj){
        image[i][j] *= filter[fi][fj];
      }
    }
  }
}


### Response

Kokkos::parallel_for("sgemm", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {n, n}), KOKKOS_LAMBDA (const int i, const int j) {
  float sum = 0.0;
  for (int l = 0; l < lda; l++) {
    sum += a(i,l) * b(l,j);
  }
  c(i, j) = alpha * sum + beta * c(i, j);
});

## Index 17

### Context

HIP to Kokkos translation


### Question

Can you translate this CUDA code to Kokkos?
__global__ void saxpy (int n, float a, const float *x, float *y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = a * x[i] + y[i];
  }
}


### Response

Kokkos::parallel_for("saxpy", Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(const int i) {
  y(i) = a * x(i) + y(i);
});

## Index 18

### Context

HIP to Kokkos translation


### Question

Can you translate this CUDA code to Kokkos?
__global__ void sdot(int n, float *x, float *y, float *sum) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  float update = 0.0;
  if (i < n) {
    update = x[i] * y[i];
  }
  atomicAdd(sum, update);
}


### Response

Kokkos::parallel_reduce("sdot", Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(const int i, float &update) {
  update = x(i) * y(i);
}, Kokkos::Sum<float>(sum));

## Index 19

### Context

HIP to Kokkos translation


### Question

Can you translate this CUDA code to Kokkos?
__global__ void sgemv(int n, const float* A, const float* x, float* y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float sum = 0.0f;
    for (int j = 0; j < n; ++j) {
      sum += A[i * n + j] * x[j];
    }
    y[i] = sum;
  }
}


### Response

Kokkos::parallel_for("sgemv", Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(const int i) {
  float sum = 0.0f;
  for (int j = 0; j < n; j++) {
    sum += A(i,j) * x(j);
  }
  y(i) = sum;
});

## Index 20

### Context

HIP to Kokkos translation


### Question

Can you translate this CUDA code to Kokkos?
__global__ void sgemm(int m, int n, int k, double alpha, const double *a, int lda, const double *b, int ldb, double beta, double *c, int ldc) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < m && j < n) {
    double temp = 0.0;
    for (int l = 0; l < k; ++l) {
      temp += a[i * lda + l] * b[l * ldb + j];
    }
    c[i * ldc + j] = alpha * temp + beta * c[i * ldc + j];
  }
}


### Response

Kokkos::parallel_for("sgemm", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {n, m}), KOKKOS_LAMBDA (const int i, const int j) {
  float sum = 0.0f;
  for (int l = 0; l < k; l++) {
    sum += a(i,l) * b(l,j);
  }
  c(i, j) = alpha * sum + beta * c(i, j);
 });

## Index 21

### Context

HIP to Kokkos translation


### Question

Can you translate this CUDA code to Kokkos?
__global__ void spmv(int n, int nnz, const float *val, const int *row, const int *col, const float *x, float *y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    for (int j = row[i]; j < row[i + 1]; ++j) {
      y[i] += val[j] * x[col[j]];
    }
  }
}


### Response

Kokkos::parallel_for("spmv", Kokkos::RangePolicy(0, n), KOKKOS_LAMBDA(const int i) {
  y(i) = 0.0;
  for (int j = row(i); j < row(i + 1); ++j) {
    y(i) += val(j) * x(col(j));
  }
});

## Index 22

### Context

HIP to Kokkos translation


### Question

Can you translate this CUDA code to Kokkos?
__global__ void stencil (int n, const float *u, float *u_new) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i > 0 && i < n - 1) {
    u_new[i] = 0.5 * (u[i-1] + u[i+1]) - u[i];
  }
}


### Response

Kokkos::parallel_for("stencil", Kokkos::RangePolicy(1, n-1), KOKKOS_LAMBDA(const int i) {
  u_new(i) = 0.5 * (u(i-1) + u(i+1)) - u(i);
});

## Index 23

### Context

HIP to Kokkos translation


### Question

Can you translate this HIP code to Kokkos?
__global__ void scal (int n, float a, float *x, float *y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = a * x[i];  
  }
}


### Response

Kokkos::parallel_for("scal", Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(const int i) {
  y(i) = a * x(i);
});

## Index 24

### Context

HIP to Kokkos translation


### Question

Can you translate this HIP code to Kokkos?
__global__ void swap (int n, float *x, float *y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    a = x[i];
    x[i] = y[i];
    y[i] = a;
  }
}


### Response

Kokkos::parallel_for("swap", Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(const int i) {
  float temp = x(i);
  x(i) = y(i);
  y(i) = temp;
});

## Index 25

### Context

HIP to Kokkos translation


### Question

Can you translate this HIP code to Kokkos?
__global__ void nrm2 (int n, float *x, float *y, float *sum) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float update = 0.0;
  if (i < n ) {
    update = x[i] * x[i];
  }
  atomicAdd(sum, update);
}


### Response

Kokkos::parallel_reduce("nrm2", Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(const int i, float &update) {
  update = x(i) * x(i);
}, Kokkos::Sum<float>(sum));

## Index 26

### Context

HIP to Kokkos translation


### Question

Can you translate this HIP code to Kokkos?
__global__ void copy (int n, float *x, float *y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = x[i];
  }
}


### Response

Kokkos::parallel_for("copy", Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(const int i) {
  y(i) = x(i);
});

## Index 27

### Context

HIP to Kokkos translation


### Question

Can you translate this HIP code to Kokkos?
__global__ void asum (int n, float *x, float *sum) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float update = 0.0;
  if (i < n) {
    update = abs(x[i]);
  }
  atomicAdd(sum, update);
} 


### Response

Kokkos::parallel_reduce("asum", Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(const int i, float &update) {
  update = std::abs(x(i));
}, Kokkos::Sum<float>(sum));

## Index 28

### Context

HIP to Kokkos translation


### Question

Can you translate this HIP code to Kokkos?
__global__ void spmv_csc (int num_cols, int* col_ptr, int* row_index, float *x, float *y, float *values) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col < num_cols) {
    for (int i = col_ptr[col]; i < col_ptr[col + 1]; ++i) {
      y[row_ind[i]] += values[i] * x[col];
    }
  }
}


### Response

Kokkos::parallel_for("spmv_csc", Kokkos::RangePolicy(0, num_cols), KOKKOS_LAMBDA(const int col) {
  for (int i = col_ptr(col); i < col_ptr(col + 1); ++i) {
    y(row_ind(i)) += values(i) * x(col);
  }
});

## Index 29

### Context

HIP to Kokkos translation


### Question

Can you translate this HIP code to Kokkos?
__global__ void spmv_coo (int numzeros, int* col_ind, int* row_ind, float *x, float *y, float *values) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nonzeros){
    y[row_ind[i]] += values[i] * x[col_ind[i]];
  }
}


### Response

Kokkos::parallel_for("spmv_coo", Kokkos::RangePolicy(0, num_nonzeros), KOKKOS_LAMBDA(const int i) {
  y(row_ind(i)) += values(i) * x(col_ind(i));
});

## Index 30

### Context

HIP to Kokkos translation


### Question

Can you translate this HIP code to Kokkos?
__global__ void stencil_2d (int n, float dx, float dy, float *u, float *u_new, float *f) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if ( i > 0 && i < n-1 && j > 0 && j < n-1){
    u_new [i][j] = 0.25 * (u[i-1][j] + u[i][j+1] + u[i][j-1] + u[i+1][j] + f[i][j] * dx * dy);
  }
}


### Response

Kokkos::parallel_for("stencil_2D", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {n-1, n-1}), KOKKOS_LAMBDA(const int i, const int j) {
  u_new(i,j) = (u(i-1,j) + u(i,j+1) + u(i,j-1 + u(i+1,j) + f(i,j)) * dx * dy * 0.25;
});

## Index 31

### Context

HIP to Kokkos translation


### Question

Can you translate this HIP code to Kokkos?
__global__ void stencil_3d (int n, float dx, float dy, float dz, float *u, float *u_new, float *f) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;
  if ( i > 0 && i < n-1 && j > 0 && j < n-1 && k > 0 && k < n - 1){
    u_new [i][j][k] = 0.125 * (u[i-1][j][k] + u[i][j+1][k] + u[i][j-1][k] + u[i+1][j][k] + u[i][j][k-1] + u[i][j][k+1] + f[i][j][k] * dx * dy * dz);
  }
}


### Response

Kokkos::parallel_for("stencil_3D", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({1, 1, 1}, {n-1, n-1, n-1}), KOKKOS_LAMBDA(const int i, const int j, cont int k) {
  u_new(i,j,k) = 0.125 * (u(i-1,j,k) + u(i,j+1,k) + u(i,j-1,k) + u(i+1,j,k) + u(i,j,k-1) + u(i,j,k+1) + f(i,j,k) * dx * dy * dz);
});

## Index 32

### Context

HIP to Kokkos translation


### Question

Can you translate this HIP code to Kokkos?
__global__ void conv (int SIZE, int FILTER_SIZE, float *filter, float *input, float *output) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  float conv = 0.0;
  for (fi = 0; fi < FILTER_SIZE; ++fi) {
    for (fj = 0; fj < FILTER_SIZE; ++fj) {
      if (i + fi < SIZE && j + fj < SIZE){
        conv += filter[fi][fj] * input[i + fi][j + fj];
      }
    }
  }
  ouput[i, j] = conv;
}


### Response

Kokkos::parallel_for("conv", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {n - f_n + 1, n - f_n + 1}), KOKKOS_LAMBDA(const int i, const int j) {
  float sum = 0.0f;
  for (int fi = 0; fi < f_n; fi++) {
    for (int fj = 0; fj < f_n; fj++) {
      if (i + fi >= 0 && j + fj >= 0 && i + fi < n && j + fj < n) {
        sum += filter(fi, fj) * input(i + fi, j + fj);
      }
    }
  }
  output(i, j) = sum;
});

## Index 33

### Context

HIP to Kokkos translation


### Question

Can you translate this HIP code to Kokkos?
__global__ void spectral (int SIZE, int FILTER_SIZE, float *filter, float *input, float *output) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  for (fi = 0; fi < FILTER_SIZE; ++fi) {
    for (fj = 0; fj < FILTER_SIZE; ++fj) {
      if (i + fi < SIZE && j + fj < SIZE){
        image[i + fi][j + fj] *= filter[fi][fj];
      }
    }
  }
}


### Response

Kokkos::parallel_for("spectral", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {n - filter_size, n - filter_size}), KOKKOS_LAMBDA(const int i, const int j) {
  float sum = 0.0f;
  for (int fi = 0; fi < filter_size; fi++) {
    for (int fj = 0; fj < filter_size; fj++) {
      if (i + fi >= 0 && j + fj >= 0 && i + fi < n && j + fj < n) {
        sum += filter(fi, fj) * image(i + fi, j + fj);
      }
    }
  }
  image(i, j) = sum;
});

## Index 34

### Context

Sequential to Kokkos parallelization


### Question

Can you parallelize this sequential code using Kokkos?
for (i = 0, i < n, i++) {
  y[i] = a * x[i];
}


### Response

Kokkos::parallel_for("saxpy", Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(const int i) {
  y(i) = a * x(i);
});

## Index 35

### Context

Sequential to Kokkos parallelization


### Question

Can you parallelize this sequential code using Kokkos?
for (i = 0, i < n, i++) {
  a = x[i];
  x[i] = y[i];
  y[i] = a;
}


### Response

Kokkos::parallel_for("sgemv", Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(const int i) {
  float sum = 0.0;
  for (int j = 0; j < n; j++) {
    sum += A(i,j) * x(j);
  }
  y(i) = sum;
});

## Index 36

### Context

Sequential to Kokkos parallelization


### Question

Can you parallelize this sequential code using Kokkos?
for (i = 0, i < n, i++) {
  sum += x[i] * x[i];
}


### Response

Kokkos::parallel_reduce("sdot", Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(const int i, float &update) {
  update = x(i) * x(i);
}, Kokkos::Sum<float>(sum));

## Index 37

### Context

Sequential to Kokkos parallelization


### Question

Can you parallelize this sequential code using Kokkos?
for (i = 0, i < n, i++) {
  y[i] = x[i];
}


### Response

Kokkos::parallel_for("saxpy", Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(const int i) {
  y(i) = x(i);
});

## Index 38

### Context

Sequential to Kokkos parallelization


### Question

Can you parallelize this sequential code using Kokkos?
for (i = 0, i < n, i++) {
  sum += abs(x[i]);
}


### Response

Kokkos::parallel_reduce("sdot", Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(const int i, float &update) {
  update = std::abs(x(i));
}, Kokkos::Sum<float>(sum));

## Index 39

### Context

Sequential to Kokkos parallelization


### Question

Can you parallelize this sequential code using Kokkos?
for (int col = 0; col < num_cols; ++col) {
  for (int i = col_ptr[col]; i < col_ptr[col + 1]; ++i) {
    y[row_ind[i]] += values[i] * x[col];
  }
}


### Response

Kokkos::parallel_for("spmv_csc", Kokkos::RangePolicy(0, num_cols), KOKKOS_LAMBDA(const int col) {
  for (int i = col_ptr(col); i < col_ptr(col + 1); ++i) {
    y(row_ind(i)) += values(i) * x(col);
  }
});

## Index 40

### Context

Sequential to Kokkos parallelization


### Question

Can you parallelize this sequential code using Kokkos?
for (int i = 0; i < nonzeros; i++) {
  y[row_ind [i]] += values[i] * x[col_ind [i]];
}


### Response

Kokkos::parallel_for("spmv", Kokkos::RangePolicy(0, n), KOKKOS_LAMBDA(const int i) {
  y(i) = 0.0;
}, Kokkos::Sum<float>(k));
for (int i = 0; i < nonzeros; i++) {
  y(row_ind[i]) += values[i] * x(col_ind[i]);
}

## Index 41

### Context

Sequential to Kokkos parallelization


### Question

Can you parallelize this sequential code using Kokkos?
for (i = 1; i < n-1; i++) {
  for (j = 1; j < n-1; j++) {
    u_new [i][j] = 0.25 * (u[i-1][j] + u[i][j+1] + u[i][j-1] + u[i+1][j] + f[i][j] * dx * dy);
  }
}


### Response

Kokkos::parallel_for("stencil_2D", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {n-1, n-1}), KOKKOS_LAMBDA(const int i, const int j) {
  u_new(i,j) = 0.25 * (u(i-1,j) + u(i,j+1) + u(i,j-1 + u(i+1,j) + f(i,j) * dx * dy);
});

## Index 42

### Context

Sequential to Kokkos parallelization


### Question

Can you parallelize this sequential code using Kokkos?
for (i = 1; i < n-1; i++) {
  for (j = 1; j < n-1; j++) {
    for (k = 1; k < n-1; k++) {
      u_new [i][j][k] = 0.125 * (u[i-1][j][k] + u[i][j+1][k] + u[i][j-1][k] + u[i+1][j][k] + u[i][j][k-1] + u[i][j][k+1] + f[i][j][k] * dx * dy * dz);
    }
  }
}


### Response

Kokkos::parallel_for("stencil_3D", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({1, 1, 1}, {n-1, n-1, n-1}), KOKKOS_LAMBDA(const int i, const int j, cont int k) {
  u_new(i,j,k) = 0.125 * (u(i-1,j,k) + u(i,j+1,k) + u(i,j-1,k) + u(i+1,j,k) + u(i,j,k-1) + u(i,j,k+1) + f(i,j,k) * dx * dy * dz);
});

## Index 43

### Context

Sequential to Kokkos parallelization


### Question

Can you parallelize this sequential code using Kokkos?
for (i = 0; i < SIZE; ++i) {
  for (j = 0; j < SIZE; ++j) {
    conv = 0.0;
    for (fi = 0; fi < FILTER_SIZE; ++fi) {
      for (fj = 0; fj < FILTER_SIZE; ++fj) {
        if (i + fi < SIZE && j + fj < SIZE){
          conv += filter[fi][fj] * input[i + fi][j + fj];
        }
      }
    }
    ouput[i, j] = conv;
  }
}


### Response

Kokkos::parallel_for("sgemm", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {n, n}), KOKKOS_LAMBDA (const int i, const int j) {
  float sum = 0.0f;
  for (int l = 0; l < l_dim; l++) {
    sum += a(i,l) * b(l,j);
  }
  c(i, j) = sum;
});

## Index 44

### Context

Sequential to Kokkos parallelization


### Question

Can you parallelize this sequential code using Kokkos?
for (i = 0; i < SIZE; i+=FILTER_SIZE) {
  for (j = 0; j < SIZE; j+=FILTER_SIZE) {
    for (fi = 0; fi < FILTER_SIZE; ++fi) {
      for (fj = 0; fj < FILTER_SIZE; ++fj){
        image[i][j] *= filter[fi][fj];
      }
    }
  }
}


### Response

Kokkos::parallel_for("sgemm", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {n, m}), KOKKOS_LAMBDA (const int i, const int j) {
  float sum = 0.0;
  for (int l = 0; l < k; l++) {
    sum += a(i,l) * b(l,j);
  }
  c(i, j) = sum;
});


