# distutils: language = c
# cython: cdivision = True
# cython: boundscheck = False
# cython: wraparound = False
# cython: profile = False

from ._util cimport gcv_adjust, log2, apply_weights_1d, apply_weights_slice
from ._basis cimport (Basis, BasisFunction, ConstantBasisFunction,
                      HingeBasisFunction, LinearBasisFunction,
                      MissingnessBasisFunction)
from ._record cimport ForwardPassIteration
from ._types import BOOL, INT
from ._knot_search cimport knot_search, MultipleOutcomeDependentData, PredictorDependentData, \
    KnotSearchReadOnlyData, KnotSearchState, KnotSearchWorkingData, KnotSearchData
from ._qr cimport UpdatingQT

from cython.view cimport array as carray

import copy
import _pickle as cPickle
from sklearn.model_selection import KFold, train_test_split
from libc.math cimport sqrt
from scipy import linalg


import numpy as np
cnp.import_array()

from heapq import heappush, heappop
class FastHeapContent:

    def __init__(self, idx, mse=-np.inf, m=-np.inf, v=None):
        """
        This class defines an entry of the priority queue as defined in [1].
        The entry stores information about parent basis functions and is
        used by the priority queue in the forward pass
        to choose the next parent basis function to try.

        References
        ----------
        .. [1] Fast MARS, Jerome H.Friedman, Technical Report No.110, May 1993.

        """
        self.idx = idx
        self.mse = mse
        self.m = m
        self.v = v

    def __lt__(self, other):
        return self.mse < other.mse

cdef int MAXTERMS = 0
cdef int MAXRSQ = 1
cdef int NOIMPRV = 2
cdef int LOWGRSQ = 3
cdef int NOCAND = 4
stopping_conditions = {
    MAXTERMS: "Reached maximum number of terms",
    MAXRSQ: "Achieved RSQ value within threshold of 1",
    NOIMPRV: "Improvement below threshold",
    LOWGRSQ: "GRSQ too low",
    NOCAND: "No remaining candidate knot locations"
}



# Implements Incremental QR Decomposition
# for solving the least squares algorithm
cdef class IncrementalQR_Cython:
    cdef:
        cnp.ndarray R
        cnp.ndarray d
        
    def __init__(IncrementalQR_Cython self, X, y):
        Q, R = np.linalg.qr(X, mode = 'complete')
        self.R = np.asarray(R, dtype=np.float64)
        self.d = np.dot(Q.T, y)

    cdef inline (long double, long double) givens_rotation(self, long double a, long double b):
        """Compute the Givens rotation matrix parameters c and s for entries a and b."""
        cdef long double s, c, tau
        cdef long double abs_a = a if a >= 0 else -a
        cdef long double abs_b = b if b >= 0 else -b

        if b == 0:
            return 1.0, 0.0
        elif abs_b > abs_a:
            tau = -a / b
            s = 1 / sqrt(1 + tau**2)
            c = s * tau
        else:
            tau = -b / a
            c = 1 / sqrt(1 + tau**2)
            s = c * tau
        return c, s

    cdef inline double[:] matmul(self, long double a, double[:] mat1, long double b, double[:] mat2, int plus):
        cdef Py_ssize_t i, n = mat1.shape[0]
        # Create a new array of the same size as mat
        cdef double[:] result1 = carray(shape=(n,), itemsize=sizeof(double), format="d")
        for i in range(n):
            result1[i] = a * mat1[i]
            
        i = 0
        cdef double[:] result2 = carray(shape=(n,), itemsize=sizeof(double), format="d")
        for i in range(n):
            result2[i] = b * mat2[i]
            
        
        i = 0
        cdef double[:] result3 = carray(shape=(n,), itemsize=sizeof(double), format="d")
        if plus == 1:
            for i in range(n):
                result3[i] = result1[i] + result2[i]
        else:
            for i in range(n):
                result3[i] = result1[i] - result2[i]
        return result3

   

    def update_qr(self, cnp.ndarray[FLOAT_t, ndim=1] new_X, long double new_y):    
        # new_X is a new row on the X matrix
        # new_y is a new y value

        cdef int n_iter = min(self.R.shape[0], self.R.shape[1])
        cdef int n = self.R.shape[1]
        cdef int j
        cdef long double c, s, t3, t4
        cdef double[:] t1, t2
        cdef double[:, :] R_view = self.R
        cdef double[:] X_view = new_X
        cdef double[:] d_view = self.d
        
        for j in range(n_iter):
            # Apply Givens rotation
            c, s = self.givens_rotation(R_view[j, j], X_view[j])
            R_view[j,j] = c*R_view[j,j] - s*X_view[j]
            if j+1 < n:
                t1 = R_view[j, j+1:n+1]
                t2 = X_view[j+1:n+1]
                self.R[j,j+1:n+1], new_X[j+1:n+1] = self.matmul(c, t1, s, t2, 0), self.matmul(s, t1, c, t2, 1)
            
            t3 = d_view[j]
            t4 = new_y
            d_view[j] = c*t3 - s*t4
            new_y = s*t3 + c*t4


        if self.R.shape[0] < n:
            addition = np.zeros(n)
            addition[n:] = new_X[n:]
            self.R = np.vstack([self.R, addition])
            # Update d with the last element of u
            self.d = np.append(self.d, new_y)

#     def update_qr(self, cnp.ndarray[FLOAT_t, ndim=1] new_X, long double new_y):    
#         # new_X is a new row on the X matrix
#         # new_y is a new y value

#         cdef int n_iter = min(self.R.shape[0], self.R.shape[1])
#         cdef int n = self.R.shape[1]
#         cdef int j
# #         cdef long double c, s, t3, t4
#         cdef double c, s #, t3, t4
#         cdef long double t3, t4
# #         cdef double[:] t1, t2
#         cdef double[:, :] R_view = self.R
#         cdef double[:] X_view = new_X
#         cdef double[:] d_view = self.d
        
#         for j in range(n_iter):
#             # Apply Givens rotation
#             c, s = self.givens_rotation(R_view[j, j], X_view[j])
#             R_view[j,j] = c*R_view[j,j] - s*X_view[j]
#             if j+1 < n:
#                 t1 = self.R[j, j+1:n+1]
#                 t2 = new_X[j+1:n+1]
#                 self.R[j,j+1:n+1], new_X[j+1:n+1] = c*t1 - s*t2, s*t1 + c*t2#self.matmul(c, t1, s, t2, 0), self.matmul(s, t1, c, t2, 1)
            
#             t3 = d_view[j]
#             t4 = new_y
#             d_view[j] = c*t3 - s*t4
#             new_y = s*t3 + c*t4


#         if self.R.shape[0] < n:
#             addition = np.zeros(n)
#             addition[n:] = new_X[n:]
#             self.R = np.vstack([self.R, addition])
#             # Update d with the last element of u
#             self.d = np.append(self.d, new_y)

    cdef inline int compute_dim(self):
        cdef double[:, :] R_view = self.R
        cdef int dim = self.R.shape[0] if R_view.shape[0] <= R_view.shape[1] else R_view.shape[1]
        cdef int i = 0
        for i in range(dim):
            if abs(R_view[i, i]) < 1e-5:
                # If R row is a row of 0's don't include it in your backsubstitution
                return i
        return dim

    def precise_back_substitution(self, U, y):
        n = U.shape[0]
        x = np.zeros(n, dtype=np.float128)

        if n == 0:  # Early return for empty input as an error handling strategy
            return np.empty(0, dtype=np.float128)

        for i in range(n - 1, -1, -1):
            sumy = 0
            for j in range(i + 1, n):
                sumy += U[i, j] * x[j]
            if abs(U[i, i]) < 1e-10:
                return np.empty(0, dtype=np.float128)  # Return an empty array to indicate an error
            x[i] = (y[i] - sumy) / U[i, i]

        return np.asarray(x)

    # def predict(self, cnp.ndarray[double, ndim=2] A):
    #     cdef int dim = self.compute_dim()
    #     # Solve with back substitution
    #     return np.dot(A[:, 0:dim], linalg.solve_triangular(self.R[:dim, :dim], self.d[0:dim]))

    def predict(self, cnp.ndarray[FLOAT_t, ndim=2] A):
        cdef int dim = self.compute_dim()
        cdef int backsub = self.R.shape[0] if self.R.shape[0] <= self.R.shape[1] else self.R.shape[1]

        
        if dim == backsub:
            return np.dot(A[:, 0:dim], linalg.solve_triangular(self.R[:dim, :dim], self.d[0:dim]))
            # return np.dot(A[:, 0:dim], self.precise_back_substitution(self.R[:dim, :dim], self.d[0:dim]))
        else:
            return np.dot(A[:, 0:backsub], np.linalg.lstsq(self.R[:backsub, :backsub], self.d[0:backsub], rcond=-2)[0])
    # def predict(self, A):
    #     # Get coefs and predict on test data
    #     cdef int dim = self.R.shape[0] if self.R.shape[0] <= self.R.shape[1] else self.R.shape[1]
    #     return np.dot(A[:, 0:dim], np.linalg.lstsq(self.R[:dim, :dim], self.d[0:self.R.shape[1]], rcond=-2)[0])



cdef class ForwardPasser:
    def __init__(ForwardPasser self, cnp.ndarray[FLOAT_t, ndim=2] X,
                 cnp.ndarray[BOOL_t, ndim=2] missing,
                 cnp.ndarray[FLOAT_t, ndim=2] y,
                 cnp.ndarray[FLOAT_t, ndim=2] sample_weight,
                 **kwargs):

        cdef INDEX_t i
        self.X = X
        self.missing = missing
        self.y = y
        self.Node = None

        # Keep track of variables used in the forward pass to use those to split on!
        self.variables_chosen = []

        # Assuming Earth.fit got capital W (the inverse of squared variance)
        # so the objective function is (sqrt(W) * residual) ^ 2)
        self.sample_weight = np.sqrt(sample_weight)
        self.m = self.X.shape[0]
        self.n = self.X.shape[1]
        self.endspan       = kwargs.get('endspan', -1)
        self.minspan       = kwargs.get('minspan', -1)
        self.endspan_alpha = kwargs.get('endspan_alpha', .05)
        self.minspan_alpha = kwargs.get('minspan_alpha', .05)
        #self.max_terms     = kwargs.get('max_terms', min(2 * self.n + self.m // 10, 400))

        # My thought, why would you want more max terms than the number of observations?
        # Cuz at that point by DOF you have hit every point exactly, any more terms unneccesary. 
        self.max_terms     = kwargs.get('max_terms', min(2 * self.n + self.m // 10, min(self.m, 400)))
        self.allow_linear  = kwargs.get('allow_linear', True)
        self.max_degree    = kwargs.get('max_degree', 1)
        self.thresh        = kwargs.get('thresh', 0.001)
        self.penalty       = kwargs.get('penalty', 3.0)
        self.check_every   = kwargs.get('check_every', -1)
        self.min_search_points = kwargs.get('min_search_points', 100)
        self.xlabels       = kwargs.get('xlabels')
        self.use_fast = kwargs.get('use_fast', False)
        self.fast_K = kwargs.get("fast_K", 5)
        self.fast_h = kwargs.get("fast_h", 1)
        self.zero_tol = kwargs.get('zero_tol', 1e-12)
        self.allow_missing = kwargs.get("allow_missing", False)
        self.verbose = kwargs.get("verbose", 0)
        #self.allow_subset = kwargs.get("allow_subset", True)
        if self.allow_missing:
            self.has_missing = np.any(self.missing, axis=0).astype(BOOL)

        self.fast_heap = []

        if self.xlabels is None:
            self.xlabels = ['x' + str(i) for i in range(self.n)]
        if self.check_every < 0:
            self.check_every = (<int > (self.m / self.min_search_points)
                                if self.m > self.min_search_points
                                else 1)

        weighted_mean = np.mean((self.sample_weight ** 2) * self.y)
        self.sst = np.sum((self.sample_weight * (self.y - weighted_mean)) ** 2)
        self.basis = Basis(self.n)
        self.basis.append(ConstantBasisFunction())
        if self.use_fast is True:
            content = FastHeapContent(idx=0)
            heappush(self.fast_heap, content)

        self.mwork = np.empty(shape=self.m, dtype=np.int64)

        self.B = np.ones(
            shape=(self.m, self.max_terms + 4), order='F', dtype=float)
        self.basis.transform(self.X, self.missing, self.B[:,0:1])

        if self.endspan < 0:
            self.endspan = round(3 - log2(self.endspan_alpha / self.n))

        self.linear_variables = np.zeros(shape=self.n, dtype=INT)
        self.init_linear_variables()

        # Removed in favor of new knot search code
        self.iteration_number = 0

        # Add in user selected linear variables
        for linvar in kwargs.get('linvars',[]):
            if linvar in self.xlabels:
                self.linear_variables[self.xlabels.index(linvar)] = 1
            elif linvar in range(self.n):
                self.linear_variables[linvar] = 1
            else:
                raise IndexError(
                    'Unknown variable selected in linvars argument.')

        # Initialize the data structures for knot search
        self.n_outcomes = self.y.shape[1]
        n_predictors = self.X.shape[1]
        n_weights = self.sample_weight.shape[1]
        self.workings = []
        self.outcome = MultipleOutcomeDependentData.alloc(self.y, self.sample_weight, self.m,
                                                          self.n_outcomes, self.max_terms + 4,
                                                          self.zero_tol)
        self.outcome.update_from_array(self.B[:,0])
        self.total_weight = 0.
        for i in range(self.n_outcomes):
            working = KnotSearchWorkingData.alloc(self.max_terms + 4)
            self.workings.append(working)
            self.total_weight += self.outcome.outcomes[i].weight.total_weight
        self.predictors = []
        for i in range(n_predictors):
            x = self.X[:, i].copy()
            x[missing[:,i]==1] = 0.
            predictor = PredictorDependentData.alloc(x)
            self.predictors.append(predictor)

        # Initialize the forward pass record
        self.record = ForwardPassRecord(
            self.m, self.n, self.penalty, self.outcome.mse(), self.xlabels)

    # Return basis functions
    cpdef Basis get_basis(ForwardPasser self):
        # Very important to use pickle, it is essentially doing a deep copy and avoids pointer issues
        data = cPickle.dumps(self.basis)
        basis_copy = cPickle.loads(data)
        return basis_copy

    def set_basis(ForwardPasser self, x):
        # Very important to use pickle, it is essentially doing a deep copy and avoids pointer issues
        data = cPickle.dumps(x)
        basis_copy = cPickle.loads(data)
        self.basis = basis_copy

    cpdef init_linear_variables(ForwardPasser self):
        cdef INDEX_t variable
        cdef cnp.ndarray[INT_t, ndim = 1] order
        cdef cnp.ndarray[INT_t, ndim = 1] linear_variables = (
            <cnp.ndarray[INT_t, ndim = 1] > self.linear_variables)
        cdef cnp.ndarray[FLOAT_t, ndim = 2] B = (
            <cnp.ndarray[FLOAT_t, ndim = 2] > self.B)
        cdef cnp.ndarray[FLOAT_t, ndim = 2] X = (
            <cnp.ndarray[FLOAT_t, ndim = 2] > self.X)
        cdef ConstantBasisFunction root_basis_function = self.basis[0]
        for variable in range(self.n):
            order = np.argsort(X[:, variable])[::-1].astype(np.int64)
            if root_basis_function.valid_knots(B[order, 0], X[order, variable],
                                               variable, self.check_every,
                                               self.endspan, self.minspan,
                                               self.minspan_alpha, self.n,
                                               self.mwork).shape[0] == 0:
                linear_variables[variable] = 1
            else:
                linear_variables[variable] = 0

    cpdef run(ForwardPasser self):
        if self.verbose >= 1:
            print('Beginning forward pass')
            print(self.record.partial_str(slice(-1, None, None), print_footer=False))

        if self.max_terms <= len(self.basis):
            # We done!
            if self.verbose >= 1:
                print("Stopping forward pass, hit max terms")
            return
        if self.max_terms > 1 and self.record.mse(0) != 0.:
            while True:
                self.next_pair()
                if self.stop_check():
                    if self.verbose >= 1:
                        print(self.record.partial_str(slice(-1, None, None), print_header=False, print_footer=True))
                        print(self.record.final_str())
                    break
                else:
                    if self.verbose >= 1:
                        print(self.record.partial_str(slice(-1, None, None), print_header=False, print_footer=False))
                #if self.Node.left_child != None and self.Node.right_child != None:
                #    return 
                #else:
                #    pass
                self.iteration_number += 1
        return

    cdef stop_check(ForwardPasser self):
        last = self.record.__len__() - 1

        assert self.record.iterations[last].get_size() == len(self.basis)
        

        if self.record.iterations[last].get_size() > self.max_terms:
            self.record.stopping_condition = MAXTERMS
            return True
        rsq = self.record.rsq(last)
        if rsq > 1 - self.thresh:
            self.record.stopping_condition = MAXRSQ
            return True
        if last > 0:
            previous_rsq = self.record.rsq(last - 1)
            if rsq - previous_rsq < self.thresh:
                self.record.stopping_condition = NOIMPRV
                return True
        
        if self.record.grsq(last) < -10:
            self.record.stopping_condition = LOWGRSQ
            return True
        if self.record.iterations[last].no_further_candidates():
            self.record.stopping_condition = NOCAND
            return True
        if self.record.mse(last) == self.zero_tol:
            self.record.stopping_condition = NOIMPRV
            return True
        return False


    cpdef orthonormal_update(ForwardPasser self, b):
        # Update the outcome data
        linear_dependence = False
        return_codes = []
        return_code = self.outcome.update_from_array(b)
        if return_code == -1:
            raise ValueError('This should not have happened.')
        if return_code == 1:
            linear_dependence = True
        return linear_dependence

    cpdef orthonormal_downdate(ForwardPasser self):
        self.outcome.downdate()

    def trace(self):
        return self.record

    # Get the variables to consider for finding next split to consider
    # Considers either all the variables or the ones chosen in the forward pass
    def get_options(ForwardPasser self, vars_to_consider):
        if vars_to_consider == "all":
            options = range(self.n)
        elif vars_to_consider == "chosen":
            # Get all unique variables chosen in the forward pass
            options = list(set(self.variables_chosen))
        else:
            raise ValueError("Unrecognized vars_to_consider: ", vars_to_consider, " Must be 'all' or 'chosen'")
        return options

    def subset_MSE_NP(ForwardPasser self, vars_to_consider):
        optimal_MSE = float('inf')
        optimal_split = None
        optimal_var = None

        options = self.get_options(vars_to_consider)

        for variable in options:
            sorted_indices = np.argsort(self.X[:, variable])
            x_sorted = self.X[sorted_indices]
            B_sorted = self.B[sorted_indices, 0:len(self.basis)]
            y_sorted = self.y[sorted_indices]
            for i in range(0, self.X.shape[0]-1):
                if x_sorted[i, variable] == x_sorted[i+1, variable]:
                    # Not valid split value
                    continue
                
                x1 = B_sorted[0:i+1]
                y1 = y_sorted[0:i+1]
                x2 = B_sorted[i+1:self.X.shape[0]]
                y2 = y_sorted[i+1:self.X.shape[0]]
                
                mse1 = np.mean((y1 - x1 @ np.linalg.lstsq(x1, y1, rcond=-2)[0])**2)
                mse2 = np.mean((y2 - x2 @ np.linalg.lstsq(x2, y2, rcond=-2)[0])**2)
                
                total_mse = (mse1*x1.shape[0] + mse2*x2.shape[0])/(x1.shape[0]+x2.shape[0])
                
                if total_mse < optimal_MSE:
                    optimal_MSE = total_mse
                    optimal_split = (x_sorted[i, variable] + x_sorted[i+1, variable])/2
                    optimal_var = variable

        return optimal_var, optimal_split

    def subset_MSE_NP_validation(ForwardPasser self, vars_to_consider):
        optimal_MSE = float('inf')
        optimal_split = None
        optimal_var = None

        options = self.get_options(vars_to_consider)

        X_train, X_val, Beta_train, Beta_val, y_train, y_val = train_test_split(
                            self.X, self.B[:, 0:len(self.basis)], self.y, test_size=0.3, random_state=1)

        for variable in options:
            sorted_indices = np.argsort(X_train[:, variable])
            x_sorted = X_train[sorted_indices]
            B_sorted = Beta_train[sorted_indices]
            y_sorted = y_train[sorted_indices]
            
            for i in range(0, X_train.shape[0] - 1):
                if x_sorted[i, variable] == x_sorted[i + 1, variable]:
                    continue

                x1_train = B_sorted[:i + 1]
                y1_train = y_sorted[:i + 1]
                x2_train = B_sorted[i + 1:]
                y2_train = y_sorted[i + 1:]

                # Fit the models
                model1 = np.linalg.lstsq(x1_train, y1_train, rcond=-2)[0]
                model2 = np.linalg.lstsq(x2_train, y2_train, rcond=-2)[0]

                # Precompute predictions for the entire validation set using both models
                preds_model1 = Beta_val @ model1
                preds_model2 = Beta_val @ model2

                # Create a boolean mask for selecting which model's predictions to use
                split_val = (x_sorted[i, variable] + x_sorted[i + 1, variable]) / 2
                mask = (X_val[:, variable] <= split_val).reshape(-1, 1)

                # Select predictions based on the mask
                preds = np.where(mask, preds_model1, preds_model2)

                mse_val = np.mean((y_val - preds)**2)
                
                if mse_val < optimal_MSE:
                    optimal_MSE = mse_val
                    optimal_split = split_val
                    optimal_var = variable

        return optimal_var, optimal_split
    

    # Uses QR decomp linreg on a validation set to find the MSE
    # If the variable is categorical, uses OLS instead. 
    def subset_MSE_QR_Cat_validation(ForwardPasser self, vars_to_consider):
        optimal_MSE = float('inf')
        optimal_split = None
        optimal_var = None
        options = self.get_options(vars_to_consider)
        X_train, X_val, Beta_train, Beta_val, y_train, y_val = train_test_split(
                        self.X, self.B[:, 0:len(self.basis)], self.y, test_size=0.3, random_state=1)

        for variable in options:
            unique = np.unique(X_train[:, variable])
            if len(unique) <= 1:
                continue

            if len(unique) < max(X_train.shape[0]/100, 70):
                sorted_indices = np.argsort(X_train[:, variable])
                x_sorted = X_train[sorted_indices]
                B_sorted = Beta_train[sorted_indices]
                y_sorted = y_train[sorted_indices]

                for i in range(0, X_train.shape[0] - 1):
                    if x_sorted[i, variable] == x_sorted[i + 1, variable]:
                        continue

                    x1_train = B_sorted[:i + 1]
                    y1_train = y_sorted[:i + 1]
                    x2_train = B_sorted[i + 1:]
                    y2_train = y_sorted[i + 1:]

                    # Fit the models
                    model1 = np.linalg.lstsq(x1_train, y1_train, rcond=-2)[0]
                    model2 = np.linalg.lstsq(x2_train, y2_train, rcond=-2)[0]

                    # Precompute predictions for the entire validation set using both models
                    preds_model1 = Beta_val @ model1
                    preds_model2 = Beta_val @ model2

                    # Create a boolean mask for selecting which model's predictions to use
                    split_val = (x_sorted[i, variable] + x_sorted[i + 1, variable]) / 2
                    mask = (X_val[:, variable] <= split_val).reshape(-1, 1)

                    # Select predictions based on the mask
                    preds = np.where(mask, preds_model1, preds_model2)

                    mse_val = np.mean((y_val - preds)**2)

                    if mse_val < optimal_MSE:
                        optimal_MSE = mse_val
                        optimal_split = split_val
                        optimal_var = variable
            else:
                sorted_indices = np.argsort(X_train[:, variable])
                x1_values = X_train[sorted_indices,:]
                x1 = Beta_train[sorted_indices]
                x2 = np.copy(x1[::-1])
                x2_values = x1_values[::-1]
                y1 = y_train[sorted_indices]
                y2 = y1[::-1]
                model1 = None
                model2 = None

                mse2_list = []

                # Do it in reverse first
                for i in range(0, len(x2)-1):
                    if model2 == None:
                        model2 = IncrementalQR_Cython(x2[i:i+1], y2[i])
                    else:
                        model2.update_qr(x2[i], y2[i][0])
                        
                                    
                    # Find the split value based on the actual X's
                    split_val = (x2_values[i, variable] + x2_values[i+1, variable])/2
                    mask = (X_val[:, variable] > split_val)
                    mask_sum = np.sum(mask)
                    if mask_sum > 0:
                        # Predict again on the Betas
                        y_pred2 = model2.predict(Beta_val[mask])
                        mse2 = np.mean((y_val[mask].flatten() - y_pred2)**2)
                    else:
                        # No validation data in this split, mse2 will not be important
                        mse2 = 0
                    mse2_list.append( mse2 )

                # Reverse the list of RSS's
                mse2_list = mse2_list[::-1]

                # Now go forward and train model1
                for i in range(0, len(x1)-1):
                    if model1 == None:
                        model1 = IncrementalQR_Cython(x1[i:i+1], y1[i])
                    else:
                        model1.update_qr(x1[i], y1[i][0])

                    if (x1_values[i, variable] != x1_values[i+1, variable]):
                        # Then its a valid split value
                        # Subset the data based on the split val on the X's
                        split_val = (x1_values[i, variable] + x1_values[i+1, variable])/2
                        mask = (X_val[:, variable] <= split_val)
                        mask_sum = np.sum(mask)
                        
                        if mask_sum > 0:
                            # Predict on the Betas
                            y_pred1 = model1.predict(Beta_val[mask])
                            mse1 = np.mean((y_val[mask].flatten() - y_pred1)**2)
                        else:
                            # No validation data in this split, mse1 will not be important
                            mse1 = 0
                        
                        
                        # Mean Squared Error of this split
                        mse_val = (mse1*mask_sum + mse2_list[i]*(X_val.shape[0]-mask_sum)) / X_val.shape[0]
                        if mse_val < optimal_MSE:
                            optimal_MSE = mse_val
                            optimal_var = variable
                            optimal_split = split_val

        return optimal_var, optimal_split

    # Uses QR decomp linreg on a it's own set to find the MSE
    # If the variable is categorical, uses OLS instead. 
    def subset_MSE_QR_Cat(ForwardPasser self, vars_to_consider):
        optimal_MSE = float('inf')
        optimal_split = None
        optimal_var = None
        options = self.get_options(vars_to_consider)

        for variable in options:
            unique = np.unique(self.X[:, variable])
            if len(unique) <= 1:
                continue

            if len(unique) < max(len(self.X[:, variable])/100.0, 100):
                sorted_indices = np.argsort(self.X[:, variable])
                x_sorted = self.X[sorted_indices]
                B_sorted = self.B[sorted_indices, 0:len(self.basis)]
                y_sorted = self.y[sorted_indices]
                for i in range(0, self.X.shape[0]-1):
                    if x_sorted[i, variable] == x_sorted[i+1, variable]:
                        # Not valid split value
                        continue
                    x1 = B_sorted[0:i+1]
                    y1 = y_sorted[0:i+1]
                    x2 = B_sorted[i+1:self.X.shape[0]]
                    y2 = y_sorted[i+1:self.X.shape[0]]

                    mse1 = np.mean((y1 - x1 @ np.linalg.lstsq(x1, y1, rcond=-2)[0])**2)
                    mse2 = np.mean((y2 - x2 @ np.linalg.lstsq(x2, y2, rcond=-2)[0])**2)

                    total_mse = (mse1*x1.shape[0] + mse2*x2.shape[0])/(x1.shape[0]+x2.shape[0])

                    if total_mse < optimal_MSE:
                        optimal_MSE = total_mse
                        optimal_split = (x_sorted[i, variable] + x_sorted[i+1, variable])/2
                        optimal_var = variable
            else:
                sorted_indices = np.argsort(self.X[:, variable])
                x1_values = self.X[sorted_indices,:]
                x1 = self.B[sorted_indices, 0:len(self.basis)]
                x1_clean = np.copy(x1)
                x2 = np.copy(x1[::-1])
                x2_clean = np.copy(x2)
                y1 = self.y[sorted_indices]
                y2 = y1[::-1]

                model1 = None
                model2 = None  
                mse2_list = []

                # Do it in reverse first
                for i in range(0, len(x2)-1):
                    if model2 == None:
                        model2 = IncrementalQR_Cython(x2[i:i+1], y2[i])
                    else:
                        model2.update_qr(x2[i], y2[i][0])
                        
                    y_pred2 = model2.predict(x2_clean[0:i+1])
                    mse2 = np.mean((y2[0:i+1].flatten() - y_pred2)**2)
                    mse2_list.append( mse2 )


                # Reverse the list of RSS's
                mse2_list = mse2_list[::-1]

                # Now go forward and 
                for i in range(0, len(x1)-1):
                    if model1 == None:
                        model1 = IncrementalQR_Cython(x1[i:i+1], y1[i])
                    else:
                        model1.update_qr(x1[i], y1[i][0])

                    if (x1_values[i, variable] != x1_values[i+1, variable]):
                        # Then its a valid split value
                        y_pred1 = model1.predict(x1_clean[0:i+1])
                        mse1 = np.mean((y1[0:i+1].flatten() - y_pred1)**2)

                        mse_val = (mse1*(i+1) + mse2_list[i]*(len(x1)-(i+1))) / x1.shape[0] # Mean Squared Error of this split
                        if mse_val < optimal_MSE:
                            optimal_MSE = mse_val
                            optimal_var = variable
                            optimal_split = (x1_values[i, variable] + x1_values[i+1, variable])/2
        return optimal_var, optimal_split


    # Get the optimal split point and variable using mse and return it
    # Does validation if it is selected and if number of datapoints is greater than 
    # or equal to points_to_validate
    # (or else there would not be enough validation data)
    def get_chosen_split(ForwardPasser self, subset_method, vars_to_consider, points_to_validate):
        if len(self.basis) >= self.B.shape[0]:
            # Then we have more basis functions than datapoints, so we shouldn't split
            return None, None

        if subset_method == 'linreg':
            optimal_var, optimal_split = self.subset_MSE_NP(vars_to_consider)
        elif subset_method == 'QR_cat':
            optimal_var, optimal_split = self.subset_MSE_QR_Cat(vars_to_consider)
        elif subset_method == 'linreg_validation':
            if self.X.shape[0] >= points_to_validate:
                if self.verbose >= 1:
                    print("Doing: linreg_validation")
                optimal_var, optimal_split = self.subset_MSE_NP_validation(vars_to_consider)
            else:
                if self.verbose >= 1:
                    print("Doing: linreg")
                optimal_var, optimal_split = self.subset_MSE_NP(vars_to_consider)
        elif subset_method == 'QR_cat_validation':
            if self.X.shape[0] >= points_to_validate:
                if self.verbose >= 1:
                    print("Doing QR_cat_validation")
                optimal_var, optimal_split = self.subset_MSE_QR_Cat_validation(vars_to_consider)
            else:
                if self.verbose >= 1:
                    print("Doing QR_cat")
                optimal_var, optimal_split = self.subset_MSE_QR_Cat(vars_to_consider)
        else:
            raise ValueError("subset_method must be: linreg, linreg_validation, QR_cat, or QR_cat_validation Got: ", subset_method)

        return optimal_var, optimal_split


    def subset_MSE_Cross_Val(ForwardPasser self, subset_method, vars_to_consider, points_to_validate):
        optimal_var, optimal_split = self.get_chosen_split(subset_method, vars_to_consider, points_to_validate)
        
        if optimal_var == None:
            return None, None 

        if len(self.B) > 1:
            MSE_List = np.array([])
            kf = KFold(n_splits=min(5, len(self.B)), shuffle=True, random_state=42)
            for fold, (train_index, test_index) in enumerate(kf.split(self.B)):
                # Split the data into training and testing sets for this fold
                X_train, X_test = self.B[train_index], self.B[test_index]
                y_train, y_test = self.y[train_index], self.y[test_index]
                mse_comparison = np.mean((y_test - X_test @ np.linalg.lstsq(X_train, y_train, rcond=-2)[0])**2)
                MSE_List = np.append(MSE_List, mse_comparison)
            original_MSE = np.mean(MSE_List)
        else:
            # Don't split, we have 1 datapoint!
            original_MSE = 0
            return None, None

        x1 = self.B[self.X[:, optimal_var] <= optimal_split, 0:len(self.basis)]
        y1 = self.y[self.X[:, optimal_var] <= optimal_split, :]
        x2 = self.B[self.X[:, optimal_var] >  optimal_split, 0:len(self.basis)]
        y2 = self.y[self.X[:, optimal_var] >  optimal_split, :]
        if len(x1) > 1:
            MSE_List = np.array([])
            kf = KFold(n_splits=min(5, len(x1)), shuffle=True, random_state=42)
            for fold, (train_index, test_index) in enumerate(kf.split(x1)):
                # Split the data into training and testing sets for this fold
                X_train, X_test = x1[train_index], x1[test_index]
                y_train, y_test = y1[train_index], y1[test_index]
                mse1_comparison = np.mean((y_test - X_test @ np.linalg.lstsq(X_train, y_train, rcond=-2)[0])**2)
                MSE_List = np.append(MSE_List, mse1_comparison)
            mse1 = np.mean(MSE_List)
        else:
            mse1 = original_MSE
            
        if len(x2) > 1:
            MSE_List = np.array([])
            kf = KFold(n_splits=min(5, len(x2)), shuffle=True, random_state=42)
            for fold, (train_index, test_index) in enumerate(kf.split(x2)):
                # Split the data into training and testing sets for this fold
                X_train, X_test = x2[train_index], x2[test_index]
                y_train, y_test = y2[train_index], y2[test_index]
                mse2_comparison = np.mean((y_test - X_test @ np.linalg.lstsq(X_train, y_train, rcond=-2)[0])**2)
                MSE_List = np.append(MSE_List, mse2_comparison)
            mse2 = np.mean(MSE_List)
        else:
            mse2 = original_MSE

        
        split_MSE = (1+(0.01))*((mse1*len(x1) + mse2*len(x2))/(len(x1)+len(x2)))
        if self.verbose >= 1:
            print("Considering the split (var, value): ", optimal_var, optimal_split )
            print("Pre split cross val MSE ", original_MSE)
            print("Post split cross val MSE : ", split_MSE)
            
            
        if split_MSE < original_MSE:
            if self.verbose >= 1:
                print("Accepted split")
            return optimal_var, optimal_split
        else:
            if self.verbose >= 1:
                print("Rejected split")
            return None, None


    cdef next_pair(ForwardPasser self):
        cdef INDEX_t variable
        cdef INDEX_t parent_idx
        cdef INDEX_t parent_degree
        cdef INDEX_t nonzero_count
        cdef BasisFunction parent
        cdef cnp.ndarray[INDEX_t, ndim = 1] candidates_idx
        cdef FLOAT_t knot
        cdef FLOAT_t mse
        cdef INDEX_t knot_idx
        cdef FLOAT_t knot_choice
        cdef FLOAT_t mse_choice
        cdef FLOAT_t mse_choice_cur_parent
        cdef int variable_choice_cur_parent
        cdef int knot_idx_choice
        cdef INDEX_t parent_idx_choice
        cdef BasisFunction parent_choice
        cdef BasisFunction new_parent
        cdef BasisFunction new_basis_function
        parent_basis_content_choice = None
        parent_basis_content = None
        cdef INDEX_t variable_choice
        cdef bint first = True
        cdef bint already_covered
        cdef INDEX_t k = len(self.basis)
        cdef INDEX_t endspan
        cdef bint linear_dependence
        cdef bint dependent
        # TODO: Shouldn't there be weights here?
        cdef FLOAT_t gcv_factor_k_plus_1 = gcv_adjust(k + 1, self.m,
                                                      self.penalty)
        cdef FLOAT_t gcv_factor_k_plus_2 = gcv_adjust(k + 2, self.m,
                                                      self.penalty)
        cdef FLOAT_t gcv_factor_k_plus_3 = gcv_adjust(k + 3, self.m,
                                                      self.penalty)
        cdef FLOAT_t gcv_factor_k_plus_4 = gcv_adjust(k + 4, self.m,
                                                      self.penalty)
        cdef FLOAT_t gcv_
        cdef FLOAT_t mse_
        cdef INDEX_t i
        cdef bint eligible
        cdef bint covered
        cdef bint missing_flag
        cdef bint choice_needs_coverage

        cdef cnp.ndarray[FLOAT_t, ndim = 2] X = (
            <cnp.ndarray[FLOAT_t, ndim = 2] > self.X)
        cdef cnp.ndarray[BOOL_t, ndim = 2] missing = (
            <cnp.ndarray[BOOL_t, ndim = 2] > self.missing)
        cdef cnp.ndarray[FLOAT_t, ndim = 2] B = (
            <cnp.ndarray[FLOAT_t, ndim = 2] > self.B)
        cdef cnp.ndarray[INT_t, ndim = 1] linear_variables = (
            <cnp.ndarray[INT_t, ndim = 1] > self.linear_variables)
        cdef cnp.ndarray[BOOL_t, ndim = 1] has_missing = (
            <cnp.ndarray[BOOL_t, ndim = 1] > self.has_missing)
        cdef cnp.ndarray[FLOAT_t, ndim = 1] b
        cdef cnp.ndarray[FLOAT_t, ndim = 1] p
        cdef bint variable_can_be_linear

        if self.use_fast:
            nb_basis = min(self.fast_K, k, len(self.fast_heap))
        else:
            nb_basis = k

        content_to_be_repushed = []
        for idx in range(nb_basis):
            # Iterate over parents
            if self.use_fast:
                # retrieve the next basis function to try as parent
                parent_basis_content = heappop(self.fast_heap)
                content_to_be_repushed.append(parent_basis_content)
                parent_idx = parent_basis_content.idx
                mse_choice_cur_parent = -1
                variable_choice_cur_parent = -1
            else:
                parent_idx = idx

            parent = self.basis.get(parent_idx)
            if not parent.is_splittable():
                continue

            if self.use_fast:
                # each "fast_h" iteration, force to pass through all the variables,
                if self.iteration_number - parent_basis_content.m >= self.fast_h:
                    variables = range(self.n)
                    parent_basis_content.m = self.iteration_number
                # in the opposite case, just use the last chosen variable
                else:
                    if parent_basis_content.v is not None:
                        variables = [parent_basis_content.v]
                    else:
                        variables = range(self.n)
            else:
                variables = range(self.n)

            parent_degree = parent.effective_degree()

            for variable in variables:
                # Determine whether this variable can be linear
                variable_can_be_linear = self.allow_linear and not parent.has_linear(variable)

                # Determine whether missingness needs to be accounted for.
                if self.allow_missing and has_missing[variable]:
                    missing_flag = True
                    eligible = parent.eligible(variable)
                    covered = parent.covered(variable)
                else:
                    missing_flag = False

                # Make sure not to exceed max_degree (but don't count the
                # covering missingness basis function if required)
                if self.max_degree >= 0:
                    if parent_degree >= self.max_degree:
                        continue

                # If there is missing data and this parent is not
                # an eligible parent for this variable with missingness
                # (because it includes a non-missing factor for the variable)
                # then skip this variable.
                if missing_flag and not eligible:
                    continue

                # Add the linear term to B
                predictor = self.predictors[variable]

#                 # If necessary, protect from missing data
#                 if missing_flag:
#                     B[missing[:, variable]==1, k] = 0.
#                     b = B[:, k]
#                     # Update the outcome data
#                     linear_dependence = self.orthonormal_update(b)

                if missing_flag and not covered:
                    p = B[:, parent_idx] * (1 - missing[:, variable])
                    b = B[:, parent_idx] * (1 - missing[:, variable])
                    self.orthonormal_update(b)
                    b = B[:, parent_idx] * missing[:, variable]
                    self.orthonormal_update(b)
                    q = k + 3
                else:
                    p = self.B[:, parent_idx]
                    q = k + 1

                b = p * predictor.x
                if missing_flag and not covered:
                    b[missing[:, variable] == 1] = 0
                linear_dependence = self.orthonormal_update(b)

                # If a new hinge function does not improve the gcv over the
                # linear term then just the linear term will be retained
                # (if variable_can_be_linear).  Calculate the gcv with just the linear
                # term in order to compare later.  Note that the mse with
                # another term never increases, but the gcv may because it
                # penalizes additional terms.
                mse_ = self.outcome.mse()
                if missing_flag and not covered:
                    gcv_ = gcv_factor_k_plus_3 * mse_
                else:
                    gcv_ = gcv_factor_k_plus_1 * mse_

                if linear_variables[variable]:
                    mse = mse_
                    knot_idx = -1
                else:
                    # Find the valid knot candidates
                    
                    candidates, candidates_idx = predictor.knot_candidates(p, self.endspan,
                                                                           self.minspan,
                                                                           self.minspan_alpha,
                                                                           self.n, set(parent.knots(variable)))
                    
                    # Choose the best candidate (if no candidate is an
                    # improvement on the linear term in terms of gcv, knot_idx
                    # is set to -1
                    if len(candidates_idx) > 0:
#                         candidates = np.array(predictor.x)[candidates_idx]

                        # Find the best knot location for this parent and
                        # variable combination
                        # Assemble the knot search data structure
                        constant = KnotSearchReadOnlyData(predictor, self.outcome)
                        search_data = KnotSearchData(constant, self.workings, q)

                        # Run knot search
                        knot, knot_idx, mse = knot_search(search_data, candidates, p, q,
                                                          self.m, len(candidates), self.n_outcomes,
                                                          self.verbose)
                        mse /= self.total_weight
                        knot_idx = candidates_idx[knot_idx]

                        # If the hinge function does not decrease the gcv then
                        # just keep the linear term (if variable_can_be_linear is True)
                        if variable_can_be_linear:
                            if missing_flag and not covered:
                                if gcv_factor_k_plus_4 * mse >= gcv_:
                                    mse = mse_
                                    knot_idx = -1
                            else:
                                if gcv_factor_k_plus_2 * mse >= gcv_:
                                    mse = mse_
                                    knot_idx = -1
                    else:
                        if variable_can_be_linear:
                            mse = mse_
                            knot_idx = -1
                        else:
                            # Do an orthonormal downdate and skip to the next
                            # iteration
                            if missing_flag and not covered:
                                self.orthonormal_downdate()
                                self.orthonormal_downdate()
                            self.orthonormal_downdate()
                            continue

                # Do an orthonormal downdate
                if missing_flag and not covered:
                    self.orthonormal_downdate()
                    self.orthonormal_downdate()
                self.orthonormal_downdate()

                # Update the choices
                if mse < mse_choice or first:
                    if first:
                        first = False
                    knot_choice = knot
                    mse_choice = mse
                    knot_idx_choice = knot_idx
                    parent_idx_choice = parent_idx
                    parent_choice = parent
                    if self.use_fast is True:
                        parent_basis_content_choice = parent_basis_content
                    variable_choice = variable
                    dependent = linear_dependence
                    if missing_flag and not covered:
                        choice_needs_coverage = True
                    else:
                        choice_needs_coverage = False

                if self.use_fast is True:
                    if (mse_choice_cur_parent == -1) or \
                       (mse < mse_choice_cur_parent):
                        mse_choice_cur_parent = mse
                        variable_choice_cur_parent = variable
            if self.use_fast is True:
                if mse_choice_cur_parent != -1:
                    parent_basis_content.mse = mse_choice_cur_parent
                    parent_basis_content.v = variable_choice_cur_parent

        if self.use_fast is True:
            for content in content_to_be_repushed:
                heappush(self.fast_heap, content)

        # Make sure at least one candidate was checked
        if first:
            self.record[len(self.record) - 1].set_no_candidates(True)
            return

        # Add the new basis functions
        label = self.xlabels[variable_choice]
        if self.use_fast is True:
            parent_basis_content_choice.m = -np.inf
        if choice_needs_coverage:
            new_parent = parent_choice.get_coverage(variable_choice)
            if new_parent is None:
                new_basis_function = MissingnessBasisFunction(parent_choice, variable_choice,
                                               True, label)
                new_basis_function.apply(X, missing, B[:, len(self.basis)])
                self.orthonormal_update(B[:, len(self.basis)])
                if self.use_fast and new_basis_function.is_splittable() and new_basis_function.effective_degree() < self.max_degree:
                    content = FastHeapContent(idx=len(self.basis))
                    heappush(self.fast_heap, content)
                self.basis.append(new_basis_function)
                new_parent = new_basis_function

                new_basis_function = MissingnessBasisFunction(parent_choice, variable_choice,
                                               False, label)
                new_basis_function.apply(X, missing, B[:, len(self.basis)])
                self.orthonormal_update(B[:, len(self.basis)])
                if self.use_fast and new_basis_function.is_splittable() and new_basis_function.effective_degree() < self.max_degree:
                    content = FastHeapContent(idx=len(self.basis))
                    heappush(self.fast_heap, content)
                self.basis.append(new_basis_function)
        else:
            new_parent = parent_choice
        
        if knot_idx_choice != -1:
            # Add the new basis functions
            new_basis_function = HingeBasisFunction(new_parent,
                                     knot_choice, knot_idx_choice,
                                     variable_choice,
                                     False, label)
            new_basis_function.apply(X, missing, B[:, len(self.basis)])
            self.orthonormal_update(B[:, len(self.basis)])
            if self.use_fast and new_basis_function.is_splittable() and new_basis_function.effective_degree() < self.max_degree:
                content = FastHeapContent(idx=len(self.basis))
                heappush(self.fast_heap, FastHeapContent(idx=len(self.basis)))
            self.basis.append(new_basis_function)

            new_basis_function = HingeBasisFunction(new_parent,
                                     knot_choice, knot_idx_choice,
                                     variable_choice,
                                     True, label)
            new_basis_function.apply(X, missing, B[:, len(self.basis)])
            self.orthonormal_update(B[:, len(self.basis)])
            if self.use_fast and new_basis_function.is_splittable() and new_basis_function.effective_degree() < self.max_degree:
                content = FastHeapContent(idx=len(self.basis))
                heappush(self.fast_heap, content)
            self.basis.append(new_basis_function)

        elif not dependent and knot_idx_choice == -1:
            # In this case, only add the linear basis function (in addition to
            # covering missingness basis functions if needed)
            new_basis_function = LinearBasisFunction(new_parent, variable_choice, label)
            new_basis_function.apply(X, missing, B[:, len(self.basis)])
            self.orthonormal_update(B[:, len(self.basis)])
            if self.use_fast and new_basis_function.is_splittable() and new_basis_function.effective_degree() < self.max_degree:
                content = FastHeapContent(idx=len(self.basis))
                heappush(self.fast_heap, content)
            self.basis.append(new_basis_function)
        else:  # dependent and knot_idx_choice == -1
            # In this case there were no acceptable choices remaining, so end
            # the forward pass
            self.record[len(self.record) - 1].set_no_candidates(True)
            return

        self.variables_chosen.append(variable_choice)

        # Compute the new mse, which is the result of the very stable
        # orthonormal updates and not the mse that comes directly from
        # the knot search
        cdef FLOAT_t final_mse = self.outcome.mse()

        # Update the build record
        self.record.append(ForwardPassIteration(parent_idx_choice,
                                                variable_choice,
                                                knot_idx_choice, final_mse,
                                                len(self.basis)))

    # If forward passer is done with forward pass, 
    # then we use this to make it into a tree
    def next_split(ForwardPasser self, subset_method, vars_to_consider, points_to_validate):
        optimal_var, optimal_split = self.subset_MSE_Cross_Val(subset_method, vars_to_consider, points_to_validate)
        if optimal_var == None and optimal_split == None:
            # Did not split
            return
        else:
            # Did in fact split!
            self.execute_split(optimal_var, optimal_split)
            #Then finish
            return


    def execute_split(ForwardPasser self, optimal_var, optimal_split):
        # Did in fact split!
        forward_left, forward_right = self.create_optimal_forward_passers(optimal_split, optimal_var)
        self.Node.split_var = optimal_var
        self.Node.split_val = optimal_split

        #Set left child node
        self.Node.left_child = Node()
        forward_left.set_node(self.Node.left_child)
        self.Node.left_child.parent = self.Node
        self.Node.left_child.forward_passer = forward_left
        self.Node.left_child.max_depth = self.Node.max_depth
        self.Node.left_child.points_to_validate = self.Node.points_to_validate
        self.Node.left_child.X = self.X[self.X[:, int(optimal_var)] <= optimal_split, :]
        self.Node.left_child.y = self.y[self.X[:, int(optimal_var)] <= optimal_split, :]
        self.Node.left_child.missing = self.missing[self.X[:, int(optimal_var)] <= optimal_split, :]
        self.Node.left_child.sample_weight = self.sample_weight[self.X[:, int(optimal_var)] <= optimal_split, :]
        self.Node.left_child.n = self.Node.left_child.X.shape[0]

        #Set right child node
        self.Node.right_child = Node()
        forward_right.set_node(self.Node.right_child)
        self.Node.right_child.parent = self.Node
        self.Node.right_child.forward_passer = forward_right
        self.Node.right_child.max_depth = self.Node.max_depth
        self.Node.right_child.points_to_validate = self.Node.points_to_validate
        self.Node.right_child.X = self.X[self.X[:, int(optimal_var)] > optimal_split, :]
        self.Node.right_child.y = self.y[self.X[:, int(optimal_var)] > optimal_split, :]
        self.Node.right_child.missing = self.missing[self.X[:, int(optimal_var)] > optimal_split, :]
        self.Node.right_child.sample_weight = self.sample_weight[self.X[:, int(optimal_var)] > optimal_split, :]
        self.Node.right_child.n = self.Node.right_child.X.shape[0]

    def set_node(ForwardPasser self, Node):
        self.Node = Node

    def create_optimal_forward_passers(ForwardPasser self, optimal_val, optimal_var):

        forward1_optimal = ForwardPasser(self.X[self.X[:, int(optimal_var)] <= optimal_val, :], self.missing[self.X[:, int(optimal_var)] <= optimal_val, :], 
                            self.y[self.X[:, int(optimal_var)] <= optimal_val, :], self.sample_weight[self.X[:, int(optimal_var)] <= optimal_val, :])
        forward2_optimal = ForwardPasser(self.X[self.X[:, int(optimal_var)] > optimal_val, :], self.missing[self.X[:, int(optimal_var)] > optimal_val, :], 
                            self.y[self.X[:, int(optimal_var)] > optimal_val, :], self.sample_weight[self.X[:, int(optimal_var)] > optimal_val, :])

        forward1_optimal.set_basis(self.get_basis())
        forward2_optimal.set_basis(self.get_basis())
            
        forward1_optimal.B = self.B[self.X[:, int(optimal_var)] <= optimal_val, :]
        forward2_optimal.B = self.B[self.X[:, int(optimal_var)] > optimal_val, :]

        if len(forward1_optimal.basis) > len(forward1_optimal.B[0]):
            raise ValueError("WOULD HAVE FAILED!")
        if len(forward2_optimal.basis) > len(forward2_optimal.B[0]):
            raise ValueError("WOULD HAVE FAILED!")

        for i in range(1, len(forward1_optimal.basis)):
            if i < len(forward1_optimal.B):
                forward1_optimal.orthonormal_update(forward1_optimal.B[:, i])
        for i in range(1, len(forward2_optimal.basis)):
            # Don't apply more bases than rows
            if i < len(forward2_optimal.B):
                forward2_optimal.orthonormal_update(forward2_optimal.B[:, i])
        
        
        # About to update record
        forward1_optimal.record = copy.deepcopy(self.record)
        forward1_optimal.variables_chosen = self.variables_chosen
        forward1_optimal.verbose = self.verbose
        forward2_optimal.record = copy.deepcopy(self.record)
        forward2_optimal.variables_chosen = self.variables_chosen
        forward2_optimal.verbose = self.verbose

        return forward1_optimal, forward2_optimal


    ######################
    ##### NODE CLASS #####
    ######################
class Node:
    def __init__(self):
        self.left_child = None
        self.right_child = None
        self.split_var = None
        self.split_val = None
        self.forward_passer = None
        self.node_GCV = None
        self.Earth = None
        self.n = None
        self.max_depth = None
        self.points_to_validate = None # Number of datapoints needed to do validation instead of training data
        
        self.parent = None

        self.X = None
        self.y = None
        self.sample_weight = None
        self.missing = None

    def set_forward_passer(self, forward_passer):
        self.forward_passer = forward_passer

    # Build tree after making polynomial
    def run_forward_pass(self, allow_subset, subset_method, vars_to_consider):
        self.forward_passer.run() # Make standard polynomial
        if allow_subset == True:
            self.run_forward_pass_helper(0, subset_method, vars_to_consider)

    def run_forward_pass_helper(self, current_depth, subset_method, vars_to_consider):
        if self.max_depth != None and current_depth >= self.max_depth:
            return
        self.forward_passer.next_split(subset_method, vars_to_consider, self.points_to_validate)
        if self.left_child == None and self.right_child == None:
            return
        else:
            self.left_child.run_forward_pass_helper(current_depth + 1, subset_method, vars_to_consider)
            self.right_child.run_forward_pass_helper(current_depth + 1, subset_method, vars_to_consider)

