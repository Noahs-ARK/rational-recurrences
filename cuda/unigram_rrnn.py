UNIGRAM_RRNN = """
            
extern "C" {
     __global__ void rrnn_fwd(
                const float * __restrict__ u, 
                const float * __restrict__ c_init,
                const int len, 
                const int batch, 
                const int dim, 
                const int k,
                float * __restrict__ c,
                int semiring_type) {
        assert (k == K);
        int ncols = batch*dim;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col >= ncols) return;
        int ncols_u = ncols*k;

        const float *up = u + (col*k);
        float *cp = c + col;
        float cur_c = *(c_init + col);
        const float eps_val = *(eps + (col%dim));
        
        for (int row = 0; row < len; ++row) {
            float u = *(up);
            float forget = *(up+1);
            float prev_c = cur_c;
            cur_c = cur_c * forget + u;
            *cp = cur_c;
            up += ncols_u;
            cp += ncols;
        }
    }
    
    __global__ void rrnn_bwd(
                const float * __restrict__ u, 
                const float * __restrict__ eps, 
                const float * __restrict__ c_init,
                const float * __restrict__ c,
                const float * __restrict__ grad_c, 
                const float * __restrict__ grad_last_c,
                const int len, 
                const int batch, 
                const int dim, 
                const int k,
                float * __restrict__ grad_u,
                float * __restrict__ grad_c_init,
                int semiring_type) {
        assert (k == K);
        int ncols = batch*dim;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col >= ncols) return;

        int ncols_u = ncols*k;

        float cur_c = *(grad_last_c + col);
        const float *up = u + (col*k) + (len-1)*ncols_u;
        const float *cp = c + col + (len-1)*ncols;
        
        const float *gcp = grad_c + col + (len-1)*ncols;
        float *gup = grad_u + (col*k) + (len-1)*ncols_u;
        
        for (int row = len-1; row >= 0; --row) {
            float u = *(up);
            float forget = *(up+1);
        
            const float c_val = *cp;
            const float prev_c_val = (row>0) ? (*(cp-ncols)) : (*(c_init+col));
            const float gc = *(gcp) + cur_c;
            
            float gu = gc;
            *(gup) = gu;
            float gforget = gc*prev_c_val;
            *(gup+1) = gforget;
            
            cur_c = gc * forget;

            up -= ncols_u; 
            cp -= ncols;
            gup -= ncols_u;
            gcp -= ncols;
        }
        
        *(grad_c_init + col) = cur_c;
    }
}
"""
