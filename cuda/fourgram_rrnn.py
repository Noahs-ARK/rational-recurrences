FOURGRAM_RRNN = """
            
extern "C" {
     __global__ void rrnn_fwd(
                const float * __restrict__ u, 
                const float * __restrict__ c1_init,
                const float * __restrict__ c2_init,
                const float * __restrict__ c3_init,
                const float * __restrict__ c4_init,
                const int len, 
                const int batch,
                const int dim,
                const int k,
                float * __restrict__ c1,
                float * __restrict__ c2,
                float * __restrict__ c3,
                float * __restrict__ c4,
                int semiring_type) {
        int ncols = batch*dim;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col >= ncols) return;
        int ncols_u = ncols*k;

        const float *up = u + (col*k);
        float *c1p = c1 + col;
        float *c2p = c2 + col;
        float *c3p = c3 + col;
        float *c4p = c4 + col;
        float cur_c1 = *(c1_init + col);
        float cur_c2 = *(c2_init + col);
        float cur_c3 = *(c3_init + col);
        float cur_c4 = *(c4_init + col);
        
        for (int row = 0; row < len; ++row) {
            float u1 = *(up);
            float u2 = *(up+1);
            float u3 = *(up+2);
            float u4 = *(up+3);

            float forget1 = *(up+4);
            float forget2 = *(up+5);
            float forget3 = *(up+6);
            float forget4 = *(up+7);
            
            float prev_c1 = cur_c1;
            float prev_c2 = cur_c2;
            float prev_c3 = cur_c3;
            cur_c1 = cur_c1 * forget1 + u1;
            cur_c2 = cur_c2 * forget2 + (prev_c1) * u2;
            cur_c3 = cur_c3 * forget3 + (prev_c2) * u3;
            cur_c4 = cur_c4 * forget4 + (prev_c3) * u4;
            
            *c1p = cur_c1;
            *c2p = cur_c2;
            *c3p = cur_c3;
            *c4p = cur_c4;
            
            up += ncols_u;
            c1p += ncols;
            c2p += ncols;
            c3p += ncols;
            c4p += ncols;
        }
    }
    
    __global__ void rrnn_bwd(
                const float * __restrict__ u, 
                const float * __restrict__ c1_init,
                const float * __restrict__ c2_init,
                const float * __restrict__ c3_init,
                const float * __restrict__ c4_init,
                const float * __restrict__ c1,
                const float * __restrict__ c2,
                const float * __restrict__ c3,
                const float * __restrict__ c4,
                const float * __restrict__ grad_c1, 
                const float * __restrict__ grad_c2, 
                const float * __restrict__ grad_c3, 
                const float * __restrict__ grad_c4, 
                const float * __restrict__ grad_last_c1,
                const float * __restrict__ grad_last_c2,
                const float * __restrict__ grad_last_c3,
                const float * __restrict__ grad_last_c4,
                const int len, 
                const int batch, 
                const int dim, 
                const int k,
                float * __restrict__ grad_u, 
                float * __restrict__ grad_c1_init,
                float * __restrict__ grad_c2_init,
                float * __restrict__ grad_c3_init,
                float * __restrict__ grad_c4_init,
                int semiring_type) {
        int ncols = batch*dim;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col >= ncols) return;

        int ncols_u = ncols*k;

        float cur_c1 = *(grad_last_c1 + col);
        float cur_c2 = *(grad_last_c2 + col);
        float cur_c3 = *(grad_last_c3 + col);
        float cur_c4 = *(grad_last_c4 + col);

        const float *up = u + (col*k) + (len-1)*ncols_u;
        const float *c1p = c1 + col + (len-1)*ncols;
        const float *c2p = c2 + col + (len-1)*ncols;
        const float *c3p = c3 + col + (len-1)*ncols;
        const float *c4p = c4 + col + (len-1)*ncols;
        
        const float *gc1p = grad_c1 + col + (len-1)*ncols;
        const float *gc2p = grad_c2 + col + (len-1)*ncols;
        const float *gc3p = grad_c3 + col + (len-1)*ncols;
        const float *gc4p = grad_c4 + col + (len-1)*ncols;
        float *gup = grad_u + (col*k) + (len-1)*ncols_u;
        
        for (int row = len-1; row >= 0; --row) {
            float u1 = *(up);
            float u2 = *(up+1);
            float u3 = *(up+2);
            float u4 = *(up+3);
            float forget1 = *(up+4);
            float forget2 = *(up+5);
            float forget3 = *(up+6);
            float forget4 = *(up+7);
        
            const float c1_val = *c1p;
            const float c2_val = *c2p;
            const float c3_val = *c3p;
            const float c4_val = *c4p;
            
            const float prev_c1_val = (row>0) ? (*(c1p-ncols)) : (*(c1_init+col));
            const float prev_c2_val = (row>0) ? (*(c2p-ncols)) : (*(c2_init+col));
            const float prev_c3_val = (row>0) ? (*(c3p-ncols)) : (*(c3_init+col));
            const float prev_c4_val = (row>0) ? (*(c4p-ncols)) : (*(c4_init+col));
            
            const float gc1 = *(gc1p) + cur_c1;
            const float gc2 = *(gc2p) + cur_c2;
            const float gc3 = *(gc3p) + cur_c3;
            const float gc4 = *(gc4p) + cur_c4;
            
            float gu1 = gc1;
            *(gup) = gu1;
            float gforget1 = gc1*prev_c1_val;
            *(gup+4) = gforget1;
            
            float gu2 = gc2*(prev_c1_val);
            *(gup+1) = gu2;
            float gforget2 = gc2*prev_c2_val;
            *(gup+5) = gforget2;

            float gu3 = gc3*(prev_c2_val);
            *(gup+2) = gu3;
            float gforget3 = gc3*prev_c3_val;
            *(gup+6) = gforget3;
            
            float gu4 = gc4*(prev_c3_val);
            *(gup+3) = gu4;
            float gforget4 = gc4*prev_c4_val;
            *(gup+7) = gforget4;

            cur_c1 = gc1 * forget1 + gc2 * u2;
            cur_c2 = gc2 * forget2 + gc3 * u3;
            cur_c3 = gc3 * forget3 + gc4 * u4;
            cur_c4 = gc4 * forget4;

            up -= ncols_u; 
            c1p -= ncols;
            c2p -= ncols;
            c3p -= ncols;
            c4p -= ncols;
            gup -= ncols_u;
            gc1p -= ncols;
            gc2p -= ncols;
            gc3p -= ncols;
            gc4p -= ncols;
        }
        
        *(grad_c1_init + col) = cur_c1;
        *(grad_c2_init + col) = cur_c2;
        *(grad_c3_init + col) = cur_c3;
        *(grad_c4_init + col) = cur_c4;
    }
}
"""
