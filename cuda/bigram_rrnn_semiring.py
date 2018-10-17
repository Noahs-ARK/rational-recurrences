BIGRAM_RRNN_SEMIRING = """

extern "C" {
     __global__ void rrnn_semiring_fwd(
                const float * __restrict__ u, 
                const float * __restrict__ eps, 
                const float * __restrict__ c1_init,
                const float * __restrict__ c2_init,
                const int len, 
                const int batch, 
                const int dim, 
                const int k,
                float * __restrict__ c1,
                float * __restrict__ c2,
                int semiring_type) {
        assert (k == K);
        int ncols = batch*dim;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col >= ncols) return;
        int ncols_u = ncols*k;

        const float *up = u + (col*k);
        float *c1p = c1 + col;
        float *c2p = c2 + col;
        float cur_c1 = *(c1_init + col);
        float cur_c2 = *(c2_init + col);
        const float eps_val = *(eps + (col%dim));

        for (int row = 0; row < len; ++row) {
            float u1 = *(up);
            float u2 = *(up+1);
            float forget1 = *(up+2);
            float forget2 = *(up+3);
            
            float prev_c1 = cur_c1;
            float op1 = times_forward(semiring_type, cur_c1, forget1);
            cur_c1 = plus_forward(semiring_type, op1, u1);

            float op2 = times_forward(semiring_type, cur_c2, forget2);
            float op3_ = plus_forward(semiring_type, eps_val, prev_c1);
            float op3 = times_forward(semiring_type, op3_, u2);
            cur_c2 = plus_forward(semiring_type, op2, op3);

            *c1p = cur_c1;
            *c2p = cur_c2;

            up += ncols_u;
            c1p += ncols;
            c2p += ncols;
        }
    }

    __global__ void rrnn_semiring_bwd(
                const float * __restrict__ u, 
                const float * __restrict__ eps, 
                const float * __restrict__ c1_init,
                const float * __restrict__ c2_init,
                const float * __restrict__ c1,
                const float * __restrict__ c2,
                const float * __restrict__ grad_c1, 
                const float * __restrict__ grad_c2, 
                const float * __restrict__ grad_last_c1,
                const float * __restrict__ grad_last_c2,
                const int len, 
                const int batch, 
                const int dim, 
                const int k,
                float * __restrict__ grad_u, 
                float * __restrict__ grad_eps, 
                float * __restrict__ grad_c1_init,
                float * __restrict__ grad_c2_init,
                int semiring_type) {
        assert (k == K);
        int ncols = batch*dim;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col >= ncols) return;

        int ncols_u = ncols*k;

        float cur_c1 = *(grad_last_c1 + col);
        float cur_c2 = *(grad_last_c2 + col);
        const float eps_val = *(eps + (col%dim));

        const float *up = u + (col*k) + (len-1)*ncols_u;
        const float *c1p = c1 + col + (len-1)*ncols;
        const float *c2p = c2 + col + (len-1)*ncols;

        const float *gc1p = grad_c1 + col + (len-1)*ncols;
        const float *gc2p = grad_c2 + col + (len-1)*ncols;
        float *gup = grad_u + (col*k) + (len-1)*ncols_u;
        float geps = 0.f;

        for (int row = len-1; row >= 0; --row) {
            float u1 = *(up);
            float u2 = *(up+1);
            float forget1 = *(up+2);
            float forget2 = *(up+3);

            const float c1_val = *c1p;
            const float c2_val = *c2p;

            const float prev_c1 = (row>0) ? (*(c1p-ncols)) : (*(c1_init+col));
            const float prev_c2 = (row>0) ? (*(c2p-ncols)) : (*(c2_init+col));

            const float gc1 = *(gc1p) + cur_c1;
            const float gc2 = *(gc2p) + cur_c2;
            cur_c1 = cur_c2 = 0.f;
            
            float op1 = times_forward(semiring_type, prev_c1, forget1);
            float gop1 = 0.f, gu1 = 0.f;
            plus_backward(semiring_type, op1, u1, gc1, gop1, gu1);
            float gprev_c1 = 0.f, gprev_c2 = 0.f, gforget1=0.f;
            times_backward(semiring_type, prev_c1, forget1, gop1, gprev_c1, gforget1);
            *(gup) = gu1;
            *(gup+2) = gforget1;
            cur_c1 += gprev_c1; 
            
            float op2 = times_forward(semiring_type, cur_c2, forget2);
            float op3_ = plus_forward(semiring_type, eps_val, prev_c1);
            float op3 =  times_forward(semiring_type, op3_, u2);
            float gop2 = 0.f, gop3 = 0.f;
            plus_backward(semiring_type, op2, op3, gc2, gop2, gop3);

            float gop3_ = 0.f, gu2 = 0.f, gforget2 = 0.f, cur_geps=0.f;
            times_backward(semiring_type, prev_c2, forget2, gop2, gprev_c2, gforget2);
            times_backward(semiring_type, op3_, u2, gop3, gop3_, gu2);
            plus_backward(semiring_type, eps_val, prev_c1, gop3_, cur_geps, gprev_c1);
            *(gup+1) = gu2;
            *(gup+3) = gforget2;
            geps += cur_geps;
            cur_c1 += gprev_c1;
            cur_c2 += gprev_c2;

            up -= ncols_u; 
            c1p -= ncols;
            c2p -= ncols;
            gup -= ncols_u;
            gc1p -= ncols;
            gc2p -= ncols;
        }

        *(grad_c1_init + col) = cur_c1;
        *(grad_c2_init + col) = cur_c2;
        *(grad_eps + col%dim) = geps;
    }
}
"""
