UTIL = """
extern "C" {
    const int K = 4;
    
        __forceinline__ __device__ 
    float plus_forward(int type, 
                       float x1, 
                       float x2) {
    
        switch (type) {
            // plus_times
            case 0:
                return x1 + x2;
            // max_plus
            case 1:
                return max(x1, x2);
            case 2:
                return max(x1, x2);
        } 
        return x1 + x2;
    }
    
     __forceinline__ __device__ 
    void plus_backward(int type, 
                       float x1, 
                       float x2,
                       float dEdf,
                       float &dEdx1,
                       float &dEdx2) {
    
        switch (type) {
            // plus_times
            case 0:
                dEdx1 = dEdx2 = dEdf;
                break;
            // max_plus
            case 1:
                dEdx1 = (x1 >= x2) ? dEdf : 0.f;
                dEdx2 = (x1 >= x2) ? 0.f : dEdf;
                break;
            case 2:
                dEdx1 = (x1 >= x2) ? dEdf : 0.f;
                dEdx2 = (x1 >= x2) ? 0.f : dEdf;
                break;
        }
        return;
    }
    
    
    __forceinline__ __device__ 
    float times_forward(int type, 
                        float x1, 
                        float x2) {
    
        switch (type) {
            // plus_times
            case 0:
                return x1 * x2;
            // max_plus
            case 1:
                return x1 + x2;
            case 2:
                return x1 * x2;
        }
        return x1 * x2;
    }
    
    
    __forceinline__ __device__ 
    void times_backward(int type, 
                        float x1, 
                        float x2,
                        float dEdf,
                        float &dEdx1,
                        float &dEdx2) {
    
        switch (type) {
            // plus_times
            case 0:
                dEdx1 = dEdf * x2;
                dEdx2 = dEdf * x1;
                break;
            // max_plus
            case 1:
                dEdx1 = dEdx2 = dEdf;
                break;
            case 2:
                dEdx1 = dEdf * x2;
                dEdx2 = dEdf * x1;
                break;
        }
        return;
    }
    
    __forceinline__ __device__ float sigmoidf(float x) {
        return 1.f / (1.f + expf(-x));
    }

    __forceinline__ __device__ float reluf(float x) {
        return (x > 0.f) ? x : 0.f;
    }

    __forceinline__ __device__ float seluf(float x) {
        return 1.0507009873554804934193349852946f * (
            (x > 0.f) ? x : 1.6732632423543772848170429916717f * (expf(x)-1.f)
        );
    }

    __forceinline__ __device__ float calc_activation(int type, float x) {
        switch (type) {
            case 0:
                return x;
            case 1:
                return tanh(x);
            case 2:
                return reluf(x);
            case 3:
                return seluf(x);
        }
        return x;
    }

    __forceinline__ __device__ float calc_grad_activation(int type, float x) {
        switch (type) {
            case 0:
                return 1.f;
            case 1:
                return 1.f-x*x;
            case 2:
                return (x > 0.f) ? 1.f : 0.f;
            case 3:
                return (x > 0.f) ? 1.0507009873554804934193349852946f :
                    x + 1.7580993408473766f;
        }
        return 1.f;
    }
}
"""
