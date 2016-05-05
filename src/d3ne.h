//
// Created by m-ochi on 16/03/17.
//

#ifndef D3NE_DEV_D3NE_H
#define D3NE_DEV_D3NE_H

#include "pteself/src/pte.h"
#include <string>
#include <iostream>

class D3NE : public PTE
{
public:
    D3NE(
        std::string ww_net_a,
        std::string wd_net_a,
        std::string wl_net_a,
        std::string outputfile_a,
        int binary_a,
        int size_a,
        int order_a,
        int negative_a,
        int samples_a,
        int threads_a,
        float rho_a
    );

    ~D3NE();

    void run();

private:
    static std::unique_ptr<EmbeddedVector> emb_all;
//    static std::unique_ptr<EmbeddedVector> emb_all_wl;
//    static double alpha;

    static void trainD3NEThread(int id);

    static void* learnVector(
            Network* network_each,
            EmbeddedVector* emb_each,
            Samplers* sampler_each,
            Network* network_all_a,
            EmbeddedVector* emb_all_a,
            int start_idx,
            int end_idx,
            uint64_t seed
    );

    static void update(
            std::vector<real>& vec_u,
            std::vector<real>& vec_v,
            std::vector<real>& vec_error,
            int learn_dim,
            int label);

    /*
    static void* learnVector_ww(
            Network* network_wd_a, EmbeddedVector* emb_wd_a,
            Network* network_wl_a, EmbeddedVector* emb_wl_a,
            Samplers* sampler_ww_a,
            Network* network_ww_a,
            Network* network_all_a,
            EmbeddedVector* emb_all_wd_a,
            EmbeddedVector* emb_all_wl_a,
            uint64_t seed);

    static void update_ww(
            std::vector<real>& vec_u_wd, std::vector<real>& vec_v_wd,
            std::vector<real>& vec_u_wl, std::vector<real>& vec_v_wl,
            std::vector<real>& vec_error_wd,
            std::vector<real>& vec_error_wl,
            double* error_alpha,
            int label);
    */

    void output(std::string mode);
};


#endif //D3NE_DEV_D3NE_H
