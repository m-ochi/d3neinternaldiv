//
// Created by m-ochi on 16/03/17.
//

#include "d3ne.h"

double D3NE::alpha;

std::unique_ptr<EmbeddedVector> D3NE::emb_all_wd;
std::unique_ptr<EmbeddedVector> D3NE::emb_all_wl;

D3NE::D3NE(
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
)
: PTE(
    ww_net_a,
    wd_net_a,
    wl_net_a,
    outputfile_a,
    binary_a,
    size_a,
    order_a,
    negative_a,
    samples_a,
    threads_a,
    rho_a
) {

}

D3NE::~D3NE() {}

void D3NE::run() {

//    std::cout << "DEBUG START" << std::endl;

    total_samples = this->constant_settings->samples * 1000000;

    if (constant_settings->order != 1 && constant_settings->order != 2) {
        std::cout << constant_settings->order << std::endl;
        printf("Error: order should be eighther 1 or 2!\n");
        exit(1);
    }
    printf("--------------------------------\n");
    printf("Order: %d\n", constant_settings->order);
    printf("Samples: %lldM\n", total_samples / 1000000);
    printf("Negative: %d\n", constant_settings->negative);
    printf("Dimension: %d\n", constant_settings->dim);
    printf("Initial rho: %lf\n", constant_settings->rho);
    printf("--------------------------------\n");

//    std::cout << "DEBUG -3" << std::endl;

    // networkの作成
    std::vector<std::string> networkfiles;
    networkfiles.push_back(this->constant_settings->ww_net);
    networkfiles.push_back(this->constant_settings->wd_net);
    networkfiles.push_back(this->constant_settings->wl_net);

    network_ww  = std::unique_ptr<Network>(new Network(networkfiles[0], "ww"));
    network_wd  = std::unique_ptr<Network>(new Network(networkfiles[1], "wd"));
    network_wl  = std::unique_ptr<Network>(new Network(networkfiles[2], "wl"));
    network_all = std::unique_ptr<Network>(new NetworkAll(networkfiles));

    // samplerの作成
    sampler_ww = std::unique_ptr<Samplers>(new Samplers(network_ww.get()));
    sampler_wd = std::unique_ptr<Samplers>(new Samplers(network_wd.get()));
    sampler_wl = std::unique_ptr<Samplers>(new Samplers(network_wl.get()));

    // embeddingの作成
    emb_ww  = std::unique_ptr<EmbeddedVector>(new EmbeddedVector(network_ww->num_vertices,  this->constant_settings->dim));
    emb_wd  = std::unique_ptr<EmbeddedVector>(new EmbeddedVector(network_wd->num_vertices,  this->constant_settings->dim));
    emb_wl  = std::unique_ptr<EmbeddedVector>(new EmbeddedVector(network_wl->num_vertices,  this->constant_settings->dim));
    emb_all_wd = std::unique_ptr<EmbeddedVector>(new EmbeddedVector(network_all->num_vertices, this->constant_settings->dim));
    emb_all_wl = std::unique_ptr<EmbeddedVector>(new EmbeddedVector(network_all->num_vertices, this->constant_settings->dim));

    sigmoid = std::unique_ptr<Sigmoid>(new Sigmoid());

//    std::cout << "total_samples:" << total_samples << std::endl;

//    std::cout << "DEBUG -2" << std::endl;

    // init alpha
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> r_uni(0, 1.0);
    alpha = r_uni(mt);

//    std::cout << "DEBUG -1" << std::endl;

    clock_t start;
    start = clock();
    printf("--------------------------------\n");

//    std::cout<<"DEBUG 0"<<std::endl;

    // マルチプロセス
    std::thread* th = new std::thread[constant_settings->threads];
    for (int i=0; i< constant_settings->threads; i++) {
        th[i] = std::thread(&D3NE::trainD3NEThread, i);
    }

//    std::cout<<"DEBUG 1"<<std::endl;

    for (int i=0; i < constant_settings->threads; i++) {
        th[i].join();
    }
    std::cout << "complete" << std::endl;

//    std::cout<<"DEBUG 2"<<std::endl;

    std::cout << std::endl;
    clock_t finish = clock();

    std::cout << "Total time: " << (double)(finish - start) / CLOCKS_PER_SEC << std::endl;

    std::cout << "trained alpha: " << alpha << std::endl;

//    std::cout<<"DEBUG 3"<<std::endl;

    this->output("_wd");
//    std::cout<<"DEBUG 4"<<std::endl;
    this->output("_wl");
    this->output("_ww");

//    std::cout<<"DEBUG END"<<std::endl;
}

void D3NE::trainD3NEThread(int id) {
    int64_t count = 0;
    int64_t last_count = 0;
    uint64_t seed = (uint64_t) (int64_t)id;

    while (1) {
        // judge for exit
        // total_samples=1Mがベース
        if (count > total_samples / constant_settings->threads + 2) {
//            std::cout<<"DEBUG 0-END"<<std::endl;
            break;
        }
//        std::cout<<"DEBUG 0-START"<<std::endl;

        // 表示の調整
        if (count - last_count>9999) {
            current_sample_count += count - last_count;
//            std::cout << "current_sample_count: " << current_sample_count << std::endl;
            last_count = count;
            printf("%cRho: %f  Progress: %.3lf%%", 13, cur_rho, (real)current_sample_count / (real)(total_samples + 1) * 100);
//            fflush(stdout);
            // rho:学習率,init_rho=0.025サンプル数に応じて下げている
            cur_rho = constant_settings->rho * (1 - current_sample_count / (real)(total_samples + 1));
            // rhoの下限設定
            if (cur_rho < constant_settings->rho * 0.0001) {
                cur_rho = constant_settings->rho * 0.0001;
            }
        }

//        std::cout<<"DEBUG 0-1"<<std::endl;
        // 途中の学習ベクトルを出力
//        if (current_sample_count % 100 == 0) {
//            output_tmp(current_sample_count);
//        }

        // joint training
        Network* n_ww = network_ww.get();
        Network* n_wd = network_wd.get();
        Network* n_wl = network_wl.get();
        Network* n_all = network_all.get();
//        std::cout<<"DEBUG 0-2"<<std::endl;

        EmbeddedVector* e_ww = emb_ww.get();
        EmbeddedVector* e_wd = emb_wd.get();
        EmbeddedVector* e_wl = emb_wl.get();
        EmbeddedVector* e_all_wd = emb_all_wd.get();
        EmbeddedVector* e_all_wl = emb_all_wl.get();
        Samplers* s_ww = sampler_ww.get();
        Samplers* s_wd = sampler_wd.get();
        Samplers* s_wl = sampler_wl.get();
//        std::cout<<"DEBUG 0-3"<<std::endl;

        learnVector_ww(n_wd, e_wd, n_wl, e_wl, s_ww, n_ww, n_all, e_all_wd, e_all_wl, seed);
//        std::cout<<"DEBUG 0-4"<<std::endl;
        learnVector(n_wd, e_wd, s_wd, n_all, e_all_wd, seed);
//        std::cout<<"DEBUG 0-5"<<std::endl;
        learnVector(n_wl, e_wl, s_wl, n_all, e_all_wl, seed);
//        std::cout<<"DEBUG 0-6"<<std::endl;

        count++;

    }
}


void* D3NE::learnVector_ww(
    Network* network_wd_a, EmbeddedVector* emb_wd_a,
    Network* network_wl_a, EmbeddedVector* emb_wl_a,
    Samplers* sampler_ww_a,
    Network* network_ww_a,
    Network* network_all_a,
    EmbeddedVector* emb_all_wd_a,
    EmbeddedVector* emb_all_wl_a,
    uint64_t seed
) {

    int64_t curedge;
    int64_t s_id;
    int64_t t_id;
    std::string s_name;
    std::string t_name;

    int64_t s_all_id;
    int64_t t_all_id;
    std::string s_all_name;
    std::string t_all_name;

    std::vector<real> vec_error_wd(constant_settings->dim);
    std::vector<real> vec_error_wl(constant_settings->dim);
//    std::vector<real> vec_error_all(constant_settings->dim);
    double error_alpha = 0;


    // edgeサンプリング（エッジのWeightに比例する形でAliasTableを利用してサンプリング）
    // for ww network
    curedge = sampler_ww_a->edgeSampling();
    s_id    = network_ww_a->edges[curedge]->source_id;
    t_id    = network_ww_a->edges[curedge]->target_id;

    //s_idとt_idをランダムに入れ替える仕様に変更してみる
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> r_uni(0, 1.0);
    double rand = r_uni(mt);
    if(rand<0.5) {
        int64_t tmp_id = s_id;
        s_id = t_id;
        t_id = tmp_id;
    }

    s_name  = network_ww_a->vertices[s_id]->name;
    t_name  = network_ww_a->vertices[t_id]->name;
//    std::cout << "s_name:: " << s_name << std::endl;
//    std::cout << "t_name:: " << t_name << std::endl;

    s_all_id    = network_all_a->searchHashTable(s_name);
    t_all_id    = network_all_a->searchHashTable(t_name);
    s_all_name  = network_all_a->vertices[s_all_id]->name;
    t_all_name  = network_all_a->vertices[t_all_id]->name;

    // copy latest emb_all to emb_each
//    for (int c = 0; c != constant_settings->dim; c++) {
//        emb_each->vertex[s_id][c] = emb_all_a->vertex[s_all_id][c];
//    }

    // vec_error, vec_error_allを０で初期化
    for (int c = 0; c != constant_settings->dim; c++) {
        vec_error_wd[c] = 0;
        vec_error_wl[c] = 0;
    }

    int64_t target;
    int64_t label;
    // NEGATIVE SAMPLING
    for (int d = 0; d != constant_settings->negative + 1; d++) {
        if (d == 0) {
            target = t_id;
            label = 1;
        } else {
            target = sampler_ww_a->negativeVertexSampling(seed);
            t_name = network_ww->vertices[target]->name;
            t_all_id = network_all_a->searchHashTable(t_name);
            label = 0;

        }

        // copy latest emb_all to emb_each
//        for (int c = 0; c != constant_settings->dim; c++) {
//            emb_each->vertex[target][c]     = emb_all_a->vertex[t_all_id][c];
//            emb_each->context[target][c]    = emb_all_a->context[t_all_id][c];
//        }

        // 更新
        if (constant_settings->order == 1) {
            update_ww(
                    emb_all_wd_a->vertex[s_all_id], emb_all_wd_a->vertex[t_all_id],
                    emb_all_wl_a->vertex[s_all_id], emb_all_wl_a->vertex[t_all_id],
                    vec_error_wd,
                    vec_error_wl,
                    &error_alpha, label);

        }

//        if (constant_settings->order == 1) {
//            update(emb_all_a->vertex[s_all_id], emb_all_a->vertex[t_all_id], vec_error_all, label);
//        }

//        std::cout << "before vec_error" << std::endl;
//        disp_vec(vec_error);

        if (constant_settings->order == 2) {
            update_ww(
                    emb_all_wd_a->vertex[s_all_id], emb_all_wd_a->context[t_all_id],
                    emb_all_wl_a->vertex[s_all_id], emb_all_wl_a->context[t_all_id],
                    vec_error_wd,
                    vec_error_wl,
                    &error_alpha, label);
        }


//        std::cout << "after vec_error" << std::endl;
//        disp_vec(vec_error);

//        if (constant_settings->order == 2) {
//            update(emb_all_a->vertex[s_all_id], emb_all_a->context[t_all_id], vec_error_all, label);
//        }
    }

    alpha -= error_alpha;
    if (alpha < 0) {
        alpha = 0;
    } else if (alpha > 1) {
        alpha = 1;
    }
//    std::cout << "alpha: " << alpha << std::endl;

    for (int c = 0; c != constant_settings->dim; c++) {
        // original LINE code is mistaken perhaps, change + to -.
//        std::cout << "vec_error_wd[" << c << "]:" << vec_error_wd[c] << std::endl;
        emb_all_wd_a->vertex[s_all_id][c] -= vec_error_wd[c];
        emb_all_wl_a->vertex[s_all_id][c] -= vec_error_wl[c];
//        std::cout << "vec_error[" << c << "]:" << vec_error[c] << std::endl;
//        std::cout << "vec_error_all[" << c << "]:" << vec_error_all[c] << std::endl;
    }
}

/* Update embeddings */
void D3NE::update_ww(
        std::vector<real>& vec_u_wd, std::vector<real>& vec_v_wd,
        std::vector<real>& vec_u_wl, std::vector<real>& vec_v_wl,
        std::vector<real>& vec_error_wd, std::vector<real>& vec_error_wl,
        double* error_alpha,
        int label
) {
    // update の式にw_ijのエッジの重みがついてないのは,サンプルを繰り返すうちに重みの分の比で更新されていくという発想だと思われる.
    real x = 0;
    real g;

    //vec_u_ww, vec_v_ww, vec_u_sub, vec_v_subの計算
    std::vector<real> vec_u_ww(constant_settings->dim);
    std::vector<real> vec_v_ww(constant_settings->dim);
    std::vector<real> vec_u_sub(constant_settings->dim);
    std::vector<real> vec_v_sub(constant_settings->dim);
    for (int c = 0; c != constant_settings->dim; c++) {
        vec_u_ww[c] = alpha * vec_u_wd[c] + (1-alpha) * vec_u_wl[c];
        vec_v_ww[c] = alpha * vec_v_wd[c] + (1-alpha) * vec_v_wl[c];
        vec_u_sub[c] = vec_u_wd[c] - vec_u_wl[c];
        vec_v_sub[c] = vec_v_wd[c] - vec_v_wl[c];

    }

    //内積の計算
    for (int c = 0; c != constant_settings->dim; c++) {
        x += vec_u_ww[c] * vec_v_ww[c];
    }


    g = - (label - sigmoid->fastFunc(x)) * cur_rho;
    for (int c = 0; c != constant_settings->dim; c++) {
        vec_error_wd[c] += g * alpha * vec_v_ww[c];
        vec_error_wl[c] += g * (1-alpha) * vec_v_ww[c];
        *error_alpha += g * ( vec_v_sub[c] * vec_u_ww[c]  + vec_v_ww[c] * vec_u_sub[c] );
//        std::cout << "g*vec_v[" << c << "]:" << g*vec_v[c] << std::endl;
    }
//    std::cout << "*error_alpha" << *error_alpha << std::endl;

//    std::cout << "tmp vec_error" << std::endl;
//    disp_vec(vec_error);

    for (int c = 0; c != constant_settings->dim; c++) {
        vec_v_wd[c] -= g * alpha * vec_u_ww[c];
        vec_v_wl[c] -= g * (1-alpha) * vec_u_ww[c];
//        std::cout << "g*vec_u[" << c << "]:" << g*vec_u[c] << std::endl;
    }
}

void D3NE::output(std::string mode) {
    std::string outputfile = this->constant_settings->outputfile + mode;
    FILE* fo = fopen(outputfile.c_str(), "wb");
    fprintf(fo, "%d %d\n", network_all->num_vertices, this->constant_settings->dim);
    for (int i = 0; i < network_all->num_vertices; i++) {
        std::string name = this->network_all->vertices[i]->name;
        fprintf(fo, "%s ", name.c_str());
        if (this->constant_settings->binary) {
            for (int j = 0; j < this->constant_settings->dim; j++) {
                if (mode=="_wd") {
                    fwrite(&emb_all_wd->vertex[i][j], sizeof(real), 1, fo);
                } else if (mode=="_wl") {
                    fwrite(&emb_all_wl->vertex[i][j], sizeof(real), 1, fo);
//                } else if (mode=="_ww") {
                } else {
//                    fwrite(&emb_all_wd->vertex[i][j] + &emb_all_wl->vertex[i][j], sizeof(real), 1, fo);
                }
            }

        } else {
            for (int j = 0; j < this->constant_settings->dim; j++) {
                if (mode=="_wd") {
                    fprintf(fo, "%lf ", emb_all_wd->vertex[i][j]);
                } else if (mode=="_wl") {
                    fprintf(fo, "%lf ", emb_all_wl->vertex[i][j]);
//                } else if (mode=="_ww") {
                } else {
                    fprintf(fo, "%lf ", emb_all_wl->vertex[i][j]+emb_all_wl->vertex[i][j]);
                }
            }
        }

        fprintf(fo, "\n");
    }
    fclose(fo);
}

