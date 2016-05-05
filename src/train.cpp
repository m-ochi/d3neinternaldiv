#include "pteself/src/cmdline/cmdline.h"
#include "d3ne.h"
#include <iostream>
#include <string>

int main(int argc, char **argv) {
    cmdline::parser option;
    option.add<std::string>("train_ww",0,"set word--word network csv file",true);
    option.add<std::string>("train_wd",0,"set word--document network csv file",true);
    option.add<std::string>("train_wl",0,"set word--label network csv file",true);
    option.add<std::string>("output",  0,"set output file",false,"out_embeddings.txt");
    option.add<int>("binary",  0,"save the learnt embeddings in binary moded; default is 0(off)",false,0);
    option.add<int>("size",    0,"Set dimension of vertex embeddings; default is ",false,100);
    option.add<int>("order",   0,"The type of the model; 1 for first order, 2 for second order; default is ",false,2);
    option.add<int>("negative",0,"Number of negative examples; default is ",false,5);
    option.add<int>("samples", 0,"Set the number of training samples as <int>Million; default is ",false,1);
    option.add<int>("threads", 0,"Use <int> threads (default is )",false,1);
    option.add<float>("rho", 0,"Set the starting learning rate; default is ",false,0.025);

    option.parse_check(argc,argv);

    std::string train_ww = option.get<std::string>("train_ww");
    std::string train_wd = option.get<std::string>("train_wd");
    std::string train_wl = option.get<std::string>("train_wl");
    std::string output   = option.get<std::string>("output");
    int binary  = option.get<int>("binary");
    int size    = option.get<int>("size");
    int order   = option.get<int>("order");
    int negative= option.get<int>("negative");
    int samples = option.get<int>("samples");
    int threads = option.get<int>("threads");
    float rho     = option.get<float>("rho");

//    std::cout << option.get<std::string>("train_ww") << std::endl;
//    std::cout << option.get<std::string>("train_wd") << std::endl;
//    std::cout << option.get<std::string>("train_wl") << std::endl;
//    std::cout << option.get<std::string>("output") << std::endl;
//    std::cout << option.get<int>("binary") << std::endl;
//    std::cout << option.get<int>("size") << std::endl;
//    std::cout << option.get<int>("order") << std::endl;
//    std::cout << option.get<int>("negative") << std::endl;
//    std::cout << option.get<int>("samples") << std::endl;
//    std::cout << option.get<int>("threads") << std::endl;
//    std::cout << option.get<float>("rho") << std::endl;

    std::unique_ptr<PTE> d3ne(new D3NE(train_ww,train_wd,train_wl,output,binary,size,order,negative,samples,threads,rho));

    d3ne->run();
}

