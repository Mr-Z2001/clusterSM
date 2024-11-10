#! /bin/bash
# ./build/ClusterSM -q ./dataset/example_q_01.in -d ./dataset/example_g_01.in --gpu 6 > output_tm.txt
# compute-sanitizer ./build/ClusterSM -q ./dataset/example_q_01.in -d ./dataset/example_g_01.in --gpu 0 > output_cs.txt
# cuda-gdb ./build/ClusterSM -q ./dataset/example_q_01.in -d ./dataset/example_g_01.in --gpu 1 > output_gdb.txt

# github
# echo "github" > output_cs.txt
# for line in `ls ~/codes/datasets/github/query_graph/`
# do
#   compute-sanitizer ./build/ClusterSM -q ~/codes/datasets/github/query_graph/$line -d ~/codes/datasets/github/data.graph --gpu 0 >> output_cs.txt
# done

compute-sanitizer ./build/ClusterSM -q ~/codes/datasets/github/query_graph/Q_1 -d ~/codes/datasets/github/data.graph --gpu 0 > output_cs.txt

# gowalla 
# ./build/SSM -q ./dataset/gowalla/label_16/query_graph/12/Q_0 -d ./dataset/gowalla/label_16/data.graph -t 1 --gpu 2 >> output_tm.txt
# ./build/SSM -q ./dataset/gowalla/query/query01.in -d ./dataset/gowalla/label_16/data.graph -t 1 --gpu 0 >> output_tm.txt
# ./build/SSM -q ./dataset/example_q_01.in -d ./dataset/gowalla/label_16/data.graph -t 1 --gpu 3 >> output_tm.txt

# github
# ./build/SSM -q ./dataset/example_q_01.in -d ./dataset/github/label_16/data.graph -t 1 --gpu 3 >> output_tm.txt
# ./build/SSM -q ./dataset/example_g_01.in -d ./dataset/github/label_16/data.graph -t 1 --gpu 3 >> output_tm.txt
# ./build/SSM -q ./dataset/github/label_16/query_graph/12/Q_0 -d ./dataset/github/label_16/data.graph -t 1 --gpu 7 >> output_tm.txt

#dblp
# ./build/SSM -q ./dataset/example_q_01.in -d ./dataset/dblp/label_16/data.graph -t 1 --gpu 7 >> output_tm.txt