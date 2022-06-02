#include <iostream>
#include <queue>
#include <random>
#include <vector>
using namespace std;

mt19937 rng(1);

struct Node {
    int feature;
    float split_val;
    int label;
    Node(int feature, float split_val, int label) : feature(feature), split_val(split_val), label(label){};
    static Node leaf(int label) {
        return Node(-1, -1, label);
    }
    static Node split(int feature, float split_val) {
        return Node(feature, split_val, -1);
    }
};

struct DecisionTree {
    int max_depth;
    vector<Node> nodes;
    vector<int> depth_l;
    int depth;
    DecisionTree(int max_depth) : max_depth(max_depth) {
        size_t l = 1 << (max_depth + 1);
        nodes.resize(l);
        depth_l.resize(l);
    }
};
int main() {
    DecisionTree a = DecisionTree(5);
}
