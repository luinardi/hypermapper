#ifndef RANDOM_FOREST_H
#define RANDOM_FOREST_H

#include <stdbool.h>
#include <math.h>
#include <stdlib.h>

#include "random.h"
#include "datapoint.h"

#define REGTREE_MIN_SPLIT 5

typedef enum RegressionTreeNodeType {
    REGTREENODE_TYPE_UNINIT = 0,
    REGTREENODE_TYPE_SPLIT,
    REGTREENODE_TYPE_LEAF
} RegressionTreeNodeType;

typedef struct RegressionTreeNode {
    union {
        // A split node is constructed from a collection of {x, y} pairs.
        // Each x is a vector of attributes.
        // A split node splits the input based on a split index and split value,
        // such that:
        //     if (x[split index] <= split value) goto left subtree
        //     else goto right subtree
        struct {
            size_t split_index;
            double split_value;
            struct RegressionTreeNode* left;
            struct RegressionTreeNode* right;
        } split;

        // A leaf node is constructed from a collection of {x, y} pairs.
        // All information about x is encoded in the split-nodes that lead to a leaf,
        // so here we're only concerned about the y-information.
        struct {
            double y; ///< Sum of all y
            double y2; ///< Sum of all y^2
            double avg; ///< y/n
            size_t n; ///< Number of {x, y} pairs in this leaf
        } leaf;
    };
    RegressionTreeNodeType type;
} RegressionTreeNode;

/**
 * Calculates and returns the variance in y for a collection of points
 */
double variance(DataPoint** points, const size_t n) {
    double y = 0.0;
    double y2 = 0.0;
    for (size_t i=0; i<n; ++i) {
        y += points[i]->y;
        y2 += points[i]->y * points[i]->y;
    }
    return y2 - y * y / n;
}

/**
 * Finds the minimum and maximun y-vale among points, and returns them in min and max respectively.
 */
void minmax_y(DataPoint** points, const size_t n, double* min, double* max) {
    double lo = points[0]->y;
    double hi = points[0]->y;
    for (size_t i=1; i<n; ++i) {
        if (points[i]->y < lo) { lo = points[i]->y; }
        if (points[i]->y > hi) { hi = points[i]->y; }
    }
    *min = lo;
    *max = hi;
}

/**
 * Finds the minimum and maximum value in x[index] among points, and returns them in min/max.
 */
void minmax_x(DataPoint** points, const size_t n, const size_t index, double* min, double* max) {
    double lo = points[0]->x[index];
    double hi = points[0]->x[index];
    for (size_t i=1; i<n; ++i) {
        if (points[i]->x[index] < lo) { lo = points[i]->x[index]; }
        if (points[i]->x[index] > hi) { hi = points[i]->x[index]; }
    }
    *min = lo;
    *max = hi;
}

double get_split_variance(DataPoint** points, const size_t n,
                          const size_t split_index, const double split_value) {
    double y_left = 0.0;
    double y2_left = 0.0;
    size_t n_left = 0;

    double y_right = 0.0;
    double y2_right = 0.0;
    size_t n_right = 0;

    for (size_t i=0; i<n; ++i) {
        if (points[i]->x[split_index] <= split_value) {
            y_left += points[i]->y;
            y2_left += points[i]->y * points[i]->y;
            n_left += 1;
        } else {
            y_right += points[i]->y;
            y2_right += points[i]->y * points[i]->y;
            n_right += 1;
        }
    }

    const double var_left = y2_left - y_left * y_left / n_left;
    const double var_right = y2_right - y_right * y_right / n_right;

    return var_left + var_right;
}

bool split(DataPoint** points, const size_t dims, const size_t n, size_t* split_index, double* split_value) {

    //fprintf(stderr, "Trying to split %zu points\n", n);

    if (n < REGTREE_MIN_SPLIT) {
        //fprintf(stderr, "Too few\n");
        return false;
    }

    // If the range of values is small,
    // there is probably no point in splitting,
    // so we early-out here.
    double min_y, max_y;
    minmax_y(points, n, &min_y, &max_y);
    if (fabs(min_y - max_y) < 0x1p-63) {
        //fprintf(stderr, "Small range: [%f, %f]\n", min_y, max_y);
        return false;
    }

    const double var_before = variance(points, n);
    //fprintf(stderr, "Variance before: %f\n", var_before);

    double best_split_var = INFINITY;
    double best_split_val = 0.0;
    size_t best_split_index = 0;
    for (size_t i=0; i<dims; ++i) {

        // Get minimum and maximum x
        double min_x, max_x;
        minmax_x(points, n, i, &min_x, &max_x);

        // Compute a random splitting value, and get variance for that split
        const double split_val = min_x + (max_x - min_x) * random_f64();
        const double split_var = get_split_variance(points, n, i, split_val);

        if (split_var < best_split_var) {
            best_split_var = split_var;
            best_split_val = split_val;
            best_split_index = i;
        }
    }
    //fprintf(stderr, "Found best split in dimension %zu: val = %f, var = %f\n",
            //best_split_index, best_split_val, best_split_var);

    // No significant reduction in variance
    if (var_before - best_split_var < 0x1p-63) {
        //fprintf(stderr, "No significant reduction: before = %f, after = %f\n", var_before, best_split_var);
        return false;
    }

    *split_index = best_split_index;
    *split_value = best_split_val;
    return true;
}

void make_leaf(RegressionTreeNode* node, DataPoint** points, const size_t n) {
    double y = 0.0;
    double y2 = 0.0;
    for (size_t i=0; i<n; ++i) {
        y += points[i]->y;
        y2 += points[i]->y * points[i]->y;
    }

    node->type = REGTREENODE_TYPE_LEAF;
    node->leaf.y = y;
    node->leaf.y2 = y2;
    node->leaf.avg = y / n;
    node->leaf.n = n;
}

void regressiontreenode_fit(RegressionTreeNode* node,
                            DataPoint** points, const size_t dims, const size_t n) {

    //fprintf(stderr, "Fitting %zu points\n", n);
    size_t split_index;
    double split_value;

    if (!split(points, dims, n, &split_index, &split_value)) {
        // If we failed to find a working split, make this node a leaf and return
        //fprintf(stderr, "No split\n");
        make_leaf(node, points, n);
        return;
    }

    // Here we begin to partition points into
    // left: x[split_index] <= split_value
    // right: x[split_index > split_value

    // Find the first 'right' point
    size_t hi = 0;
    while (hi < n && points[hi]->x[split_index] <= split_value) { hi += 1; }

    // If no 'right' point was found, then all are left,
    // and there is no split,
    // so we make a leaf...
    if (hi >= n) {
        //fprintf(stderr, "Bad split\n");
        make_leaf(node, points, n);
        return;
    }

    // Perform actual partitioning
    for (size_t i=hi+1; i<n; ++i) {
        if (points[i]->x[split_index] <= split_value) {
            DataPoint* temp = points[i];
            points[i] = points[hi];
            points[hi] = temp;
            hi += 1;
        }
    }

    // Partitioning done

    const size_t n_left = hi;
    const size_t n_right = n - hi;

    //fprintf(stderr, "Left x[%zu] <= %f: %zu\n", split_index, split_value, n_left);
    //fprintf(stderr, "Right x[%zu] > %f: %zu\n", split_index, split_value, n_right);

    // Fit left subtree
    //fprintf(stderr, "Fitting left\n");
    RegressionTreeNode* left = calloc(1, sizeof(RegressionTreeNode));
    regressiontreenode_fit(left, points, dims, n_left);

    // Fit right subtree
    //fprintf(stderr, "Fitting right\n");
    RegressionTreeNode* right = calloc(1, sizeof(RegressionTreeNode));
    regressiontreenode_fit(right, points+hi, dims, n_right);

    // Construct this node
    node->type = REGTREENODE_TYPE_SPLIT;
    node->split.split_index = split_index;
    node->split.split_value = split_value;
    node->split.left = left;
    node->split.right = right;
}

void regressiontreenode_free(RegressionTreeNode* node) {
    if (node->type == REGTREENODE_TYPE_SPLIT) {
        regressiontreenode_free(node->split.left);
        free(node->split.left);
        regressiontreenode_free(node->split.right);
        free(node->split.right);
    }
}

typedef struct RegressionTree {
    RegressionTreeNode* root;
} RegressionTree;

static inline void regressiontree_fit(RegressionTree* tree,
                                      DataPoint** points, const size_t dims, const size_t n) {
    tree->root = calloc(1, sizeof(RegressionTreeNode));
    regressiontreenode_fit(tree->root, points, dims, n);
}

RegressionTreeNode* regressiontree_propagate(RegressionTree* tree, const double* x) {
    RegressionTreeNode* n = tree->root;
    while (n->type != REGTREENODE_TYPE_LEAF) {
        if (x[n->split.split_index] <= n->split.split_value) {
            n = n->split.left;
        } else {
            n = n->split.right;
        }
    }
    return n;
}

static inline double regressiontree_predict(RegressionTree* tree, const double* x) {
    RegressionTreeNode* n = regressiontree_propagate(tree, x);
    return n->leaf.avg;
}

double regressiontree_fulL_predict(RegressionTree* tree, const double* x, double* variance) {
    RegressionTreeNode* n = regressiontree_propagate(tree, x);
    const double y = n->leaf.y / n->leaf.n;
    const double y2 = n->leaf.y2 / n->leaf.n;
    *variance = y2 - y * y;
    return y;
}

void regressiontree_free(RegressionTree* tree) {
    regressiontreenode_free(tree->root);
    free(tree->root);
}

typedef struct RandomForest {
    RegressionTree* trees;
    size_t n_trees;
} RandomForest;

void randomforest_fit(RandomForest* forest, const size_t n_trees,
                      DataPoint** points, const size_t dims, const size_t n) {
    RegressionTree* trees = calloc(n_trees, sizeof(RegressionTree));
    for (size_t i=0; i<n_trees; ++i) {
        regressiontree_fit(&trees[i], points, dims, n);
    }
    forest->trees = trees;
    forest->n_trees = n_trees;
}

double randomforest_predict(RandomForest* forest, const double* x) {
    double tot = 0.0;
    for (size_t i=0; i<forest->n_trees; ++i) {
        tot += regressiontree_predict(&forest->trees[i], x);
    }
    return tot / forest->n_trees;
}

double randomforest_full_predict(RandomForest* forest, const double* x, double* variance) {
    double y = 0.0;
    double y2 = 0.0;
    size_t n = 0;
    for (size_t i=0; i<forest->n_trees; ++i) {
        RegressionTreeNode* node = regressiontree_propagate(&forest->trees[i], x);
        y += node->leaf.y;
        y2 += node->leaf.y2;
        n += node->leaf.n;
    }
    y /= n;
    y2 /= n;
    *variance = y2 - y * y;
    return y;
}

void randomforest_free(RandomForest* forest) {
    for (size_t i=0; i<forest->n_trees; ++i) {
        regressiontree_free(&forest->trees[i]);
    }
    free(forest->trees);
}

#endif
