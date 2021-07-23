#ifndef BAYESIAN_H
#define BAYESIAN_H

#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "random.h"
#include "random_forest.h"
#include "datapoint.h"

#define RANDOM_SAMPLE_PROBABILITY 0.05
#define NUMBER_OF_TREES 10
#define EVO_POPULATION 50
#define MUTATE_PROB 0.1

typedef double (*OptFunc)(const double*);

void get_random(const size_t n, const double* lower, const double* upper, double* x) {
    for (size_t i=0; i<n; ++i) {
        x[i] = lower[i] + (upper[i] - lower[i]) * random_f64();
    }
}

static inline double pdf(const double x) {
    return exp(-x*x/2.0) / sqrt(2.0 * M_PI);
}

static inline double cdf(const double x) {
    return 0.5 * (1.0 + erf(x / M_SQRT2));
}

static inline double ei(const double y, const double sdev, const double fmin) {
    const double delta = fmin - y;
    const double z = delta / sdev;
    return delta + sdev * pdf(z) + delta * cdf(z);
}

void optimize_acquisition_function(RandomForest* forest, const double* lower, const double* upper,
                     double* out, const size_t dims, const double fmin, DataPoint** points) {

    // This will be our workhorse x
    DataPoint* p = points[EVO_POPULATION];

    double best = -INFINITY;
    
    // Initialize the workspace with random points.
    for (size_t i=0; i<EVO_POPULATION; ++i) {
        get_random(dims, lower, upper, points[i]->x);
        double variance;
        double y = randomforest_full_predict(forest, points[i]->x, &variance);
        points[i]->y = ei(y, sqrt(variance), fmin);

        if (points[i]->y > best) {
            best = points[i]->y;
            for (size_t j=0; j<dims; ++j) { out[j] = points[i]->x[j]; }
        }
    }

    size_t oldest = 0;
    for (size_t i=0; i<10000; ++i) {

        DataPoint* p1 = points[0];
        DataPoint* p2 = points[1];
        if (p2->y > p1->y) {
            DataPoint* temp = p1;
            p1 = p2;
            p2 = temp;
        }

        for (size_t j=2; j<EVO_POPULATION; ++j) {
            if (points[j]->y > p1->y) {
                p2 = p1;
                p1 = points[j];
            } else if (points[j]->y > p2->y) {
                p2 = points[j];
            }
        }

        const double sum = p1->y + p2->y;
        const double p1_prob = p1->y / sum;

        for (size_t j=0; j<dims; ++j) {
            p->x[j] = (random_f64() < p1_prob) ? p1->x[j] : p2->x[j];
        }

        for (size_t j=0; j<dims; ++j) {
            if (random_f64() < MUTATE_PROB) {
                p->x[j] = lower[j] + (upper[j] - lower[j]) * random_f64();
            }
        }

        double variance;
        double y = randomforest_full_predict(forest, p->x, &variance);
        p->y = ei(y, sqrt(variance), fmin);

        if (p->y > best) {
            best = p->y;
            for (size_t j=0; j<dims; ++j) { out[j] = p->x[j]; }
        }

        DataPoint* temp = points[oldest];
        points[oldest] = p;
        p = temp;

        oldest = (oldest + 1) % EVO_POPULATION;

    }

    // Make sure to update the backing pointer array, since p may have been swapped
    points[EVO_POPULATION] = p;
}

void shuffle_points(DataPoint** points, const size_t n) {
    for (size_t i=0; i<n-1; ++i) {
        const size_t j = i + random_u64() % (n-i);
        DataPoint* temp = points[i];
        points[i] = points[j];
        points[j] = temp;
    }
}

double bayesian_optimization(OptFunc f, const double* lower, const double* upper, double* x, const size_t dims,
                             const size_t doe, const size_t iter) {

    const size_t dp_size = sizeof(DataPoint) + dims * sizeof(double);

    // Alloc backing data for all points (1 per iteration)
    DataPoint* data = calloc(iter, dp_size);
    
    // Alloc pointers that can point into backing data points
    DataPoint** points = calloc(iter, sizeof(DataPoint*));

    // Assign each pointer to an allocated data point
    points[0] = data;
    for (size_t i=1; i<iter; ++i) { points[i] = (void*)points[i-1] + dp_size; }

    // Alloc work space for local search algorithm
    DataPoint* workspace = calloc(EVO_POPULATION + 1, dp_size);

    // Pointers into workspace
    DataPoint** workspace_points = calloc(EVO_POPULATION + 1, sizeof(DataPoint*));

    workspace_points[0] = workspace;
    for (size_t i=1; i<EVO_POPULATION+1; ++i) {
        workspace_points[i] = (void*)workspace_points[i-1] + dp_size;
    }

    // This will correspond to the point in the parameter x
    double best_val = INFINITY;

    // Initialize DoE phase as a latin hypercube
    for (size_t i=0; i<dims; ++i) {
        for (size_t j=0; j<doe; ++j) {
            points[j]->x[i] = lower[i] + (upper[i] - lower[i]) * (j + random_f64()) / doe;
        }
        shuffle_points(points, doe);
    }

    // Evaluate DoE points
    for (size_t i=0; i<doe; ++i) {
        const double y = f(points[i]->x);
        points[i]->y = y;

        if (y < best_val) {
            best_val = y;
            for (size_t j=0; j<dims; ++j) { x[j] = points[i]->x[j]; }
        }
    }

    double mean_iter = 0.0;

    for (size_t i=0; i<iter; ++i) {

        if (random_f64() < RANDOM_SAMPLE_PROBABILITY) {

            get_random(dims, lower, upper, points[i]->x);

        } else {
            // Else get a guess from random forest model

            // Fit forest
            RandomForest forest;
            randomforest_fit(&forest, NUMBER_OF_TREES, points, dims, i);

            optimize_acquisition_function(&forest, lower, upper, points[i]->x, dims, best_val, workspace_points);

            randomforest_free(&forest);
        }

        const double y = f(points[i]->x);
        points[i]->y = y;

        if (y < best_val) {
            best_val = y;
            for (size_t j=0; j<dims; ++j) { x[j] = points[i]->x[j]; }
        }
    }

    free(workspace_points);
    free(workspace);
    free(points);
    free(data);

    return best_val;
}

#endif
