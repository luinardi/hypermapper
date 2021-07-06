#ifndef RANDOM_H
#define RANDOM_H

#include <stdint.h>
#include <stdbool.h>
#include <math.h>

/***
 * This file contains an implementation of the SplitMix generator,
 * and the xoshiro256** generator.
 * They are adapted from:
 * https://prng.di.unimi.it/splitmix64.c and
 * https://prng.di.unimi.it/xoshiro256starstar.c
 * respectively.
 */

/**
 * Returns a double-precision floating-point number in the range [0, 1),
 * using bit manipulation on the given 64-bit integer.
 */
static inline double u64_to_f64(const uint64_t x) {
    union {
        uint64_t u64;
        double f64;
    } u = { .u64 = 0x3ff0000000000000ull | (x >> 12) };
    return 2.0 - u.f64;
}


typedef uint64_t Splitmix;

static inline void splitmix_init(Splitmix* state, const uint64_t seed) {
    *state = seed;
}

uint64_t splitmix_u64(Splitmix* state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
}

static inline double splitmix_f64(Splitmix* state) {
    return u64_to_f64(splitmix_u64(state));
}

typedef struct Xoshiro256 {
    uint64_t s[4];
} Xoshiro256;

void xoshiro256_init(Xoshiro256* state, const uint64_t seed) {
    Splitmix sm;
    splitmix_init(&sm, seed);

    state->s[0] = splitmix_u64(&sm);
    state->s[1] = splitmix_u64(&sm);
    state->s[2] = splitmix_u64(&sm);
    state->s[3] = splitmix_u64(&sm);
}

static inline uint64_t rotl64(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

uint64_t xoshiro256_u64(Xoshiro256* state) {
    const uint64_t result = rotl64(state->s[1] * 5, 7) * 9;

    const uint64_t t = state->s[1] << 17;

    state->s[2] ^= state->s[0];
    state->s[3] ^= state->s[1];
    state->s[1] ^= state->s[2];
    state->s[0] ^= state->s[3];

    state->s[2] ^= t;

    state->s[3] = rotl64(state->s[3], 45);

    return result;
}

static inline double xoshiro256_f64(Xoshiro256* state) {
    return u64_to_f64(xoshiro256_u64(state));
}

Xoshiro256 global_xoshiro256_state;

extern inline void random_init(const uint64_t seed) {
    xoshiro256_init(&global_xoshiro256_state, seed);
}

static inline uint64_t random_u64(void) {
    return xoshiro256_u64(&global_xoshiro256_state);
}

static inline double random_f64(void) {
    return xoshiro256_f64(&global_xoshiro256_state);
}

double random_normal(void) {
    double v, u, q;
    do {
        u = random_f64();
        v = 1.7156*(random_f64() - 0.5);
        const double x = u - 0.449871;
        const double y = fabs(v) + 0.386595;
        q = x*x + y*(0.19600*y-0.25472*x);
    } while (q > 0.27597 && (q > 0.27846 || v*v > -4.0*log(u)*u*u));
    return v / u;
}

double random_cauchy(void) {
    double v1, v2;
    do {
        v1 = 2.0 * random_f64() - 1.0;
        v2 = random_f64();
    } while (v1*v1+v2*v2 >= 1.0 || v2 < 0x1p-63);
    return v1 / v2;
}

#endif
