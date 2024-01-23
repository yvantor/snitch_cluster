// Copyright 2024 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Author: Luca Colagrande <colluca@iis.ee.ethz.ch>

#include <stdint.h>

#include "math.h"
#include "snrt.h"

double euclidean_distance_squared(uint32_t n_features, double* point1, double* point2) {
    double sum = 0;
    for (uint32_t i = 0; i < n_features; i++) {
        double diff = point1[i] - point2[i];
        sum += diff * diff;
    }
    return sum;
}

// Allocate space in L1, returns pointer to the same location for each core in cluster.
// Different clusters get different pointers, to the same offset within its TCDM.
inline void* snrt_l1_alloc_cluster_private(void* base, size_t size, void** new_base) {
    *new_base = base + size;
    return base;
}

// Allocate space in L1, each compute core gets unique space.
inline void* snrt_l1_alloc_compute_core_private(void* base, size_t size, void** new_base) {
    *new_base = base + size * snrt_cluster_compute_core_num();
    return base + size * snrt_cluster_core_idx();
}

// Allocate space in L1, all clusters get pointer to cluster 0's allocation.
inline void* snrt_l1_alloc_common(void* base, size_t size, void** new_base) {
    *new_base = base + size;
    return (void*)((uintptr_t)base - snrt_cluster_idx() * SNRT_CLUSTER_OFFSET);
}

// Takes the pointer to a variable in one cluster's TCDM (src), and returns the pointer
// to the variable at the same offset in another cluster's TCDM (dst)
inline void* snrt_remote_cluster_ptr(void* src, uint32_t src_cluster_idx, uint32_t dst_cluster_idx) {
    return (void *)((uintptr_t)src + (dst_cluster_idx - src_cluster_idx) * SNRT_CLUSTER_OFFSET);
}

void kmeans(uint32_t n_samples, uint32_t n_features, uint32_t n_clusters, uint32_t n_iter, double* samples, double* centroids) {
    // Distribute work across clusters
    uint32_t n_samples_per_cluster = n_samples / snrt_cluster_num();

    // Allocate space for operands in TCDM
    void *l1_base, *prev_l1_base;
    l1_base = snrt_l1_next();
    double *local_samples = snrt_l1_alloc_cluster_private(l1_base, n_samples_per_cluster * n_features * sizeof(double), &l1_base);
    double *local_centroids = snrt_l1_alloc_cluster_private(l1_base, n_clusters * n_features * sizeof(double), &l1_base);
    // Allocate space for intermediate variables in TCDM
    uint32_t *membership = snrt_l1_alloc_cluster_private(l1_base, n_samples_per_cluster * sizeof(uint32_t), &l1_base);
    // Alias first core's partial membership counters with final membership counters
    prev_l1_base = l1_base;
    uint32_t *final_membership_cnt = snrt_l1_alloc_common(l1_base, n_clusters * sizeof(uint32_t), &l1_base);
    uint32_t *partial_membership_cnt = snrt_l1_alloc_compute_core_private(prev_l1_base, n_clusters * sizeof(uint32_t), &l1_base);
    // Alias first core's partial centroids with final centroids
    prev_l1_base = l1_base;
    double *final_centroids = snrt_l1_alloc_common(l1_base, n_clusters * n_features * sizeof(double), &l1_base);
    double *partial_centroids = snrt_l1_alloc_compute_core_private(prev_l1_base, n_clusters * n_features * sizeof(double), &l1_base);

    // Transfer samples and initial centroids with DMA
    size_t size;
    size_t offset;
    if (snrt_is_dm_core()) {
        size = n_samples_per_cluster * n_features * sizeof(double);
        offset = snrt_cluster_idx() * size;
        snrt_dma_start_1d((void *)local_samples, (void *)samples + offset, size);
        size = n_clusters * n_features * sizeof(double);
        snrt_dma_start_1d((void *)local_centroids, (void *)centroids, size);
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();

    // Iterations of Lloyd's K-means algorithm
    for (uint32_t iter_idx = 0; iter_idx < n_iter; iter_idx++) {

        // Distribute work across compute cores in a cluster
        uint32_t n_samples_per_core;
        uint32_t start_sample_idx;
        uint32_t end_sample_idx;
        if (snrt_is_compute_core()) {
            n_samples_per_core = n_samples_per_cluster / snrt_cluster_compute_core_num();
            start_sample_idx = snrt_cluster_core_idx() * n_samples_per_core;
            end_sample_idx = start_sample_idx + n_samples_per_core;

            // Assignment step
            for (uint32_t centroid_idx = 0; centroid_idx < n_clusters; centroid_idx++) {
                partial_membership_cnt[centroid_idx] = 0;
            }
            snrt_fpu_fence();
            for (uint32_t sample_idx = start_sample_idx; sample_idx < end_sample_idx; sample_idx++) {
                
                double min_dist = INFINITY;
                membership[sample_idx] = 0;
                
                for (uint32_t centroid_idx = 0; centroid_idx < n_clusters; centroid_idx++) {
                    double dist = euclidean_distance_squared(
                        n_features, &local_samples[sample_idx * n_features], &local_centroids[centroid_idx * n_features]);
                    if (dist < min_dist) {
                        min_dist = dist;
                        membership[sample_idx] = centroid_idx;
                    }
                }
                partial_membership_cnt[membership[sample_idx]]++;
            }
        }

        snrt_global_barrier();

        if (snrt_is_compute_core()) {

            // Update step
            for (uint32_t centroid_idx = 0; centroid_idx < n_clusters; centroid_idx++) {
                for (uint32_t feature_idx = 0; feature_idx < n_features; feature_idx++) {
                    partial_centroids[centroid_idx * n_features + feature_idx] = 0;
                }
            }
            snrt_fpu_fence();
            for (uint32_t sample_idx = start_sample_idx; sample_idx < end_sample_idx; sample_idx++) {
                for (uint32_t feature_idx = 0; feature_idx < n_features; feature_idx++) {
                    partial_centroids[membership[sample_idx] * n_features + feature_idx] += local_samples[sample_idx * n_features + feature_idx];
                }
            }
            if (snrt_cluster_core_idx() == 0) {
                // Intra-cluster reduction
                for (uint32_t core_idx = 1; core_idx < snrt_cluster_compute_core_num(); core_idx++) {
                    // Pointers to variables of the other core
                    uint32_t* remote_partial_membership_cnt = partial_membership_cnt + core_idx * n_clusters;
                    double* remote_partial_centroids = partial_centroids + core_idx * n_clusters * n_features;
                    for (uint32_t centroid_idx = 0; centroid_idx < n_clusters; centroid_idx++) {
                        // Accumulate membership counters
                        partial_membership_cnt[centroid_idx] += remote_partial_membership_cnt[centroid_idx];
                        // Accumulate centroid features
                        for (uint32_t feature_idx = 0; feature_idx < n_features; feature_idx++) {
                            partial_centroids[centroid_idx * n_features + feature_idx] += remote_partial_centroids[centroid_idx * n_features + feature_idx];
                        }
                    }
                }
                snrt_inter_cluster_barrier();
                if (snrt_cluster_idx() == 0) {
                    // Inter-cluster reduction
                    for (uint32_t cluster_idx = 1; cluster_idx < snrt_cluster_num(); cluster_idx++) {
                        // Pointers to variables of remote clusters
                        uint32_t* remote_partial_membership_cnt = (uint32_t *)snrt_remote_cluster_ptr(partial_membership_cnt, 0, cluster_idx);
                        double* remote_partial_centroids = (double *)snrt_remote_cluster_ptr(partial_centroids, 0, cluster_idx);
                        for (uint32_t centroid_idx = 0; centroid_idx < n_clusters; centroid_idx++) {
                            // Accumulate membership counters
                            final_membership_cnt[centroid_idx] += remote_partial_membership_cnt[centroid_idx];
                            // Accumulate centroid features
                            for (uint32_t feature_idx = 0; feature_idx < n_features; feature_idx++) {
                                final_centroids[centroid_idx * n_features + feature_idx] += remote_partial_centroids[centroid_idx * n_features + feature_idx];
                            }
                        }
                    }
                    // Normalize
                    for (uint32_t centroid_idx = 0; centroid_idx < n_clusters; centroid_idx++) {
                        for (uint32_t feature_idx = 0; feature_idx < n_features; feature_idx++) {
                            final_centroids[centroid_idx * n_features + feature_idx] /= final_membership_cnt[centroid_idx];
                        }
                    }
                }
            }
        }
        
        snrt_global_barrier();
        local_centroids = final_centroids;
    }

    snrt_cluster_hw_barrier();

    // Transfer final centroids with DMA
    if (snrt_is_dm_core() && snrt_cluster_idx() == 0) {
        snrt_dma_start_1d((void *)centroids, (void *)final_centroids, size);
        snrt_dma_wait_all();
    }
}
