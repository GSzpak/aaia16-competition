#include <assert.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

int min(int a, int b) {
    return a < b ? a : b;
}

int max(int a, int b) {
    return a > b ? a : b;
}

double md_dtw(int len, double *md_time_series1, double *md_time_series2) {
    int i, j, k;
    int seq_len = 24;
    int window_size = 4;
    int num_of_dimensions = len / seq_len;
    double distances[seq_len][seq_len];
    for (i = 0; i < seq_len; ++i) {
        for (j = 0; j < seq_len; ++j) {
            distances[i][j] = 0;
            for (k = 0; k < num_of_dimensions; ++k) {
                distances[i][j] += fabs(md_time_series1[i + k * seq_len] - md_time_series2[j + k * seq_len]);
            }

        }
    }
    double accumulated_cost[seq_len][seq_len];
    accumulated_cost[0][0] = distances[0][0];
    for (i = 1; i < seq_len; ++i) {
        accumulated_cost[i][0] = accumulated_cost[i - 1][0] + distances[i][0];
        accumulated_cost[0][i] = accumulated_cost[0][i - 1] + distances[0][i];
    }
    for (i = 1; i < seq_len; ++i) {
        for (j = max(0, i - window_size); j < min(seq_len, i + window_size); ++j) {
            accumulated_cost[i][j] = fmin(accumulated_cost[i - 1][j - 1],
                fmin(accumulated_cost[i][j - 1], accumulated_cost[i - 1][j])) + distances[i][j];
        }
    }
    return accumulated_cost[seq_len - 1][seq_len - 1];
}