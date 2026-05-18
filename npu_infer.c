#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "rknn_api.h"

int main() {
    // Load model
    FILE* f = fopen("/root/yolov8n_final.rknn", "rb");
    fseek(f, 0, SEEK_END);
    int size = ftell(f);
    fseek(f, 0, SEEK_SET);
    void* model = malloc(size);
    fread(model, 1, size, f);
    fclose(f);

    rknn_context ctx;
    int ret = rknn_init(&ctx, model, size, 0, NULL);
    printf("init: %d\n", ret);

    // Set input
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    int input_size = 640*640*3;
    void* input_data = calloc(1, input_size);
    inputs[0].index = 0;
    inputs[0].buf = input_data;
    inputs[0].size = input_size;
    inputs[0].pass_through = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;

    ret = rknn_inputs_set(ctx, 1, inputs);
    printf("inputs_set: %d\n", ret);

    ret = rknn_run(ctx, NULL);
    printf("run: %d\n", ret);

    // Get output
    rknn_output outputs[1];
    memset(outputs, 0, sizeof(outputs));
    outputs[0].want_float = 1;
    ret = rknn_outputs_get(ctx, 1, outputs, NULL);
    printf("outputs_get: %d, size: %d\n", ret, outputs[0].size);

    free(input_data);
    free(model);
    rknn_destroy(ctx);
    return 0;
}
