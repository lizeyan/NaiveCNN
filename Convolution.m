classdef Convolution < Layer
    properties
        kernel_size;
        num_input;
        num_output;
        pad;

        input;
        output;

        W;
        b;

        grad_W;
        grad_b;
        diff_W; % last update for W
        diff_b; % last update for b
        delta;
    end

    methods
        function layer = Convolution(name, kernel_size, num_input, ...
                                     num_output, pad, init_std)
            layer = layer@Layer(name);
            layer.is_trainable = true;
            layer.kernel_size = kernel_size;
            layer.num_input = num_input;
            layer.num_output = num_output;
            layer.pad = pad;

            layer.W = single(random('norm', 0, init_std, kernel_size, kernel_size, num_input, num_output));
            layer.b = zeros(num_output, 1, 'single');
            layer.diff_W = zeros(size(layer.W), 'single');
            layer.diff_b = zeros(size(layer.b), 'single');
        end

        function layer = forward(layer, input)
            assert(size(input, 3) == layer.num_input, ...
                ['input channel is inconsistent with ' layer.name ' filter']);

            layer.input = input;
            layer.output = nnconv(layer.input, layer.kernel_size, layer.num_output, ...
                layer.W, layer.b, layer.pad);
        end

        function layer = backprop(layer, delta)
            assert(isequal(size(delta), size(layer.output)), ...
                ['delta is inconsistent with ' layer.name ' output']);

            [layer.delta, layer.grad_W, layer.grad_b] = nnconv_bp(layer.input, delta, ...
                layer.W, layer.b, layer.pad);
        end

        function layer = update(layer, config)
            mm = config.momentum;
            lr = config.learning_rate;
            wd = config.weight_decay;

            layer.diff_W = mm * layer.diff_W - lr * (layer.grad_W + wd * layer.W);
            layer.W = layer.W + layer.diff_W;

            layer.diff_b = mm * layer.diff_b - lr * (layer.grad_b + wd * layer.b);
            layer.b = layer.b + layer.diff_b;
        end

        function params = get_params(layer)
            params.weight = layer.W;
            params.bias = layer.b;
        end
    end
end


