classdef Linear < Layer

    properties
        num_input;
        num_output;

        W;
        b;

        input;
        input_shape;
        output;

        grad_W;
        grad_b;
        diff_W; % last update for W
        diff_b; % last update for b

        delta;
    end

    methods
        function layer = Linear(name, num_input, num_output, init_std)
            layer = layer@Layer(name);
            layer.is_trainable = true;
            layer.num_input = num_input;
            layer.num_output = num_output;

            layer.W = single(random('norm', 0, init_std, num_output, num_input));
            layer.b = zeros(num_output, 1, 'single');
            layer.diff_W = zeros(size(layer.W), 'single');
            layer.diff_b = zeros(size(layer.b), 'single');
        end

        function layer = forward(layer, input)
            % Your codes here
            % hint: 
            %     1. reshape input if necessary 
            %     2. calc y = Wx + b

        end

        function layer = backprop(layer, delta)
            % Your codes here
            % hint: 
            %     1. calc grad_W, grad_b and delta for input
            %     2. reshape delta if necessary

        end

        function layer = update(layer, config)
            % SGD with momentum and weight decay
            mm = config.momentum;
            lr = config.learning_rate;
            wd = config.weight_decay;

            layer.diff_W = mm * layer.diff_W - lr * (layer.grad_W + wd * layer.W);
            layer.W = layer.W + layer.diff_W;

            layer.diff_b = mm * layer.diff_b - lr * (layer.grad_b + wd * layer.b);
            layer.b = layer.b + layer.diff_b;
        end
    end
end

