classdef SoftmaxLoss < handle
    properties
        name;
    end

    methods
        function layer = SoftmaxLoss(name)
            layer = layer@handle();
        end

        function loss = forward(layer, input, target)
            % Your codes here
            % hint:
            %     1. calculate probability from input using Softmax form.
            %        Notice: how to avoid overflow in exponential?
            %     2. loss = sum(target * -log(probability)), where target
            %        is one-hot encoding form label
            exp_input = exp (input);
            probability = exp_input ./ repmat (sum(exp_input), [size(input, 1), 1]);
            loss = -sum (target .* log(probability));
            
        end

        function delta = backprop(layer, input, target)
            % Your codes here
            exp_input = exp (input);
            probability = exp_input ./ repmat (sum(exp_input), [size(input, 1), 1]);
            delta = (probability - target) .* probability .* (1- probability);
        end
    end
end
