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
            probability = exp_input / sum (exp_input);
            loss = -sum (target .* log(probability));
            
        end

        function delta = backprop(layer, input, target)
            % Your codes here
            delta = input - target;
        end
    end
end
