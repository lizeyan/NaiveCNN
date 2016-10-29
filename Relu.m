classdef Relu < Layer
    properties
        input;
        output;
        delta;
    end

    methods
        function layer = Relu(name)
            layer = layer@Layer(name);
        end

        function layer = forward(layer, input)
            % Your codes here
            layer.input = input;
            layer.output = max (zeros(size(input)), input);
        end

        function layer = backprop(layer, delta)
            % Your codes here
            layer.delta = delta .* (layer.input > 0);
        end
    end
end
