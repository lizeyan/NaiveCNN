classdef Layer < handle
    properties
        name;
        is_trainable;
    end

    methods
        function layer = Layer(name)
            layer = layer@handle();
            layer.name = name;
            layer.is_trainable = false;
        end

        function layer = forward(layer, input)
            % implemented by derived class
        end

        function layer = backprop(layer, delta)
            % implemented by derived class
        end

        function layer = update(layer, config)
            % implemented by trainable derived class
        end
    end
end
