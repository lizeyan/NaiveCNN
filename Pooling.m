classdef Pooling < Layer
    properties
        kernel_size;
        pad;

        input;
        output;
        delta;
    end

    methods
        function layer = Pooling(name, kernel_size, pad)
            layer = layer@Layer(name);
            layer.kernel_size = kernel_size;
            layer.pad = pad;
        end

        function layer = forward(layer, input)
            layer.input = input;
            layer.output = nnpool(input, layer.kernel_size, layer.pad);
        end

        function layer = backprop(layer, delta)
            assert(isequal(size(delta), size(layer.output)), ...
                ['delta is inconsistent with ' layer.name ' output']);
            layer.delta = nnpool_bp(layer.input, delta, layer.kernel_size, layer.pad);
        end
    end
end


