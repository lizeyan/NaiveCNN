classdef Network < handle
    properties
        layer_list
        num_layer
    end

    methods
        function net = Network()
            net = net@handle();
            net.layer_list = {};
            net.num_layer = 0;
        end

        function net = add(net, layer)
            net.num_layer = net.num_layer + 1;
            net.layer_list{net.num_layer} = layer;
        end

        function net = Forward(net, input)
            net.layer_list{1}.forward(input);
            for k = 2:net.num_layer
                net.layer_list{k}.forward(net.layer_list{k - 1}.output);
            end
        end

        function net = Backpropagation(net, delta)
            net.layer_list{net.num_layer}.backprop(delta);
            for k = net.num_layer-1:-1:1
                net.layer_list{k}.backprop(net.layer_list{k + 1}.delta);
            end
        end

        function net = Update(net, update_config)
            for k = 1:net.num_layer
                if net.layer_list{k}.is_trainable
                    net.layer_list{k}.update(update_config);
                end
            end
        end

        function output = Output(net)
            output = net.layer_list{net.num_layer}.output;
        end

        function preds = Predict(net, input)
            % Your codes here
            % hint:
            %    1. forward input through net
            %    2. get the net's output
            net.Forward (input);
            preds = net.Output ();
        end
        
        function output = filt (net, input)
            net.layer_list{1}.forward(input);
            net.layer_list{2}.forward(net.layer_list{1}.output);
            output = net.layer_list{2}.output;
        end
    end
end
