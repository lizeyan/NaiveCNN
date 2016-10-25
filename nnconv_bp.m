function [down_delta, grad_W, grad_b] = nnconv_bp(input, delta, W, b, pad)
    % Your codes here
    % hint:
    %     follow the formula in slides(page 17). 
    %     Generally speaking, the backpropagation process is also
    %     a convolution process, with input and rotated delta.
    %     Gradient checking can assure you of a correct implementation
    [width, height, channel, number] = size(input);
    down_delta = zeros (width, height, channel, number);
    grad_W = zeros (size (W));
    for n = 1:size(delta, 4)
        for f = 1:size (W, 4)
            for ch = 1:size(W, 3)
                tmp = conv2 (delta(:, :, f, n), W(:, :, ch, f), 'full');
                down_delta(:, :, ch, n) = down_delta(:, :, ch, n) + tmp (pad + 1: pad + width, pad + 1: pad + height);
                grad_W (:, :, ch, f) = grad_W (:, :, ch, f) + conv2(input(:, :, ch, n), rot90(rot90(delta(:, :, ch, n))), 'valid');
            end
        end
    end
    grad_b =  reshape(sum(sum (sum (delta, 4), 2), 1), [size(W, 4), 1]);
end