function output = nnconv(input, kernel_size, num_output, W, b, pad)
    % Your codes here
    % hint:
    %     1. first pad zeros on the input's each side
    %     2. convolve input with W
    %           notice the output of j-th filter in W convolved with inpuint
    %           correspond to the j-th channel in output
    %     3. don't forget adding bias
    %
    % ps: there are more than one way in step 2, try to find the fastest method
    for i = 1:pad
        input = pad_border (input);
    end
    output = zeros(size(input, 1) - kernel_size + 1, size(input, 2) - kernel_size + 1, num_output, size(input, 4));
    [height, width, channel, number] = size(output);
    for n = 1:size(input, 4)
        input_case = input(:, :, :, n);
        for f = 1:num_output
            w_f = W(:, :, :, f);
            lhs = zeros (size(output, 1), size(output, 2), size(input, 3));
            for ch = 1:size(input_case,3)
                lhs(:, :, ch) = conv2 (input_case(:, :, ch), w_f(:, :, ch), 'valid');
            end
            output(:, :, f, n) = sum (lhs, 3) + ones(height, width) * b(f);
        end
    end
end

function output = pad_border (input)
    zero_column = input (:, 1, :, :) .* 0;
    output = [zero_column input zero_column];
    zero_row = output (1, :, :, :) .* 0;
    output = [zero_row; output; zero_row];
%     disp(size(input));
%     disp(size(output));
end

function output = conv_case (input, w, b)
    output = input;
end